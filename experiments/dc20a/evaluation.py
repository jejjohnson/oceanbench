import autoroot
import hydra
from loguru import logger
from pathlib import Path
from omegaconf import OmegaConf
import viz
import preprocess
import postprocess
import utils

import xarray as xr
import pint_xarray
import pandas as pd
pd.set_option('display.precision', 4)
from oceanbench._src.geoprocessing.gridding import grid_to_regular_grid
from oceanbench._src.geoprocessing.interpolate import fillnans
from oceanbench._src.geoprocessing.spatial import transform_360_to_180
from oceanbench._src.metrics.utils import find_intercept_1D, find_intercept_2D


def open_ssh_results(file, variable):
    
    da = xr.open_dataset(file, decode_times=True)
    
    da = da.sortby("time")
    
    da = da.rename({variable: "ssh"})
    
    da = da.sel(
        time=slice("2012-10-22", "2012-12-01"),
        lon=slice(-64.975, -55.007),
        lat=slice(33.025, 42.9917),
        drop=True
    )
    
    da = da.resample(time="1D").mean()
    
    return da

def open_ssh_reference(file, variable="gssh"):
    da = xr.open_dataset(file, decode_times=False)
    da["time"] = pd.to_datetime(da.time)
    
    da = da.sortby("time")
    da = da.sel(
        time=slice("2012-10-22", "2012-12-01"),
        lon=slice(-64.975, -55.007),
        lat=slice(33.025, 42.9917),
        drop=True
    )
    da = da.resample(time="1D").mean()
    return da

def correct_names(da):
    
    da["ssh"].attrs["long_name"] = "Sea Surface Height"
    da["ssh"].attrs["standard_name"] = "sea_surface_height"

    da["lat"] = da.lat.pint.quantify("degrees_north")
    da["lat"].attrs["long_name"] = "Latitude"
    da["lat"].attrs["standard_name"] = "latitude"

    da["lon"].attrs["long_name"] = "Longitude"
    da["lon"].attrs["standard_name"] = "longitude"

    da["lon"] = transform_360_to_180(da.lon)
    
    return da

def add_units(da):
    da = da.pint.quantify(
        {"ssh": "meter", 
         "lon": "degrees_east", 
         "lat": "degrees_north",
         "time": "nanoseconds"
        }
    )
    return da


@hydra.main(config_path='config', config_name='main', version_base='1.2')
def main(cfg):
    
    # OPEN DATASETS
    logger.info(f"Loading datasets...")
    da_ref = open_ssh_reference(cfg.reference.data)
    da = open_ssh_results(cfg.results.data, cfg.results.variable)
    results = list()
    results.append(cfg.results.experiment.upper())
    results.append(cfg.results.name.upper())
    
    # CORRECT NAMES
    logger.info(f"Correcting labels...")
    da_ref = correct_names(da_ref)
    da = correct_names(da)
    
    # REGRIDDING
    logger.info(f"Regridding...")
    da = grid_to_regular_grid(
        src_grid_ds=da.pint.dequantify(),
        tgt_grid_ds=da_ref.pint.dequantify(), keep_attrs=True
    )
    
    # INTERPOLATE NANS
    logger.info(f"Interpolating nans...")
    da = hydra.utils.instantiate(cfg.evaluation.fill_nans)(da)
    da_ref = hydra.utils.instantiate(cfg.evaluation.fill_nans)(da_ref)
    
    # ADD UNITS
    da = add_units(da)
    da_ref = add_units(da_ref)
    


    # CALCULATING PHYSICAL VARIABLE
    logger.info(f"Calculating physical variable...")
    logger.info(f"{cfg.postprocess.name.upper()}")
    da = postprocess.calculate_physical_quantities(da, cfg.postprocess.name)
    da_ref = postprocess.calculate_physical_quantities(da_ref, cfg.postprocess.name)
    results.append(cfg.postprocess.name.upper())
    variable = cfg.postprocess.variable
    
    # RESCALING
    logger.info(f"Rescaling Space...")
    da = hydra.utils.instantiate(cfg.evaluation.rescale_space)(da)
    da_ref = hydra.utils.instantiate(cfg.evaluation.rescale_space)(da_ref)
    
    logger.info(f"Rescaling Time...")

    da = hydra.utils.instantiate(cfg.evaluation.rescale_time)(da)
    da_ref = hydra.utils.instantiate(cfg.evaluation.rescale_time)(da_ref)

    
    # CALCULATING STATS
    logger.info(f"Calculating nrmse...")
    nrmse_mu = hydra.utils.instantiate(cfg.evaluation.nrmse_spacetime)(da=da, da_ref=da_ref)
    results.append(nrmse_mu.values)
    
    nrmse_std = hydra.utils.instantiate(cfg.evaluation.nrmse_space)(da=da, da_ref=da_ref).std()
    results.append(nrmse_std.values)
    
    
    
    logger.info(f"nrmse (mean) : {nrmse_mu.values:.2f}")
    logger.info(f"nrmse (temporal variance): {nrmse_std.values:.2f}")
        
    # PSD ISOTROPIC SCORE
    logger.info(f"Calculating Isotropic PSD...")
    da_psd_score_iso = hydra.utils.instantiate(
        cfg.evaluation.psd_isotropic_score
    )(da=da, da_ref=da_ref)
    space_rs = find_intercept_1D(
        y=1./(da_psd_score_iso[variable].freq_r.values+1e-10),
        x=da_psd_score_iso[variable].values,
        level=0.5
    )
    results.append(space_rs/1e3)
    results.append(space_rs/1e3/111)
    
    logger.info(f"(Isotropic) Spatial Resolved Scale: {space_rs/1e3:.2f} [km]")
    logger.info(f"(Isotropic) Spatial Resolved Scale: {space_rs/1e3/111:.2f} [degrees]")
    
    da_psd_score_st = hydra.utils.instantiate(
        cfg.evaluation.psd_spacetime_score
    )(da=da, da_ref=da_ref)
    
    # PSD (SPACETIME) SCORE
    logger.info(f"Calculating SpaceTime PSD...")
    lon_rs, time_rs = find_intercept_2D(
        x=1./da_psd_score_st[variable].freq_lon.values,
        y=1./da_psd_score_st[variable].freq_time.values, 
        z=da_psd_score_st[variable].values,
        levels=0.5
    )
    results.append(lon_rs/1e3)
    results.append(lon_rs/1e3/111)
    results.append(time_rs)
    
    logger.info(f"Spatial Resolved Scale: {lon_rs/1e3:.2f} [km]")
    logger.info(f"Spatial Resolved Scale: {lon_rs/1e3/111:.2f} [degrees]")
    logger.info(f"Time Resolved Scale: {time_rs:.2f} [days]")
    
    data = [results]
    
    columns = [
            "Experiment",
            "Method", 
            "Variable",
            "µ(RMSE)",
            "σ(RMSE)",
            "iso λx [km]",
            "iso λx [degree]",
            "λx [km]",
            "λx [degree]",
            "λt [days]",
        ]
    Leaderboard = pd.DataFrame(data, columns=columns)


    save_path = Path(cfg.metrics_csv).joinpath(f"{cfg.csv_name}.csv")

    logger.info(f"Saving Results: \n{save_path}")

    if cfg.overwrite_results or not save_path.is_file():
        logger.info(f"Overwriting...")
        Leaderboard.to_csv(save_path, mode="w", header=True)
    else:
        logger.info(f"Creating new file...")
        header = False  if save_path.is_file() else False
        Leaderboard.to_csv(save_path, mode="a", header=header)
        
    
    
    print(Leaderboard.T)
    



if __name__ == "__main__":
    main()