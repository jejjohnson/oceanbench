import autoroot
import hydra
from loguru import logger
from pathlib import Path
from omegaconf import OmegaConf
import viz
import preprocess
import utils

import xarray as xr
import pint_xarray
import pandas as pd
pd.set_option('display.precision', 4)
from oceanbench._src.geoprocessing.gridding import grid_to_regular_grid
from oceanbench._src.metrics.utils import find_intercept_1D, find_intercept_2D


def add_units(da):
    da = da.pint.quantify(
        {"ssh": "meter", 
         "lon": "degrees_east", 
         "lat": "degrees_north",
         "time": "seconds"
        }
    )
    return da


@hydra.main(config_path='config', config_name='main', version_base='1.2')
def main(cfg):
    
    # OPEN DATASETS
    logger.info(f"Loading reference datasets...")
    ds_ref = hydra.utils.instantiate(cfg.reference.data).compute()
    logger.info(f"Loading results datasets...")
    ds = hydra.utils.instantiate(cfg.results.data).compute()

    results = list()
    results.append(cfg.results.name.upper())

    
    # REGRIDDING
    logger.info(f"Regridding...")
    ds = grid_to_regular_grid(
        src_grid_ds=ds.pint.dequantify(),
        tgt_grid_ds=ds_ref.pint.dequantify(), keep_attrs=False
    )
    ds["time"] = ds_ref["time"]

    # INTERPOLATE NANS
    logger.info(f"Interpolating nans...")
    ds = hydra.utils.instantiate(cfg.evaluation.fill_nans)(ds)
    ds_ref = hydra.utils.instantiate(cfg.evaluation.fill_nans)(ds_ref)
    
    # ADD UNITS
    ds = add_units(ds)
    ds_ref = add_units(ds_ref)


    # CALCULATE PHYSICAL VARIABLE
    
    # RESCALING
    logger.info(f"Rescaling Space...")
    ds = hydra.utils.instantiate(cfg.evaluation.rescale_space)(ds)
    ds_ref = hydra.utils.instantiate(cfg.evaluation.rescale_space)(ds_ref)
    
    logger.info(f"Rescaling Time...")
    ds = hydra.utils.instantiate(cfg.evaluation.rescale_time)(ds)
    ds_ref = hydra.utils.instantiate(cfg.evaluation.rescale_time)(ds_ref)
    
    # CALCULATING STATISTICS
    logger.info(f"Calculating nrmse...")
    nrmse_mu = hydra.utils.instantiate(cfg.evaluation.nrmse_spacetime)(da=ds.pint.dequantify(), da_ref=ds_ref.pint.dequantify())
    results.append(nrmse_mu.values)
    
    nrmse_std = hydra.utils.instantiate(cfg.evaluation.nrmse_space)(da=ds.pint.dequantify(), da_ref=ds_ref.pint.dequantify()).std()
    results.append(nrmse_std.values)
    
    logger.info(f"nrmse (mean) : {nrmse_mu.values:.2f}")
    logger.info(f"nrmse (temporal variance): {nrmse_std.values:.2f}")
        
    # PSD ISOTROPIC SCORE
    logger.info(f"Calculating Isotropic PSD...")
    ds_psd_score_iso = hydra.utils.instantiate(
        cfg.evaluation.psd_isotropic_score
    )(da=ds.pint.dequantify(), da_ref=ds_ref.pint.dequantify())

    logger.info(f"Calculating Resolved Scales...")
    space_rs = find_intercept_1D(
        y=1./ds_psd_score_iso.ssh.freq_r.values,
        x=ds_psd_score_iso.ssh.values,
        level=0.5
    )
    results.append(space_rs/1e3)
    results.append(space_rs/1e3/111)
    
    logger.info(f"(Isotropic) Spatial Resolved Scale: {space_rs/1e3:.2f} [km]")
    logger.info(f"(Isotropic) Spatial Resolved Scale: {space_rs/1e3/111:.2f} [degrees]")
    
    
    # PSD (SPACETIME) SCORE
    logger.info(f"Calculating SpaceTime PSD...")
    ds_psd_score_st = hydra.utils.instantiate(
        cfg.evaluation.psd_spacetime_score
    )(da=ds.pint.dequantify(), da_ref=ds_ref.pint.dequantify())

    logger.info(f"Calculating Resolved Scales...")
    lon_rs, time_rs = find_intercept_2D(
        x=1./ds_psd_score_st.ssh.freq_lon.values,
        y=1./ds_psd_score_st.ssh.freq_time.values, 
        z=ds_psd_score_st.ssh.values,
        levels=0.5
    )
    results.append(lon_rs/1e3)
    results.append(lon_rs/1e3/111)
    results.append(time_rs)
    
    logger.info(f"Spatial Resolved Scale: {lon_rs/1e3:.2f} [km]")
    logger.info(f"Spatial Resolved Scale: {lon_rs/1e3/111:.2f} [degrees]")
    logger.info(f"Time Resolved Scale: {time_rs:.2f} [days]")
    
    logger.info(f"Creating Leaderboard...")
    data = [results]
    
    Leaderboard = pd.DataFrame(
        data,
        columns = [
            "Method", 
            "µ(RMSE)",
            "σ(RMSE)",
            "iso λx [km]",
            "iso λx [degree]",
            "λx [km]",
            "λx [degree]",
            "λt [days]",
        ]
    )
    
    print(Leaderboard.T)
    



if __name__ == "__main__":
    main()