import autoroot
import hydra
from loguru import logger
from pathlib import Path
from omegaconf import OmegaConf
import functools as ft
import xarray as xr
import pint_xarray
import pandas as pd
pd.set_option('display.precision', 4)
import utils

from oceanbench._src.geoprocessing.gridding import grid_to_coord_based
from oceanbench._src.preprocessing.alongtrack import select_track_segments
import oceanbench._src.metrics.power_spectrum as psd_calc
from oceanbench._src.metrics.utils import find_intercept_1D
from oceanbench._src.geoprocessing.subset import where_slice


@hydra.main(config_path='config', config_name='main', version_base='1.2')
def main(cfg):

    # OPEN DATASETS
    logger.info(f"Loading result dataset...")
    # da_ref = open_ssh_reference(cfg.reference.data)
    da_map = utils.open_ssh_results(cfg.results.data, cfg.results.variable, cfg.postprocess)

    postprocess = utils.preprocess_nadir(process_cfg=cfg.postprocess, variable=cfg.postprocess.nadir.variable)

    logger.info(f"Loading alongtrack dataset...")
    da_alongtrack = xr.open_mfdataset(
        cfg.postprocess.nadir.data,
        preprocess=postprocess,
        combine="nested",
        engine="netcdf4",
        concat_dim="time"
    )

    da_alongtrack = da_alongtrack.sortby("time").compute().to_dataset(name="ssh")

    results = list()
    results.append(cfg.results.name.upper())

    logger.info(f"Regriding map to alongtrack...")
    da_alongtrack["ssh_interp"] = grid_to_coord_based(
        da_map.transpose("lon", "lat", "time"),
        da_alongtrack,
        data_vars=["ssh"], 
    )["ssh"]

    # drop all nans
    logger.info(f"Selecting variables and dropping NANs...")
    ds_results = da_alongtrack[["ssh", "ssh_interp"]].dropna(dim="time")

    ds_results = where_slice(ds_results, "lon", -64.975 - 0.25, -55.007 + 0.25)
    ds_results = where_slice(ds_results, "lat", 33.025 - 0.25, 42.9917 + 0.25)

    logger.info(f"Calculating nrmse...")

    nrmse_mu, nrmse_std = utils.ts_stats(ds_results, cfg, "ssh", "ssh_interp")
    results.append(nrmse_mu)
    results.append(nrmse_std)

    logger.info(f"nrmse (mean) : {nrmse_mu:.2f}")
    logger.info(f"nrmse (temporal variance): {nrmse_std:.2f}")

    # gather segments
    logger.info(f"Gathering AlongTrack segments...")
    ds_segments = hydra.utils.instantiate(cfg.evaluation.alongtrack_segments)(ds=ds_results)


    # PSD Score
    logger.info(f"Calculating Isotropic PSD...")
    delta_x = ds_segments.ssh.delta_x
    nperseg = ds_segments.ssh.nperseg

    # ds_psd_score = hydra.utils.instantiate(cfg.evaluation.psd_score)(da=ds_segments, delta_x=delta_x, nperseg=nperseg).score
    ds_psd_score = psd_calc.psd_welch_score(ds_segments, "ssh", "ssh_interp", delta_x=delta_x, nperseg=nperseg).score

    space_rs = find_intercept_1D(
        y=1./(ds_psd_score.wavenumber.values+1e-10),
        x=ds_psd_score.values,
        level=0.5
    )

    results.append(space_rs)
    results.append(space_rs/111)

    logger.info(f"Spatial Resolved Scale: {space_rs:.2f} [km]")

    Leaderboard = pd.DataFrame(
        [results],
        columns = [
            "Method", 
            "µ(RMSE)",
            "σ(RMSE)",
            "λx [km]",
            "λx [degree]",
        ]
    )
    
    print(Leaderboard.T)

if __name__ == "__main__":
    main()