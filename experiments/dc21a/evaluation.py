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
import numpy as np

from oceanbench._src.geoprocessing.gridding import grid_to_coord_based
from oceanbench._src.preprocessing.alongtrack import select_track_segments
import oceanbench._src.metrics.power_spectrum as psd_calc
from oceanbench._src.metrics.utils import find_intercept_1D
from oceanbench._src.geoprocessing.subset import where_slice


@hydra.main(config_path='config', config_name='main', version_base='1.2')
def main(cfg):

    # OPEN DATASETS
    logger.info(f"Loading result dataset...")
    ds_map = hydra.utils.instantiate(cfg.results.data)

    # OPEN Datasets
    logger.info(f"Loading alongtrack dataset...")
    ds_alongtrack = hydra.utils.instantiate(cfg.reference.data)

    results = list()
    results.append(cfg.results.name.upper())

    logger.info(f"Regriding map to alongtrack...")
    ds_alongtrack["ssh_interp"] = grid_to_coord_based(
        ds_map.transpose("lon", "lat", "time"),
        ds_alongtrack,
        data_vars=["ssh"], 
    )["ssh"]

    # drop all nans
    logger.info(f"Selecting variables and dropping NANs...")
    ds_results = ds_alongtrack[["ssh", "ssh_interp"]].dropna(dim="time").copy()
    ds_results["ssh"] = ds_results.ssh.astype(np.float64)
    ds_results["ssh_interp"] = ds_results.ssh_interp.astype(np.float64)

    # Statistically Significant
    ds_results = where_slice(
        ds_results, 
        "lon", 
        cfg.domain.lon.min_val + 0.25, 
        cfg.domain.lon.max_val - 0.25
    )
    ds_results = where_slice(
        ds_results, 
        "lat", 
        cfg.domain.lat.min_val + 0.25, 
        cfg.domain.lat.max_val - 0.25
    )

    logger.info(f"Calculating nrmse...")
    nrmse, rmse_mu, rmse_std = utils.ts_stats(ds_results, cfg, "ssh", "ssh_interp")
    results.append(nrmse)
    results.append(rmse_mu)
    results.append(rmse_std)

    logger.info(f"nrmse: {nrmse:.2f}")
    logger.info(f"rmse (mean) : {rmse_mu:.2f}")
    logger.info(f"rmse (temporal variance): {rmse_std:.2f}")

    # gather segments
    logger.info(f"Gathering AlongTrack segments...")
    ds_segments = hydra.utils.instantiate(cfg.postprocess.alongtrack_segments)(ds=ds_results)

    # PSD Score
    logger.info(f"Calculating Isotropic PSD...")
    delta_x = ds_segments.delta_x
    nperseg = ds_segments.nperseg
    ds_psd_score = hydra.utils.instantiate(cfg.evaluation.psd_score)(da=ds_segments, delta_x=delta_x, nperseg=nperseg).score
    
    # Resolved Spatial Scale
    space_rs = find_intercept_1D(
        y=1./(ds_psd_score.wavenumber.values+1e-15),
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
            "nRMSE",
            "RMSE (µ)",
            "RMSE (σ)",
            "λx [km]",
            "λx [degree]",
        ]
    )

    # save_path = Path(cfg.metrics_csv).joinpath("leaderboard.csv")

    # logger.info(f"Saving Results: \n{save_path}")

    # if cfg.overwrite_results or not save_path.is_file():
    #     logger.info(f"Overwriting...")
    #     Leaderboard.to_csv(save_path, mode="w", header=True)
    # else:
    #     logger.info(f"Creating new file...")
    #     header = False  if save_path.is_file() else False
    #     Leaderboard.to_csv(save_path, mode="a", header=header)
        
    
    print(Leaderboard.T)

if __name__ == "__main__":
    main()