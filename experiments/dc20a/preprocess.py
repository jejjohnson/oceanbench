import autoroot
import hydra
import xarray as xr
from loguru import logger
from omegaconf import OmegaConf
from oceanbench._src.geoprocessing.spatial import transform_360_to_180
from oceanbench._src.preprocessing.alongtrack import remove_swath_dimension
from pathlib import Path


    # # OPEN Datasets
    # logger.info(f"Creating Dataset...")
    # ds_ = hydra.utils.instantiate(cfg.experiment.nadir5)
    # print(ds)
    # ds = hydra.utils.instantiate(cfg.experiment.data).sortby("time").compute()
    # name = ""

    # if cfg.grid.regrid:
    #     logger.info(f"Regridding dataset...")
    #     ds = hydra.utils.instantiate(cfg.grid.regrid)(ds)
    #     name += "gridded"
    # else:
    #     name += "alongtrack"

    # logger.info(f"Saving Preprocessed Dataset...")
    # name += "_" + str(cfg.domain.name)
    # save_path = Path(cfg.directories.ml_ready).joinpath(f"{name}.nc")
    # logger.debug(f"Saving @:{save_path}")
    # ds.to_netcdf(save_path)
    # logger.info(f"Done...!")

@hydra.main(config_path='config', config_name='main', version_base='1.2')
def main(cfg):

    # OPEN EXPERIMENT DATASET
    logger.info(f"Creating {cfg.experiment.name.upper()} Dataset...")
    ds = hydra.utils.instantiate(cfg.experiment.data)

    save_name = ""
    

    if cfg.grid.regrid:
        logger.info(f"Regridding dataset...")
        ds = hydra.utils.instantiate(cfg.grid.regrid)(ds)
        save_name += "gridded"
    else:
        save_name += "alongtrack"

    logger.debug(f"Dataset size: {ds.ssh.size:,}")
    logger.debug(f"Dataset shape: {ds.ssh.shape}")

    logger.info(f"Saving {cfg.experiment.name.upper()} Dataset...")
    save_name = Path(cfg.staging_dir).joinpath(f"{save_name}_{cfg.experiment.name}.nc")
    logger.debug(f"{save_name}")
    
    ds.to_netcdf(save_name)
    
    logger.info(f"Done...!")
    
    
    

if __name__ == "__main__":
    main()