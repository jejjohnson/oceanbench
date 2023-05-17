import autoroot
import hydra
import xarray as xr
from loguru import logger
from omegaconf import OmegaConf
from oceanbench._src.geoprocessing.spatial import transform_360_to_180
from omegaconf.errors import ConfigAttributeError
from pathlib import Path


@hydra.main(config_path='config', config_name='main', version_base='1.2')
def main(cfg):
    
    def preprocess_nadir(da):

        da["ssh"] = da["sla_unfiltered"] + da["mdt"] - da["lwe"]

        # fixed coordinate names
        da = da.rename({"longitude": "lon", "latitude": "lat"})

        # subset time dimensions
        try:
            da = da.sel(
                time=hydra.utils.instantiate(cfg.preprocess.time),
                drop=True
            ).compute()
        except ConfigAttributeError:
            pass

        da["lon"] = transform_360_to_180(da["lon"])


        try:
            da = hydra.utils.instantiate(cfg.preprocess.lat)(da)
        except ConfigAttributeError:
            pass

        try:
            da = hydra.utils.instantiate(cfg.preprocess.lon)(da)
        except ConfigAttributeError:
            pass
            
        da = da[cfg.preprocess.nadir.variable]

        return da
    
    # OPEN Datasets
    logger.info(f"Creating Dataset...")
    ds = hydra.utils.instantiate(cfg.experiment.data).sortby("time").compute()
    name = ""

    if cfg.grid.regrid:
        logger.info(f"Regridding dataset...")
        ds = hydra.utils.instantiate(cfg.grid.regrid)(ds)
        name += "gridded"
    else:
        name += "alongtrack"

    logger.info(f"Saving Preprocessed Dataset...")
    name += "_" + str(cfg.domain.name)
    save_path = Path(cfg.directories.ml_ready).joinpath(f"{name}.nc")
    logger.debug(f"Saving @:{save_path}")
    ds.to_netcdf(save_path)
    logger.info(f"Done...!")

if __name__ == "__main__":
    main()