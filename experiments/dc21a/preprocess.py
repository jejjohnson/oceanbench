import autoroot
import hydra
import xarray as xr
from loguru import logger
from omegaconf import OmegaConf
from oceanbench._src.geoprocessing.spatial import transform_360_to_180
from omegaconf.errors import ConfigAttributeError


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
    
    # OPEN NADIR Datasets
    logger.info(f"Creating NADIR Dataset...")
    
    out_nadir = xr.open_mfdataset(
        cfg.preprocess.nadir.data, preprocess=preprocess_nadir, combine="nested", engine="netcdf4", concat_dim="time"
    )

    out_nadir = out_nadir.compute().sortby("time")

    print(out_nadir)

    logger.info(f"Saving NADIR Dataset...")
    out_nadir.to_netcdf(cfg.preprocess.nadir.saved_model)
    logger.info(f"Done...!")

if __name__ == "__main__":
    main()