import autoroot
import hydra
import xarray as xr
from loguru import logger
from omegaconf import OmegaConf
from oceanbench._src.geoprocessing.spatial import transform_360_to_180
from oceanbench._src.preprocessing.alongtrack import remove_swath_dimension
from pathlib import Path


@hydra.main(config_path='config', config_name='main', version_base='1.2')
def main(cfg):

    # OPEN EXPERIMENT DATASET
    logger.info(f"Creating {cfg.experiment.name.upper()} Dataset...")

    ds = hydra.utils.instantiate(cfg.experiment.data)

    logger.debug(f"Dataset size: {ds.ssh.size:,}")

    logger.info(f"Saving {cfg.experiment.name.upper()} Dataset...")
    save_name = Path(cfg.staging_dir).joinpath(f"{cfg.experiment.name}.nc")
    logger.debug(f"{save_name}")
    
    ds.to_netcdf(save_name)
    
    # out_nadir = xr.open_mfdataset(
    #     cfg.preprocess.nadir1.data, preprocess=preprocess_nadir, combine="nested", engine="netcdf4", concat_dim="time"
    # )
    
    # logger.info(f"Saving NADIR 1 Dataset...")
    # out_nadir.to_netcdf(cfg.preprocess.nadir1.saved_model)
    # logger.info(f"Done...!")
    
    # # OPEN NADIR 4 Datasets
    # logger.info(f"Creating NADIR 4 Dataset...")
    
    # out_nadir = xr.open_mfdataset(
    #     cfg.preprocess.nadir4.data, preprocess=preprocess_nadir, combine="nested", engine="netcdf4", concat_dim="time"
    # )

    
    # logger.info(f"Saving NADIR 4 Dataset...")
    # out_nadir.to_netcdf(cfg.preprocess.nadir4.saved_model)
    # logger.info(f"Done...!")

    # print(out_nadir)


    # # OPEN NADIR 5 Datasets
    # logger.info(f"Creating NADIR 5 Dataset...")
    
    # out_nadir = xr.open_mfdataset(
    #     cfg.preprocess.nadir5.data, preprocess=preprocess_nadir, combine="nested", engine="netcdf4", concat_dim="time"
    # )

    
    # logger.info(f"Saving NADIR 5 Dataset...")
    # out_nadir.to_netcdf(cfg.preprocess.nadir5.saved_model)
    # logger.info(f"Done...!")

    # print(out_nadir)
    
    
    
    # def preprocess_swot(da):

    #     da = da.rename({cfg.preprocess.swot.variable: "ssh"})
        
    #     da = remove_swath_dimension(da, "nC")

    #     da = da.sel(
    #         time=hydra.utils.instantiate(cfg.preprocess.time),
    #         drop=True
    #     ).compute()

    #     da["lon"] = transform_360_to_180(da["lon"])
        
    #     da = hydra.utils.instantiate(cfg.preprocess.lat)(da)
    #     da = hydra.utils.instantiate(cfg.preprocess.lon)(da)
        
    #     da = da.sortby("time")

    #     da = da[["lon", "lat", "ssh"]]
        
    #     return da
    
    # # OPEN NADIR Datasets
    # logger.info(f"Creating SWOT Dataset...")
    
    # out_swot = xr.open_mfdataset(
    #     cfg.preprocess.swot.data, preprocess=preprocess_swot, combine="nested", engine="netcdf4", concat_dim="time"
    # )


    # print(out_swot)
    
    # logger.info(f"Saving SWOT Dataset...")
    # out_swot.to_netcdf(cfg.preprocess.swot.saved_model)
    # logger.info(f"Done...!")
    
    
    # logger.info(f"Creating combined dataset...")
    # ds_swotnadir = xr.concat([out_nadir, out_swot], dim="time")
    # ds_swotnadir = ds_swotnadir.sortby("time")

    # print(ds_swotnadir)
    
    # logger.info(f"Saving SWOT1NADIR5 Dataset...")
    # ds_swotnadir.to_netcdf(cfg.preprocess.swot1nadir5.saved_model)
    # logger.info(f"Done...!")
    
    
    

if __name__ == "__main__":
    main()