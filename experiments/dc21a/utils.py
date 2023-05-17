import autoroot
import hydra
import numpy as np
import xarray as xr
from oceanbench._src.geoprocessing.spatial import transform_360_to_180
from omegaconf.errors import ConfigAttributeError
import pint_xarray


def preprocess_nadir(process_cfg, variable):

    def preprocess(da):

        da["ssh"] = da["sla_unfiltered"] + da["mdt"] - da["lwe"]

        # fixed coordinate names
        da = correct_names(da)

        # subset time dimensions
        try:
            da = da.sel(
                time=hydra.utils.instantiate(process_cfg.time),
                drop=True
            ).compute()
        except ConfigAttributeError:
            pass
        try:
            da = hydra.utils.instantiate(process_cfg.lat)(da)
        except ConfigAttributeError:
            pass
        try:
            da = hydra.utils.instantiate(process_cfg.lon)(da)
        except ConfigAttributeError:
            pass
            
        da = da[variable]


        return da
    
    return preprocess

def correct_names(da):

    try:
        # fixed coordinate names
        da = da.rename({"longitude": "lon", "latitude": "lat"})
    except:
        pass
    
    da["ssh"].attrs["long_name"] = "Sea Surface Height"
    da["ssh"].attrs["standard_name"] = "sea_surface_height"

    da["lat"].attrs["long_name"] = "Latitude"
    da["lat"].attrs["standard_name"] = "latitude"

    da["lon"].attrs["long_name"] = "Longitude"
    da["lon"].attrs["standard_name"] = "longitude"

    da["lon"] = transform_360_to_180(da.lon)
    
    return da


def open_ssh_results(file, variable, preprocess):

    da = xr.open_dataset(file, decode_times=True)

    da = da.sortby("time")

    # fixed coordinate names
    da = correct_names(da)

    da = da.rename({variable: "ssh"})


    # subset time dimensions
    try:
        da = da.sel(
            time=hydra.utils.instantiate(preprocess.time),
            drop=True
        ).compute()
    except ConfigAttributeError:
        pass

    try:
        da = hydra.utils.instantiate(preprocess.lat)(da)
    except ConfigAttributeError:
        pass

    try:
        da = hydra.utils.instantiate(preprocess.lon)(da)
    except ConfigAttributeError:
        pass
    return da


def ts_stats(da, cfg, variable: str, variable_interp: str):

    # resample time series
    da = hydra.utils.instantiate(cfg.postprocess.rescale_time)(da).pint.dequantify()

    # calculate statistics
    rmse = np.sqrt(((da[variable] - da[variable_interp])**2).mean())

    rmse_alongtrack = np.sqrt((da[variable]**2).mean())

    rmse_score = 1. - rmse/rmse_alongtrack

    std_rmse = np.sqrt(((da[variable] - da[variable_interp])**2).std())


    return rmse_score.values, rmse.values, std_rmse.values