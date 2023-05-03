from typing import List
import xarray as xr


def rmse_da(
    da: xr.Dataset, 
    da_ref: xr.Dataset, 
    variable: str, 
    dim: List[str],
) -> xr.Dataset:
    return ((da[variable] - da_ref[variable]) ** 2).mean(dim=dim) ** 0.5


def nrmse_da(
    da: xr.Dataset, 
    da_ref: xr.Dataset, 
    variable: str, 
    dim: List[str],
) -> xr.Dataset:
    rmse = rmse_da(da=da, da_ref=da_ref, variable=variable, dim=dim)
    std = (da_ref[variable]**2).mean(dim=dim) ** 0.5 
    return 1.0 - (rmse / std)
