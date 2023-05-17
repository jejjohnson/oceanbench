from typing import List
import xarray as xr

def rmse_ds(
    ds: xr.Dataset, 
    target: str,
    reference: str,
    dim: List[str],
) -> xr.Dataset:
    return ((ds[target] - ds[reference]) ** 2).mean(dim=dim) ** 0.5


def nrmse_ds(
    ds: xr.Dataset,
    target: str,
    reference: str,
    dim: List[str],
) -> xr.Dataset:
    rmse = rmse_ds(ds=ds, target=target, reference=reference, dim=dim)
    std = (ds[reference]**2).mean(dim=dim) ** 0.5 
    return 1.0 - (rmse / std)



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
