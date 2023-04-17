from typing import List
import xarray as xr
from oceanbench._src.geoprocessing.spectral import psd_isotropic, psd_spacetime
from oceanbench._src.preprocessing.mean import xr_cond_average


def psd_isotropic_error(
    da: xr.DataArray, 
    da_ref: xr.DataArray, 
    variable: str,
    psd_dims: List[str],
    avg_dims: List[str],
) -> xr.DataArray:
    psd_error = psd_isotropic(ds=da_ref - da, variable=variable, dims=psd_dims)
    return xr_cond_average(psd_error, dims=avg_dims, drop=True)

def psd_isotropic_score(
    da: xr.DataArray, 
    da_ref: xr.DataArray, 
    variable: str,
    psd_dims: List[str],
    avg_dims: List[str],
) -> xr.DataArray:

    
    # error
    score = psd_isotropic_error(
        da=da, da_ref=da_ref, variable=variable, psd_dims=psd_dims, avg_dims=avg_dims
    )
    
    # reference signal
    psd_ref = xr_cond_average(
        psd_isotropic(ds=da_ref, variable=variable, dims=psd_dims),
        dims=avg_dims
    )

    # normalized score
    score = 1.0 - (score / psd_ref)
    
    return score