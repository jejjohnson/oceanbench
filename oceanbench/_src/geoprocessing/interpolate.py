from typing import List
import xarray as xr
#import pyinterp.backends.xarray
#import pyinterp.fill
#import pyinterp

def fillnan_latlon(
    ds: xr.Dataset, 
    dims: List[str], 
    method: str="slinear", 
    **kwargs
) -> xr.Dataset:
    
    if isinstance(dims, str):
        dims = list(dims)
        
    for idim in dims:
        ds = ds.interpolate_na(dim=idim, method=method, **kwargs)
        ds = ds.interpolate_na(dim=idim, method=method, **kwargs)
    
    return ds


# def interpolate_gauss_seidel(
#     ds: xr.Dataset, 
#     variable: str, 
#     geodetic: bool=False, 
#     increasing_axes=False,
#     **kwargs
# ) -> xr.Dataset:
    
#     ds = ds.copy()
    
#     grid = pyinterp.backends.xarray.Grid3D(ds[variable], geodetic=geodetic, increasing_axes=increasing_axes)
    
#     dims = ds.dims
    
#     has_converged, filled = pyinterp.fill.gauss_seidel(grid, **kwargs)
    
#     ds[variable] = (dims, filled)
    
#     return ds


# def interpolate_loess(
#     ds: xr.Dataset, 
#     variable: str, 
#     geodetic: bool=False, 
#     increasing_axes=False,
#     **kwargs
# ) -> xr.Dataset:
    
#     ds = ds.copy()
    
#     dims = ds.dims
    
#     grid = pyinterp.backends.xarray.Grid3D(ds[variable], geodetic=geodetic, increasing_axes=increasing_axes)
        
#     attrs = ds[variable].attrs
    
#     filled = pyinterp.fill.loess(grid, **kwargs)
    
#     ds[variable] = (dims, filled)
    
#     ds[variable].attrs = attrs
    
#     return ds