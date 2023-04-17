from typing import List
import xarray as xr


def fillnans(
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


