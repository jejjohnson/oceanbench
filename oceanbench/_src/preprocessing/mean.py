from typing import Optional
import xarray as xr
from functools import reduce


def xr_cond_average(
    ds: xr.Dataset, dims: Optional[str]=None, drop: bool=True
) -> xr.Dataset:
    """Function to conditionally average dimensions
    
    Args:
        ds (xr.Dataset): the xarray with mixed dimensions
        dims (List[str]): the list of dimensions to average
        drop (bool): option to drop all relevent dimensions
    
    Returns:
        ds (xr.Dataset): the xr.Dataset with averaged non frequency dimensions
    """
    
    # get all dims to be conditioned on
    cond_dims = [idim for idim in list(ds.dims) if idim not in dims]
    
    # create condition 
    if len(cond_dims) > 1:
        cond = reduce(lambda x,y: (ds[x]>0.0) & (ds[y]>0.0), cond_dims)
    else:
        cond = ds[cond_dims[0]] > 0.0
        
    # take mean of remaining dims
    return ds.mean(dim=dims).where(cond, drop=drop)