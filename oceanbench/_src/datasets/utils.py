from typing import List, Dict, Optional
import xarray as xr


def check_lists_equal(list_1: List, list_2: List):
    msg = f"Lists not equal...: \n{list_1}\n{list_2}"
    assert sorted(list_1) == sorted(list_2), msg
    

def check_lists_subset(list_1: List, list_2: List):
    msg = f"Lists not subset...: \n{list_1}\n{list_2}"
    assert set(list_1) <= set(list_2), msg
    

def get_xrda_dims(da: xr.DataArray) -> Dict[str, int]:
    return dict(zip(da.dims, da.shape))


def update_dict_xdims(da: xr.DataArray, dims: Dict={}) -> Dict:
    
    update_dims = {f"{idim}":1 for idim in da.dims if idim not in list(dims.keys())}
    
    dims = {**dims, **update_dims}
    
    check_lists_equal(list(da.dims), list(dims.keys()))

    # dims = sorted(dims.items(), key=lambda x: da.dims())
    
    return dims


def get_xrda_size(da: xr.DataArray, patches: Dict[str, int], strides: Dict[str, int]) -> Dict[str, int]:
    
    da_dims = get_xrda_dims(da)
    
    check_lists_equal(list(da_dims.keys()), list(patches.keys()))
    check_lists_equal(list(da_dims.keys()), list(strides.keys()))
    
    dim_size = {}
    
    for dim in patches:
        dim_size[dim] = max((da_dims[dim] - patches[dim]) // strides[dim] + 1, 0)
    
    return dim_size
