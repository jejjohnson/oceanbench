from typing import List, Dict, Optional
import xarray as xr


def check_lists_equal(list_1: List, list_2: List):
    msg = f"Lists not equal...: \n{list_1}\n{list_2}"
    assert sorted(list_1) == sorted(list_2), msg
    

def check_lists_subset(list_1: List, list_2: List):
    msg = f"Lists not subset...: \n{list_1}\n{list_2}"
    assert set(list_1) <= set(list_2), msg
    

def get_dims_xrda(da: xr.DataArray) -> Dict[str, int]:
    return dict(zip(da.dims, da.shape))


def update_dict_xdims(da: xr.DataArray, dims: Dict={}) -> Dict:
    
    update_dims = {f"{idim}":1 for idim in da.dims if idim not in list(dims.keys())}
    dims = {**dims, **update_dims}
    
    check_lists_equal(list(da.dims), list(dims.keys()))

    # dims = sorted(dims.items(), key=lambda x: da.dims())
    
    return dims


def update_dict_keys(source: Dict[str, int], new: Dict[str, int]) -> Dict[str, int]:
    """Updates new dictionary with keys from source
    
    Args:
        source (Dict[str, int]): the source dictionary
        new (Dict[str, int]): the new dictionary
    
    Returns:
        update (Dict[str, int]): the updated dictionary
        
    Examples:
        >>> source = {"x": 1, "y": 100}
        >>> new = {"x": 100}
        >>> update = update_dict_keys(source, new)
        >>> update
        {"x": 100, "y": 1}
    """
    update = {f"{ikey}": 1 for ikey in source.keys() if ikey not in list(new.keys())}

    update = {**new, **update}

    msg = f"Dict keys not same...: \n{source.keys()}\n{update.keys()}"
    assert source.keys() == update.keys(), msg

    return update


def get_xrda_size(da: xr.DataArray, patches: Dict[str, int], strides: Dict[str, int]) -> Dict[str, int]:
    
    da_dims = get_dims_xrda(da)
    
    check_lists_equal(list(da_dims.keys()), list(patches.keys()))
    check_lists_equal(list(da_dims.keys()), list(strides.keys()))
    
    dim_size = {}
    
    for dim in patches:
        dim_size[dim] = max((da_dims[dim] - patches[dim]) // strides[dim] + 1, 0)
    
    return dim_size


def get_patches_size(dims: Dict[str, int], patches: Dict[str, int], strides: Dict[str, int]) -> Dict[str, int]:

    patches = update_dict_keys(dims, patches)
    strides = update_dict_keys(dims, strides)
    
    dim_size = {}
    
    for idim in patches:
        dim_size[idim] = max((dims[idim] - patches[idim]) // strides[idim] + 1, 0)
    
    return dim_size
