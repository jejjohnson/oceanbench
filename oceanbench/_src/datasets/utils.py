from typing import List, Dict, Tuple, NamedTuple, Optional, Iterable, OrderedDict
import xarray as xr
import numpy as np
from functools import reduce
from tqdm import tqdm


def check_lists_equal(list_1: List, list_2: List):
    msg = f"Lists not equal...: \n{list_1}\n{list_2}"
    assert sorted(list_1) == sorted(list_2), msg
    

def check_lists_subset(list_1: List, list_2: List):
    msg = f"Lists not subset...: \n{list_1}\n{list_2}"
    assert set(list_1) <= set(list_2), msg
    

def get_dims_xrda(da: xr.DataArray) -> Dict[str, int]:
    return OrderedDict(zip(da.dims, da.shape))


def update_dict_xdims(da: xr.DataArray, dims: Dict={}, default_size: bool=True) -> OrderedDict:
    """Updates a dims dictionary based on an xr.DataArray.
    This is useful when ensuring the dims dictionary has the same
    keys (and maybe values) as the dims in the xr.DataArray.
    If a key is present in the xr.DataArray but not the dims dict,
    the default_size says whether to set the value of said key to the
    xr.DataArray or we set it to 1. In the patch framework, an
    unspecified key value should have the same dimensions as the
    xr.DataArray. However, for strides, an unspecified key value
    should be 1. Any keys in the dims dict that are NOT present
    in the xr.DataArray are discarded.

    Args:
        da (xr.DataArray): the xr.DataArray with the dimensions
        dims (Optional[Dict]): a dictionary of dims. Defaults to {}.
        default_size (Optional[bool]): determines which value to choose
          if the xr.DataArray has a dimension that isn't in the dims
          dictionary. True means that we use the value from the
          xr.DataArray and False means we set it to 1.
          Defaults to True.

    Returns:
        dims (OrderedDict): An ordered dictionary with all the keys
          (and maybe values) as the xr.DataArray dims and any extra keys
          specified in the dims.
    """
    update = OrderedDict()

    for idim in da.dims:
        if idim not in list(dims.keys()):
            if default_size:
                update[idim] = da[idim].shape[0]
            else:
                update[idim] = 1
        else:
            update[idim] = dims[idim]

    assert set(da.dims) == update.keys()
    return update


def update_dict_keys(source: Dict[str, int], new: Dict[str, int], default: bool=True) -> OrderedDict[str, int]:
    """Updates new dictionary with keys from source
    
    Args:
        source (Dict[str, int]): the source dictionary
        new (Dict[str, int]): the new dictionary
        default (bool): initialize values of new dict with values of old dict
            (detault=True)
    
    Returns:
        update (Dict[str, int]): the updated dictionary
        
    Examples:
        >>> source = {"x": 1, "y": 100}
        >>> new = {"x": 100}
        >>> update = update_dict_keys(source, new)
        >>> update
        {"x": 100, "y": 1}
    """

    update = OrderedDict()

    for ikey in source.keys():
        if ikey not in new.keys():
            if default:
                update[ikey] = source[ikey]
            else:
                update[ikey] = 1
        else:
            update[ikey] = new[ikey]

    msg = f"Dict keys not same...: \n{source.keys()}\n{update.keys()}"

    assert source.keys() == update.keys(), msg

    return update


def get_xrda_size(da: xr.DataArray, patches: Dict[str, int], strides: Dict[str, int]) -> OrderedDict[str, int]:
    
    da_dims = get_dims_xrda(da)
    
    check_lists_equal(list(da_dims.keys()), list(patches.keys()))
    check_lists_equal(list(da_dims.keys()), list(strides.keys()))
    
    dim_size = {}
    
    for dim in patches:
        dim_size[dim] = max((da_dims[dim] - patches[dim]) // strides[dim] + 1, 0)
    
    return dim_size


def get_patches_size(dims: Dict[str, int], patches: Dict[str, int], strides: Dict[str, int]
                     ) -> Tuple[OrderedDict[str, int], OrderedDict[str, int], OrderedDict[str, int]]:

    patches = update_dict_keys(dims, patches, default=True)
    strides = update_dict_keys(dims, strides, default=False)
    
    dim_size = OrderedDict()
    
    for idim in patches:
        dim_size[idim] = max((dims[idim] - patches[idim]) // strides[idim] + 1, 0)
    
    return dim_size, patches, strides


def get_slices(idx: int, da_size: Dict[str, int], patches: Dict[str, int], strides: Dict[str, int]) -> OrderedDict[str, slice]:
    slices = {
        dim: slice(strides[dim] * idx,
                   strides[dim] * idx + patches[dim]) for dim, idx in zip(
            da_size.keys(), np.unravel_index(idx, tuple(da_size.values()))
        )
    }
    return slices
