from typing import List, Dict, Tuple, NamedTuple, Optional, Iterable
from collections import OrderedDict
import xarray as xr
import numpy as np
from functools import reduce
from tqdm import tqdm
# from oceanbench._src.datasets.base import XArrayDataset


# class DAPatchParams(NamedTuple):
#     patches: Dict[str, int]
#     strides: Dict[str, int]
#     da_dims: Dict[str, int]
#     da_size: Dict[str, int]
    
#     def __init__(self, 
#                  da: xr.DataArray,
#                  patches: Optional[Dict[str, int]],
#                  strides: Optional[Dict[str, int]],
#                 ):
        
#         self.da_dims = get_dims_xrda(da)
        
#         self.da_size, self.patches, self.strides = get_patches_size(
#             dims=self.da_dims,
#             patches=patches if patches is not None else {},
#             strides=strides if strides is not None else {},
#         )
    
#     def get_da_slice(self, idx, da: xr.DataArray) -> xr.DataArray:
        
#         slices = get_slices(idx, 
#                           da_size=self.da_size, 
#                           patches=self.patches, 
#                           strides=self.strides
#                          )
#         return da.isel(**slices)
    
#     def num_items(self):
#         return list_product(self.da_size.values)


def check_lists_equal(list_1: List, list_2: List):
    msg = f"Lists not equal...: \n{list_1}\n{list_2}"
    assert sorted(list_1) == sorted(list_2), msg
    

def check_lists_subset(list_1: List, list_2: List):
    msg = f"Lists not subset...: \n{list_1}\n{list_2}"
    assert set(list_1) <= set(list_2), msg
    

def get_dims_xrda(da: xr.DataArray) -> Dict[str, int]:
    return OrderedDict(zip(da.dims, da.shape))


def update_dict_xdims(da: xr.DataArray, dims: Dict={}, default_size: bool=True) -> Dict:

    if default_size:
        update_dims = {f"{idim}":da[idim].shape[0] for idim in da.dims if idim not in list(dims.keys())}
    else:
        update_dims = {f"{idim}":1 for idim in da.dims if idim not in list(dims.keys())}
    
    dims = {**dims, **update_dims}
    
    # check_lists_equal(list(da.dims), list(dims.keys()))
    assert set(da.dims) == dims.keys()
    return dims


def update_dict_keys(source: Dict[str, int], new: Dict[str, int], default: bool=True) -> Dict[str, int]:
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
#     if default:
#         update = OrderedDict()
        
#         for ikey in source.keys():
#             if ikey not in new.keys():
#                 update[ikey] = source[ikey]
#             else:
#                 update[ikey] = new[ikey]
                
#         # update = {f"{ikey}": source[ikey] for ikey in source.keys() if ikey not in list(new.keys())}
#     else:
    update = OrderedDict()

    for ikey in source.keys():
        if ikey not in new.keys():
            if default:
                update[ikey] = source[ikey]
            else:
                update[ikey] = 1
        else:
            update[ikey] = new[ikey]
                
        # update = {f"{ikey}": 1 for ikey in source.keys() if ikey not in list(new.keys())}

    # update = {**new, **update}

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


def get_patches_size(dims: Dict[str, int], patches: Dict[str, int], strides: Dict[str, int]
                     ) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:

    patches = update_dict_keys(dims, patches, default=True)
    strides = update_dict_keys(dims, strides, default=False)
    
    dim_size = OrderedDict()
    
    for idim in patches:
        dim_size[idim] = max((dims[idim] - patches[idim]) // strides[idim] + 1, 0)
    
    return dim_size, patches, strides


def get_slices(idx: int, da_size: Dict[str, int], patches: Dict[str, int], strides: Dict[str, int]) -> Dict[str, slice]:
    slices = {
        dim: slice(strides[dim] * idx,
                   strides[dim] * idx + patches[dim]) for dim, idx in zip(
            da_size.keys(), np.unravel_index(idx, tuple(da_size.values()))
        )
    }
    return slices


def list_product(items: List[int]) -> int:
    return reduce((lambda x, y: x * y), [1] + items)


def reconstruct_from_items(
    items: Iterable, 
    dims_label: Iterable[str], 
    xrda_batcher,
    weight=None
):
    msg = f"Length of dim labels does not match length of non-batch dims."
    msg += f"\nDims Label: {dims_label} \nShape: {items[0].shape}"
    assert len(dims_label) == len(items[0].shape), msg
    
    coords = xrda_batcher.get_coords()
    
    # check for subset of coordinate arrays
    coords_labels = list(coords[0].dims.keys())
    check_lists_subset(coords_labels, dims_label)
    items_shape = dict(zip(dims_label, items[0].shape))
    
    
    patch_values = list(xrda_batcher.patches.values())
    patch_names = list(xrda_batcher.patches.keys())
    
    # (maybe) update weight matrix
    if weight is None:
        weight = np.ones(patch_values)
    else:
        msg = "Weight array is not the same size as patch dims "
        msg += "or not the same value"
        msg += f"\nWeight: {list(weight.shape)} | Patches: {patch_values}"
        assert len(weight.shape) == len(xrda_batcher.patches), msg
        assert any(list(ishape == ivalue for ishape, ivalue in zip(weight.shape, patch_values))), msg
                    
    w = xr.DataArray(weight, dims=patch_names)
    
    # create data arrays from (maybe) coords
    das = [
        xr.DataArray(it, dims=dims_label, coords=co.coords)
        for it, co in zip(items, coords)
          ]
    
    msg = "New Data Array is not the same size as items"
    msg += "or not the same value"
    msg += f"\n{das[0].shape} | Items: {items_shape}"
    assert len(das[0].shape) == len(items_shape), msg
    assert set(das[0].shape) == set(items_shape.values()), msg
    
    # get new shape from 
    new_shape = {
        idim: xrda_batcher.da[idim].shape[0] if idim in coords_labels
        else items[0].shape[i]
        for i, idim in enumerate(dims_label) 
    }
    
    rec_da = xr.DataArray(
        np.zeros([*new_shape.values()]),
        dims=dims_label,
        coords={d: xrda_batcher.da[d] for d in xrda_batcher.patches}
    )
    count_da = xr.zeros_like(rec_da)
    
    for ida in tqdm(das):
        icoord = ida.coords
        rec_da.loc[ida.coords] = rec_da.sel(ida.coords) + ida * w
        count_da.loc[ida.coords] = count_da.sel(ida.coords) + w
        
    return rec_da / count_da
