import typing as tp
from dataclasses import dataclass
import itertools
import numpy as np
import xarray as xr
import tqdm
import torch
from oceanbench._src.utils.exceptions import IncompleteScanConfiguration, DangerousDimOrdering
from oceanbench._src.geoprocessing.select import select_bounds, select_bounds_multiple
from oceanbench._src.utils.custom_dtypes import Bounds
from oceanbench._src.datasets.utils import (
    get_dims_xrda,
    check_lists_equal,
    update_dict_xdims,
    check_lists_subset,
    get_patches_size,
    get_slices,
    reconstruct_from_items,
    list_product
)


@dataclass
class XRDABatcher:
    da: xr.DataArray
    patches: tp.Dict[str, int]
    strides: tp.Dict[str, int]
    da_dims: tp.Dict[str, int]
    da_size: tp.Dict[str, int]
    return_coords: bool = False
    
    def __init__(
            self, 
            da: xr.DataArray,
            patches: tp.Optional[tp.Dict[str, int]]=None,
            strides: tp.Optional[tp.Dict[str, int]]=None,
            domain_limits: tp.Optional[tp.Dict]=None,
            check_full_scan: bool=False,
        ):
        
        if domain_limits is not None:
            da_dims = get_dims_xrda(da)
            check_lists_subset(list(domain_limits.keys()), list(da_dims.keys()))
            da = da.sel(**domain_limits)
        
        self.da = da
        self.da_dims = get_dims_xrda(da)
        
        self.da_size, self.patches, self.strides = get_patches_size(
            dims=self.da_dims,
            patches=patches if patches is not None else {},
            strides=strides if strides is not None else {},
        )
        if check_full_scan:
            for dim in self.patches:
                if (self.da_dims[dim] - self.patches[dim]) % self.strides[dim] != 0:
                    msg = f"\nIncomplete scan in dimension dim {dim}:"
                    msg += f"\nDataArray shape on this dim {self.da_dims[dim]} "
                    msg += f"\nPatch_size along this dim {self.patches[dim]} "
                    msg += f"\nStride along this dim {self.strides[dim]} "
                    msg += f"\n[shape - patch_size] should be divisible by stride: "
                    msg += f"{(self.da_dims[dim] - self.patches[dim]) % self.strides[dim]}"
                    raise IncompleteScanConfiguration(msg)
        
    def __repr__(self) -> str:
        msg = "XArray Patcher"
        msg += "\n=============="
        msg += f"\nDataArray Size: {self.da_dims}"
        msg += f"\nPatches:        {self.patches}"
        msg += f"\nStrides:        {self.strides}"
        msg += f"\nNum Batches:    {self.da_size}"
        return msg
    
    def __str__(self) -> str:
        msg = "XArray Patcher"
        msg += "\n=============="
        msg += f"\nDataArray size: {self.da_dims}"
        msg += f"\nPatches:        {self.patches}"
        msg += f"\nStrides:        {self.strides}"
        msg += f"\nNum Batches:    {self.da_size}"
        return msg
    
    @property
    def coord_names(self) -> tp.List[str]:
        return list(self.da_dims.keys())
    def get_da_slice(self, idx: int, da: xr.DataArray) -> xr.DataArray:
        
        slices = get_slices(idx, 
                          da_size=self.da_size, 
                          patches=self.patches, 
                          strides=self.strides
                         )
        return self.da.isel(**slices)
    
    def __len__(self):
        return list_product(list(self.da_size.values()))
    
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def __getitem__(self, item):
        output = {}

        slices = get_slices(
            idx=item, 
            da_size=self.da_size, 
            patches=self.patches, 
            strides=self.strides
        )
        
        item = self.da.isel(**slices)
        
        item = item.transpose(*self.coord_names)
        
        if self.return_coords:
            return item.coords.to_dataset()
                    
        return item.data.astype(np.float32)
    
    def reconstruct(
        self, 
        batches: tp.List[np.ndarray], 
        dims_label: tp.List[str], 
        weight: tp.Optional[np.ndarray]=None
    ) -> xr.DataArray:
        """
        takes as input a list of np.ndarray of dimensions (b, *, *patch_dims)
        return a stitched xarray.DataArray with the coords of patch_dims
    batches: list of torch tensor correspondin to batches without shuffle
        weight: tensor of size patch_dims corresponding to the weight of a prediction depending on the position on the patch (default to ones everywhere)
        overlapping patches will be averaged with weighting
        """

        items = list(itertools.chain(*batches))
        rec_da = reconstruct_from_items(
            items=items,
            dims_label=dims_label,
            torch_xrds=self,
            weight=weight
        ) 
        
        rec_da.attrs = self.da.attrs
        
        return rec_da
    
    def get_coords(self) -> tp.List[xr.DataArray]:
        self.return_coords = True
        coords = []
        try:
            for i in range(len(self)):
                coords.append(self[i])
        finally:
            self.return_coords = False
            return coords
        
        


class XArrayDataset(torch.utils.data.Dataset):
    """
    torch Dataset based on an xarray.DataArray with on the fly slicing.
    ### Usage: ####
    If you want to be able to reconstruct the input
    the input xr.DataArray should:
        - have coordinates
        - have the last dims correspond to the patch dims in same order
        - have for each dim of patch_dim (size(dim) - patch_dim(dim)) divisible by stride(dim)
    the batches passed to self.reconstruct should:
        - have the last dims correspond to the patch dims in same order
    """

    def __init__(
            self,
            da: xr.DataArray,
            patches: tp.Optional[tp.Dict]=None,
            strides: tp.Optional[tp.Dict]=None,
            domain_limits: tp.Optional[tp.Dict]=None,
            check_full_scan: bool=False,
            check_dim_order: bool=False,
            transforms: tp.Optional[tp.Callable]=None,
    ):
        """
        Args:
            da (xr.DataArray): xarray datarray to be referenced during the iterations
            patches (Optional[Dict]): dict of da dimension to size of a patch
                (defaults to one stride per dimension)
            strides (Optional[Dict]): dict of dims to stride size
                (defaults to one stride per dimension)
            domain_limits (Optional[Dict]): dict of da dimension to slices of domain
                to select for patch extractions
            check_full_scan bool: if True raise an error if the whole domain is
                not scanned by the patch size stride combination
            check_dim_order (bool): if True raise an error for incorrect specification
                of patch dims
            transforms (Optional[Callable]): functions to be called when calling the data.

        Attributes:
            return_coords (bool): Option to return coords during the iterations
            transforms (Optional[Callable]): functions to be called when calling the data
            da (xr.DataArray): xarray datarray to be referenced during the iterations
            patch_dims (Dict): dict of da dimension to size of a patch
                (defaults to the same dimension as dataset stride per dimension)
            strides (Dict): dict of dims to stride size
                (defaults to one stride per dimension)
            domain_limits (Dict): dict of da dimension to slices of domain
                to select for patch extractions
            ds_size (Dict): the dictionary of dimensions for the slicing
            da_dims (Dict): the dictionary of the original dimensions
        """
        super().__init__()
        self.return_coords = False
        self.transforms = transforms

        if domain_limits is not None:
            da_dims = get_dims_xrda(da)
            check_lists_subset(list(domain_limits.keys()), list(da_dims.keys()))
            da = da.sel(**domain_limits)
        
        self.da_dims = get_dims_xrda(da)
        
        self.da = da

        self.da_size, self.patches, self.strides = get_patches_size(
            dims=self.da_dims, 
            patches=patches if patches is not None else {}, 
            strides=strides if strides is not None else {}
            )
        
        if check_full_scan:
            for dim in self.patches:
                if (da_dims[dim] - self.patches[dim]) % self.strides[dim] != 0:
                    raise IncompleteScanConfiguration(
                        f"""
                            Incomplete scan in dimension dim {dim}:
                            dataarray shape on this dim {da_dims[dim]}
                            patch_size along this dim {self.patches[dim]}
                            stride along this dim {self.strides[dim]}
                            [shape - patch_size] should be divisible by stride
                            """
                    )

        if check_dim_order:
            for dim in self.patches:
                if not '#'.join(da.dims).endswith('#'.join(list(self.patches))):
                    raise DangerousDimOrdering(
                        f"""
                            input dataarray's dims should end with patch_dims 
                            dataarray's dim {da.dims}:
                            patch_dims {list(self.patches)}
                            """
                    )

    def __len__(self):
        size = 1
        for v in self.da_size.values():
            size *= v
        return size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_coords(self):
        self.return_coords = True
        coords = []
        try:
            for i in range(len(self)):
                coords.append(self[i])
        finally:
            self.return_coords = False
            return coords

    def __getitem__(self, item):
        
        output = {}

        slices = get_slices(
            idx=item, 
            da_size=self.da_size, 
            patches=self.patches, 
            strides=self.strides
        )
        
        item = self.da.isel(**slices)
        
        if self.return_coords:
            return item.coords.to_dataset()
            
        item = item.data.astype(np.float32)
        if self.transforms is not None:
            return self.transforms(item)
        return item
    

    def reconstruct(self, batches, weight=None):
        """
        takes as input a list of np.ndarray of dimensions (b, *, *patch_dims)
        return a stitched xarray.DataArray with the coords of patch_dims
    batches: list of torch tensor correspondin to batches without shuffle
        weight: tensor of size patch_dims corresponding to the weight of a prediction depending on the position on the patch (default to ones everywhere)
        overlapping patches will be averaged with weighting
        """

        items = list(itertools.chain(*batches))
        return self.reconstruct_from_items(items, weight)

    def reconstruct_from_items(self, items, weight=None):
        if weight is None:
            weight = np.ones(list(self.patches.values()))
        w = xr.DataArray(weight, dims=list(self.patches.keys()))

        coords = self.get_coords()

        new_dims = [f'v{i}' for i in range(len(items[0].shape) - len(coords[0].dims))]
        dims = new_dims + list(coords[0].dims)

        das = [xr.DataArray(it.numpy(), dims=dims, coords=co.coords)
               for it, co in zip(items, coords)]

        da_shape = dict(zip(coords[0].dims, self.da.shape[-len(coords[0].dims):]))
        new_shape = dict(zip(new_dims, items[0].shape[:len(new_dims)]))

        rec_da = xr.DataArray(
            np.zeros([*new_shape.values(), *da_shape.values()]),
            dims=dims,
            coords={d: self.da[d] for d in self.patches}
        )
        count_da = xr.zeros_like(rec_da)

        for da in tqdm.tqdm(das):
            rec_da.loc[da.coords] = rec_da.sel(da.coords) + da * w
            count_da.loc[da.coords] = count_da.sel(da.coords) + w

        return rec_da / count_da


class XRConcatDataset(torch.utils.data.ConcatDataset):
    """
    Concatenation of XrDatasets
    """

    def reconstruct(self, batches, weight=None):
        """
        Returns list of xarray object, reconstructed from batches
        """
        items_iter = itertools.chain(*batches)
        rec_das = []
        for ds in self.datasets:
            ds_items = list(itertools.islice(items_iter, len(ds)))
            rec_das.append(ds.reconstruct_from_items(ds_items, weight))

        return rec_das


# class XRDatasetLatLon(XRDataset):
#     """
#     torch Dataset based on an xarray.DataArray with on the fly slicing.
#     ### Usage: ####
#     If you want to be able to reconstruct the input
#     the input xr.DataArray should:
#         - have coordinates
#         - have the last dims correspond to the patch dims in same order
#         - have for each dim of patch_dim (size(dim) - patch_dim(dim)) divisible by stride(dim)
#     the batches passed to self.reconstruct should:
#         - have the last dims correspond to the patch dims in same order
#     """

#     def __init__(
#             self,
#             ds: xr.Dataset,
#             patch_dims: tp.Dict, 
#             region: tp.Optional[Region]=None,
#             period: tp.Optional[Period]=None,
#             strides: tp.Optional[tp.Dict]=None,
#             check_full_scan: bool=False,
#             check_dim_order: bool=False,
#             transforms: tp.Optional[tp.Callable]=None
#     ):
#         """
#         Args:
#             ds (xr.Dataset): xarray dataset to be referenced during the iterations
#             patch_dims (Dict): dict of da dimension to size of a patch
#             region (Optional[Region]): a custom region to subset the xr.Dataset
#             period (Optional[Period]): a custom period to subset the xr.Dataset
#             strides (Optional[Dict]): dict of dims to stride size (default to one)
#             check_full_scan bool: if True raise an error if the whole domain is
#                 not scanned by the patch size stride combination
#             check_dim_order (bool): if True raise an error for incorrect specification
#                 of patch dims
#             transforms (Optional[Callable]): functions to be called when calling the data.

#         Attributes:
#             return_coords (bool): Option to return coords during the iterations
#             transforms (Optional[Callable]): functions to be called when calling the data
#             ds (xr.Dataset): xarray dataset to be referenced during the iterations
#             patch_dims (Dict): dict of da dimension to size of a patch
#             domain_limits (Optional[Dict]): dict of da dimension to slices of domain
#                 to select for patch extractions
#             strides (Optional[Dict]): dict of dims to stride size (default to one)
#             ds_size (Dict): the dictionary of dimensions
#         """
#         super().__init__()

#         if period is not None:
#             ds = select_period(ds=ds, period=period)
#         if region is not None:
#             ds = select_region(ds=ds, region=region)
#         self.ds = ds
#         self.return_coords = False
#         self.transforms = transforms
#         self.patch_dims = patch_dims
#         self.strides = strides or {}
#         da_dims = dict(zip(self.ds.dims, self.ds.shape))
#         self.ds_size = {
#             dim: max((da_dims[dim] - patch_dims[dim]) // self.strides.get(dim, 1) + 1, 0)
#             for dim in patch_dims
#         }

#         if check_full_scan:
#             for dim in patch_dims:
#                 if (da_dims[dim] - self.patch_dims[dim]) % self.strides.get(dim, 1) != 0:
#                     raise IncompleteScanConfiguration(
#                         f"""
#                             Incomplete scan in dimension dim {dim}:
#                             dataarray shape on this dim {da_dims[dim]}
#                             patch_size along this dim {self.patch_dims[dim]}
#                             stride along this dim {self.strides.get(dim, 1)}
#                             [shape - patch_size] should be divisible by stride
#                             """
#                     )

#         if check_dim_order:
#             for dim in patch_dims:
#                 if not '#'.join(ds.dims).endswith('#'.join(list(patch_dims))):
#                     raise DangerousDimOrdering(
#                         f"""
#                             input dataarray's dims should end with patch_dims 
#                             dataarray's dim {ds.dims}:
#                             patch_dims {list(patch_dims)}
#                             """
#                     )