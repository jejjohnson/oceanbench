import typing as tp
from dataclasses import dataclass
import itertools
import numpy as np
import xarray as xr
from tqdm import tqdm
from oceanbench._src.utils.exceptions import IncompleteScanConfiguration
from oceanbench._src.datasets.utils import (
    get_dims_xrda,
    check_lists_subset,
    get_patches_size,
    get_slices
)


class XRDABatcher:
    """
    A dataclass for xarray.DataArray with on the fly slicing and arbitrary
    dimension reconstruction.

    ### Usage: ####
    If you want to be able to reconstruct the input
    the input xr.DataArray should:
        - have coordinates
        - have for each dim of patch_dim (size(dim) - patch_dim(dim)) divisible by stride(dim)
        (optional warning)
    """
    
    def __init__(
            self, 
            da: xr.DataArray,
            patches: tp.Optional[tp.Dict[str, int]]=None,
            strides: tp.Optional[tp.Dict[str, int]]=None,
            domain_limits: tp.Optional[tp.Dict]=None,
            check_full_scan: bool=False,
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
                
        Attributes:
            da (xr.DataArray): xarray datarray to be referenced during the iterations
            patches (OrderedDict): dict of da dimension to size of a patch
                (defaults to the same dimension as dataset stride per dimension)
            strides (OrderedDict): dict of dims to stride size
                (defaults to one stride per dimension)
            ds_size (OrderedDict): the dictionary of dimensions for the slicing
            da_dims (OrderedDict): the dictionary of the original dimensions
        """
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
                    msg += f"{self.da_dims[dim]} - {self.patches[dim]} % {self.strides[dim]} "
                    soln = (self.da_dims[dim] - self.patches[dim]) / self.strides[dim]
                    msg += f"= {soln}"
                    raise IncompleteScanConfiguration(msg)
        
    def __repr__(self) -> str:
        msg = "XArray Patcher"
        msg += "\n=============="
        msg += f"\nDataArray Size: {self.da_dims}"
        msg += f"\nPatches:        {self.patches}"
        msg += f"\nStrides:        {self.strides}"
        msg += f"\nNum Items:    {self.da_size}"
        return msg
    
    def __str__(self) -> str:
        msg = "XArray Patcher"
        msg += "\n=============="
        msg += f"\nDataArray size: {self.da_dims}"
        msg += f"\nPatches:        {self.patches}"
        msg += f"\nStrides:        {self.strides}"
        msg += f"\nNum Items:    {self.da_size}"
        return msg
    
    @property
    def coord_names(self) -> tp.List[str]:
        return list(self.da_dims.keys())
    
    def __len__(self):
        return np.prod(list(self.da_size.values()))
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def __getitem__(self, item):

        slices = get_slices(
            idx=item,
            da_size=self.da_size,
            patches=self.patches,
            strides=self.strides
        )
                        
        return self.da.isel(**slices)
    
    def get_coords(self) -> tp.List[xr.DataArray]:
        """"Returns a list of xr.DataArray's with the
        coordinate values that correspond to each item.
        """
        coords = []
        for i in range(len(self)):
            coords.append(self[i].coords.to_dataset()[list(self.patches)])
        return coords
    
    def reconstruct(
        self,
        items: tp.Iterable,
        dims_labels: tp.Optional[tp.Iterable[str]]=None, 
        weight=None
    ) -> xr.DataArray:
        """Reconstructs based on a list of patches, e.g. the output of
        a dataloader.

        Args:
            items (List[np.ndarray]): a list of np.ndarrays with arbitrary dimensions where
                at least one dimension matches the patch dimensions
            dims_label (List[str]): a list of dimension names for the patches. If explicit, it
                will match all names that correspond with the patch dims. If missing, it will
                infer based on the shapes (use with caution). Any extra patch dims will be added
                dimensions for the reconstructed xr.DataArray.
            weight (np.ndarray): the tensor which is the same size as the patch dimensions requested
                to be reconstructed. If not specified and the dims_label is specific, it will be
                constructed based on the matching dimensions specified. If not specified and the
                dims_label is not specified, it will be inferred based on the first N dimensions
                that correspond with the patch dimensions. The default is that all overlapping
                patches will be averaged with ones.

        Returns:
            rec_da (xr.DataArray): the reconstructed xr.DataArray that corresponds to
                to the original array of the corresponding requested coordinates.
        """
        
        item_shape = items[0].shape
        num_items = len(item_shape)
        
        # get coordinate labels
        coords = self.get_coords()
        coords_labels = list(coords[0].dims.keys())
        
        # assume the items are the same as the coordinates
        if dims_labels is None:
            dims_labels = [coords_labels[i] for i in range(len(coords_labels))]
        
        num_dim_labels = len(dims_labels)

        # add any extra dimensions not specified
        if num_dim_labels < num_items:
            new_dims  = [f"v{i+1}" for i in range(num_items - num_dim_labels)]
            dims_labels = dims_labels + new_dims
            num_dim_labels = len(dims_labels)

        # check user specified dimensions
        msg = f"Length of dim labels does not match length of dims."
        msg += f"\nDims Label: {dims_labels} \nShape: {item_shape}"
        msg += f"\nNum Labels: {num_dim_labels} \nNum Items: {num_items}"

        assert num_dim_labels == num_items, msg
        
        # check for subset of coordinate arrays
        coords_labels = set(dims_labels).intersection(coords_labels)

        # check_lists_subset(coords_labels, dims_labels)
        all_items_shape = dict(zip(dims_labels, item_shape))

        patches = {ikey: ivalue for ikey, ivalue in self.patches.items() if ikey in dims_labels}
        patch_values = list(patches.values())
        patch_names = list(patches.keys())
        
        msg = "No Coordinates to merge..."
        msg += f"\nDims: {dims_labels}"
        msg += f"\nCoords: {list(coords[0].dims.keys())}"
        msg += f"\nPatches: {self.patches.keys()}"

        assert len(coords_labels) > 0, msg

        msg = "Mismatch between coords and patches..."
        msg += f"\nDims: {dims_labels}"
        msg += f"\nCoords: {list(coords[0].dims.keys())}"
        msg += f"\nPatches: {self.patches.keys()}"

        assert len(coords_labels) == len(patch_names), msg

        if weight is None:
            weight = np.ones(patch_values)

        else:
            msg = "Weight array is not the same size as total dims "
            msg += "or not the same value"
            msg += f"\nWeight: {list(weight.shape)} | Patches: {patch_values} | Dims: {item_shape}"
            assert len(weight.shape) == len(patch_values), msg

        w = xr.DataArray(weight, dims=patch_names)

        # create data arrays from coords
        das = [
            xr.DataArray(it, dims=dims_labels, coords=co[coords_labels].coords)
            for it, co in zip(items, coords)
                ]

        msg = "New Data Array is not the same size as items"
        msg += "or not the same value"
        msg += f"\n{das[0].shape} | Items: {all_items_shape}"

        assert len(das[0].shape) == len(all_items_shape), msg
        assert set(das[0].shape) == set(all_items_shape.values()), msg

        # get new shape from 
        new_shape = {
            idim: self.da[idim].shape[0] if idim in coords_labels
            else item_shape[i]
            for i, idim in enumerate(dims_labels) 
        }
        coords = {d: self.da[d] for d in das[0].coords}
        rec_da = xr.DataArray(
            np.zeros([*new_shape.values()]),
            dims=dims_labels,
            coords=coords
        )

        count_da = xr.zeros_like(rec_da)

        for ida in tqdm(das):
            da_co = {c: ida[c] for c in coords_labels}
            rec_da.loc[da_co] = rec_da.sel(da_co) + ida * w
            count_da.loc[da_co] = count_da.sel(da_co) + w

        rec_da.attrs = self.da.attrs
        return rec_da / count_da
    
