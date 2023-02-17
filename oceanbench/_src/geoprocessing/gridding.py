import xarray as xr
import xesmf as xe
def coord_based_to_grid(coord_based_ds: xr.Dataset, target_grid_ds: xr.Dataset):
    ...

def grid_to_grid(source_grid_ds: xr.Dataset, target_grid_ds: xr.Dataset):
    xe.
    reggridder = xe.Regridder(ds, ds_out, "bilinear")
    dr_out = reggridder(ds, keep_attrs=True)
