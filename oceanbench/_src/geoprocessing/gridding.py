import xarray as xr
import xesmf as xe
import numpy as np
import pyinterp
import pyinterp.backends.xarray


def to_dim(ds, v):
    """
    ds: xr.Dataset
    v: one dimensional variable or coordinates name

    Return: xr.Dataset  with v as a dimension
    """
    if v in ds.dims:
        return ds
    return ds.swap_dims({ds[v].dims[0]: v})


def grid_da(da, binning, var):
    """
    da: xr.DataArray (with lon and lat coordinates
    binning: pyinterp.Binning

    Return: tuple (dim_names, binned_da_values)
    """
    binning.clear()
    values = np.ravel(da[var].values)
    lons = np.ravel(da.lon.values)
    lats = np.ravel(da.lat.values)
    msk = np.isfinite(values)
    binning.push(lons[msk], lats[msk], values[msk])
    variable = binning.variable('mean').T[None, ...]
    
    return (('time', 'lat', 'lon'), variable)

def coord_based_to_grid(coord_based_ds, target_grid_ds, data_vars=None, t_res=None):
    """
        coord_based_ds: xr.Dataset with time, lat, lon coordinates
        target_grid_ds: xr.Dataset with  uniform time, lat, lon coordinates
        data_vars: Optional[Iterable[str]] variables of coord_based_ds to include in return dataset,
                      if None, use all the variables (other that time, lat, lon)

        Return: xr.Dataset a dataset with same dimensions and coordinates as target_grid_ds and averaged 
                      values of coord_based_ds for each data_vars
    """
    if data_vars is None:
        data_vars = set(coord_based_ds.variables) - {'time', 'lat', 'lon'}

    ds = to_dim(coord_based_ds, 'time')
    if t_res is None:
        t_res = target_grid_ds.time.diff('time').values.mean()#/2
    binning = pyinterp.Binning2D(
        pyinterp.Axis(target_grid_ds.lon.values), 
        pyinterp.Axis(target_grid_ds.lat.values)
    )
    grid_dses = []
    for t in target_grid_ds.time:
        tds = ds.isel(time=(ds.time > (t - t_res/2)) & (ds.time <= (t + t_res/2)))
        grid_dses.append(
           xr.Dataset(
               {v: grid_da(tds, binning, v) for v in data_vars},
               {'time': [t.values], 'lat': np.array(binning.y), 'lon': np.array(binning.x)}
            ).astype('float32', casting='same_kind')
        )
    tgt_ds = xr.concat(grid_dses, dim='time')
    return tgt_ds



def grid_to_regular_grid(src_grid_ds, tgt_grid_ds, keep_attrs: bool=True):
    """
        src_grid_ds: xr.Dataset with regular lat, lon coordinates (uniform or curvilinear)
        tgt_grid_ds: xr.Dataset with  uniform lat, lon coordinates

        Return: xr.Dataset a dataset with same lat, lon coordinates as tgt_grid_ds 
                      and bilinearly interpolated  values of src_grid_ds.
                    (time coordinates remains unchanged)
    """
    reggridder = xe.Regridder(src_grid_ds, tgt_grid_ds, "bilinear", unmapped_to_nan=True)
    return reggridder(src_grid_ds, keep_attrs=keep_attrs)


def interp_da(da, tgt_coords):
    """
    da: xr.DataArray with uniform time, lat, lon coordinates
    tgt_coords: dict[str: np.array] Mapping from dimension names to the coordinates to interpolate.
        Coordinates must be array-like.  array of coordinates of the points of interpolation

    Return: np.array The interpolated values.

    Perform bilinear interpolation spatially followed by a linear interpolation temporally
    """
    interpolator = pyinterp.backends.xarray.Grid3D(da)
    return interpolator.trivariate(tgt_coords)


def grid_to_coord_based(src_grid_ds, tgt_coord_based_ds, data_vars=None):
    """
        src_grid_ds: xr.Dataset with uniform time, lat, lon coordinates
        tgt_coord_based_ds: xr.Dataset with  time, lat, lon coordinates
        data_vars: Optional[Iterable[str]] variables of src_grid_ds to include in return dataset,
                      if None, use all the variables with dimensions (time, lat, lon)

        Return: xr.Dataset a dataset with same time, lat, lon coordinates as tgt_coord_based_ds 
                      and interpolated  values of src_grid_ds.

        Perform bilinear interpolation spatially followed by a linear interpolation temporally
    """
    if data_vars is None:
        data_vars = [
            v for v in src_grid_ds.variables
            if set(src_grid_ds[v].dims) == {'time', 'lat', 'lon'}
        ]

    ref = tgt_coord_based_ds.lon
    coords = dict(
        lon=np.ravel(ref.values),
        lat=np.ravel(tgt_coord_based_ds.lat.transpose(*ref.dims).values),
        time=np.ravel(tgt_coord_based_ds.time.transpose(*ref.dims).values),
    )
    return xr.Dataset(
        {v: (ref.dims, np.reshape(interp_da(src_grid_ds[v], coords), ref.shape), src_grid_ds[v].attrs) for v in data_vars},
        tgt_coord_based_ds.coords
    )