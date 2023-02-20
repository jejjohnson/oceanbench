import xarray as xr
import functools as ft
import pandas as pd
from collections import defaultdict

def parse_regular_dim(ds, dim, tol):
    dmin, dmax, dd = ds.pipe(
        lambda ds: (ds[dim].min().values, ds[dim].max().values, ds[dim].diff(dim).values)
    )
    if (dd.min() - dd.max()) > tol:
        raise Exception(f"Irregular {dim} dimension")
    return dmin, dmax, dd.min()


def bin_values(values, resolution, start_from=None):
    start_from = start_from or 0
    return (values - start_from) / resolution // 1 * resolution + start_from


def compute_grid_coords(ds, resolutions, start_froms=None, prefix='grid_'):
    start_froms = start_froms or defaultdict(lambda :None)
    return (
            ds
            .pipe(
                lambda ds: ds.assign(**{
                    f'{prefix}{coord}': bin_values(ds[coord], resolutions[coord], start_froms[coord])
                    for coord in resolutions.keys()
            }))
        )


def multi_groupby(ds, groupbys, aggfn=None):
    if aggfn is None:
        aggfn = ft.partial(pd.DataFrame.mean, numeric_only=False, skipna=True)

    return  (
        ds
        .to_dataframe()
        .reset_index()
        .groupby(groupbys)
        .agg(aggfn)
        .to_xarray()
    )

def coord_based_to_grid(coord_based_ds: xr.Dataset, target_grid_ds: xr.Dataset):
    tmin, _, dt = parse_regular_dim(target_grid_ds, 'time', pd.to_timedelta('1s'))
    xmin, _, dx = parse_regular_dim(target_grid_ds, 'lon', 1e-6)
    ymin, _, dy = parse_regular_dim(target_grid_ds, 'lat', 1e-6)

    resolutions = dict(time=dt, lon=dx, lat=dy)
    start_froms = dict(time=tmin, lon=xmin, lat=ymin)
    ds_with_grid_coords = compute_grid_coords(coord_based_ds, resolutions, start_froms)

    gridded = (
        ds_with_grid_coords
        .pipe(ft.partial(multi_groupby, groupbys=[f'grid_{coord}' for coord in resolutions.keys()]))
        .drop_vars(resolutions.keys())
        .rename(**{f'grid_{coord}': coord for coord in resolutions.keys()})
        .reindex_like(target_grid_ds)
    )

    return gridded

