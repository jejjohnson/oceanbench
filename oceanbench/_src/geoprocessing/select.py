from typing import Iterable, Union
import xarray as xr
from oceanbench._src.utils.custom_dtypes import Region, Period, Bounds
from functools import reduce

xrdata = Union[xr.Dataset, xr.DataArray]


def select_bounds(ds: xrdata, bounds: Bounds) -> xrdata:
    """This is syntactic sugar to select a coordinate/data based on
    the bounds

    Args:
        ds (xrdata): the xarray dataarray/dataset
        bounds (Bounds): the custom Bounds object

    Returns:
        ds (xrdata): the xarray dataset after the subset
    """
    return ds.where(
        (ds[bounds.name] >= bounds.val_min)
        & (ds[bounds.name] <= bounds.val_max),
        drop=True
    )


# TODO: This is a great case for multiple dispatch...
def select_bounds_multiple(ds: xrdata, bounds: Iterable[Bounds]) -> xrdata:
    """This is syntatic sugar to select data based on a list of bounds.

    Args:
        ds (xrdata): the xarray dataarry/dataset
        bounds (Iterable[Bounds]): A list of bounds

    Returns:
        ds (xrdata): the xarray dataset after the multiple subsets
    """
    # this is very pythonic way to do:
    # d_sub = ssh_xrds.copy()
    # for ibnd in bounds:
    #     d_sub = select_bounds(d_sub, ibnd)

    bounds = [ds] + bounds
    
    return reduce((lambda ds, ibounds: select_bounds(ds, ibounds)), bounds)


def select_region(ds: xr.Dataset, region: Region) -> xr.Dataset:
    """This is syntactic sugar to select a region

    Args:
        ds (xr.Dataset): the xarray dataset
        region (Region): the region

    Returns:
        ds (xr.Dataset): the xarray dataset with the slices
    """
    ds = ds.where(
        (ds["lon"] >= region.lon_min)
        & (ds["lon"] <= region.lon_max)
        & (ds["lat"] >= region.lat_min)
        & (ds["lat"] <= region.lat_max),
        drop=True,
    )

    return ds


def select_preiod(ds: xr.Dataset, period: Period) -> xr.Dataset:
    """This is syntactic sugar to select a time period from an 
    xr.Dataset

    Args:
        ds (xr.Dataset): _description_
        period (Period): the time period

    Returns:
        ds (xr.Dataset): the new xarray dataset
    """
    ds = ds.where(
        (ds["time"] >= period.t_min)
        & (ds["time"] <= period.t_max),
        drop=True,
    )

    return ds
