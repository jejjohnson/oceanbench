from typing import Optional
import pandas as pd
import xarray as xr


def validate_latlon(ds: xr.Dataset) -> xr.Dataset:
    """Format lat and lon variables

    Set units, ranges and names

    Args:
        ds: input data

    Returns:
        formatted data
    """
    new_ds = ds.copy()
    new_ds["lon"] = (ds.lon + 180) % 360 - 180
    new_ds["lon"] = new_ds.lon.assign_attrs(
        **{
            **dict(
                units="degrees_east",
                standard_name="longitude",
                long_name="Longitude",
            ),
            **ds.lon.attrs,
        }
    )

    new_ds["lat"] = (ds.lat + 90) % 180 - 90
    new_ds["lat"] = new_ds.lat.assign_attrs(
        **{
            **dict(
                units="degrees_north",
                standard_name="latitude",
                long_name="Latitude",
            ),
            **ds.lat,
        }
    )
    return new_ds


def decode_cf_time(
    ds: xr.Dataset, units: Optional[str] = "seconds since 2012-10-01"
) -> xr.Dataset:
    """Decode time variable in cf format

    Args:
        ds: input dataset
        units: cf time units

    Returns:
        Decoded Dataset
    """
    ds = ds.copy()
    if units is not None:
        ds["time"] = ds.time.assign_attrs(units=units)
    return xr.decode_cf(ds)


def validate_time(ds: xr.Dataset) -> xr.Dataset:
    """Convert time to pandas datetime"""
    ds = ds.copy()
    ds["time"] = pd.to_datetime(ds.time)
    return ds


def validate_ssh(ds: xr.Dataset, variable: str = 'ssh') -> xr.Dataset:
    """ Assign ssh attributes to variable """
    ds = ds.copy()
    ds[variable] = ds.ssh.assign_attrs(
        units="m",
        standard_name="sea_surface_height",
        long_name="Sea Surface Height",
    )
    return ds


def check_time_lat_lon(ds: xr.Dataset) -> None:
    """Raise AssertError if dataset is not correctly formatted"""
    assert {"lat", "lon", "time"} < set(ds.variables)
    xr.testing.assert_identical(ds[["lat", "lon"]], validate_latlon(ds)[["lat", "lon"]])
    xr.testing.assert_identical(ds["time"], validate_latlon(ds)["time"])
