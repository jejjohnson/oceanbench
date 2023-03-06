import pandas as pd
import xarray as xr


def validate_latlon(da):
    da = da.copy()
    da["lon"] = da.lon.assign_attrs(
        units="degrees_east",
        standard_name="longitude",
        long_name="Longitude",
    )
    da["lon"] = (da.lon + 180) % 360 - 180

    da["lat"] = da.lat.assign_attrs(
        units="degrees_north",
        standard_name="latitude",
        long_name="Latitude",
    )
    da["lat"] = (da.lat + 90) % 180 - 90
    return da


def decode_cf_time(da, units='seconds since 2012-10-01'):
    da = da.copy()
    if units is not None:
        da["time"] = da.time.assign_attrs(units=units)
    return xr.decode_cf(da)


def validate_time(da):
    da = da.copy()
    da["time"] = pd.to_datetime(da.time)
    return da


def validate_ssh(da):
    da = da.copy()
    da["ssh"] = da.ssh.assign_attrs(
        units="m",
        standard_name="sea_surface_height",
        long_name="Sea Surface Height",
    )
    return da


def check_time_lat_lon(da):
    assert {"lat", "lon", "time"} < set(da.variables)
    xr.testing.assert_identical(da[["lat", "lon"]], validate_latlon(da)[["lat", "lon"]])
    xr.testing.assert_identical(da["time"], validate_latlon(da)["time"])
