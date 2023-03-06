import pandas as pd
import xarray as xr


def validate_latlon(da):
    new_da = da.copy()
    new_da["lon"] = (da.lon + 180) % 360 - 180
    new_da["lon"] = new_da.lon.assign_attrs(
        **{**dict(units="degrees_east",
        standard_name="longitude",
        long_name="Longitude",),
        **da.lon.attrs}
    )

    new_da["lat"] = (da.lat + 90) % 180 - 90
    new_da["lat"] = new_da.lat.assign_attrs(
        **{**dict(units="degrees_north",
        standard_name="latitude",
        long_name="Latitude",),
        **da.lat.attrs,}
    )
    return new_da


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
