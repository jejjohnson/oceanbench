import pytest
import numpy as np
import pandas as pd
import xarray as xr
from .temporal import time_rescale


@pytest.mark.parametrize(
    "freq_unit, num_points",
    [
        ("day", 31),
        ("minute", 43_201),
        ("second", 2_592_001),
    ],
)
def test_time_rescale(freq_unit, num_points):
    tmin, tmax = np.datetime64("2013-01-01"), np.datetime64("2013-01-31")
    freq_dt = 1

    dt = pd.to_timedelta(freq_dt, freq_unit)

    time_coords = np.arange(tmin, tmax + dt, dt)

    da = xr.Dataset()
    da["time"] = time_coords

    da = time_rescale(da, freq_dt=freq_dt, freq_unit=freq_unit, t0=None)

    np.testing.assert_array_equal(da["time"].values, np.arange(0, num_points, 1))
    assert da["time"].shape[0] == num_points
    assert da["time"].attrs["units"] == freq_unit


# def test_time_unrescale():
#
#     t0 = "2012-12-15"
#     tmin, tmax = np.datetime64("2013-01-01"), np.datetime64("2013-01-31")
#     freq_dt = 1
#     freq_unit = "D"
#
#     dt = np.timedelta64(freq_dt, freq_unit)
#
#     time_coords = np.arange(tmin, tmax + dt, dt)
#
#     da = xr.Dataset()
#     da["time"] = time_coords
#
#     da = time_rescale(da, freq_dt=freq_dt, freq_unit=freq_unit, t0=t0)
#
#     np.testing.assert_array_equal(da["time"].values, np.arange(17.0, 47.0 + 1.0, 1.0))
#     assert da["time"].shape[0] == 31
#     assert da["time"].attrs["units"] == "d"
#
#     # inverse transformation
#     da = time_unrescale(da, freq_dt=freq_dt, freq_unit=freq_unit, t0=t0)
#
#     np.testing.assert_array_equal(da["time"].values, time_coords)
#     assert da["time"].shape[0] == 31
#     assert da["time"].attrs["units"] == "ns"
