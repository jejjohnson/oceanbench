from typing import Optional, Union
import numpy as np
import xarray as xr
import pandas as pd
import pint_xarray


def time_rescale(
    ds: xr.Dataset,
    freq_dt: int = 1,
    freq_unit: str = "seconds",
    t0: Optional[Union[str, np.datetime64]] = None,
) -> xr.Dataset:
    """Rescales time dimensions of np.datetim64 to an output frequency.

    t' = (t - t_0) / dt

    Args:
        ds (xr.Dataset): the xr.Dataset with a time dimensions
        freq_dt (int): the frequency of the temporal coordinate
        freq_unit (str): the unit for the time frequency parameter
        t0 (datetime64, str): the starting point. Optional. If none, assumes the
            minimum value of the time coordinate

    Returns:
        ds (xr.Dataset): the xr.Dataset with the rescaled time dimensions in the
            freq_unit.
    """

    ds = ds.copy()

    if t0 is None:
        t0 = ds["time"].min()

    if isinstance(t0, str):
        t0 = np.datetime64(t0)

    ds["time"] = ((ds["time"] - t0) / pd.to_timedelta(freq_dt, unit=freq_unit)).astype(
        np.float32
    )

    ds = ds.pint.quantify({"time": freq_unit}).pint.dequantify()

    return ds


# def time_unrescale(
#     ds: xr.Dataset,
#     t0: Union[str, np.datetime64],
#     freq_dt: int=1,
#     freq_unit: str="D",
# ) -> xr.Dataset:
#     """UnRescales time dimensions of an output frequency to a np.datetim64
#
#     t' = t * dt + t_0
#
#     Args:
#         ds (xr.Dataset): the xr.Dataset with a time dimensions
#         t0 (datetime64, str): the starting point.
#         freq_dt (int): the frequency of the temporal coordinate
#         freq_unit (str): the unit for the time frequency parameter
#
#     Returns:
#         ds (xr.Dataset): the xr.Dataset with the rescaled time dimensions in the
#             np.datetime64 units.
#     """
#
#     ds = ds.copy()
#
#     if isinstance(t0, str):
#         t0 = np.datetime64(t0)
#
#     ds["time"] = ds["time"].dt * np.timedelta64(freq_dt, freq_unit) + t0
#
#     ds["time"].attrs["units"] = "ns"
#
#     return ds
