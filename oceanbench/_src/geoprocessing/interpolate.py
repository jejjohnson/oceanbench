from typing import List
import xarray as xr
import pyinterp
import pyinterp.fill
import pyinterp.backends.xarray


def fillnan_gauss_seidel(ds: xr.Dataset, variable: str):
    ds = ds.copy()
    ds["lon"] = ds.lon.assign_attrs(units="degrees_east")
    ds["lat"] = ds.lat.assign_attrs(units="degrees_north")

    ds[variable].transpose("lon", "lat", "time")[:, :] = pyinterp.fill.gauss_seidel(
        pyinterp.backends.xarray.Grid3D(ds[variable])
    )[1]
    return ds


