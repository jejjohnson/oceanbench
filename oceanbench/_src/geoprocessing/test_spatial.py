import numpy as np
import xarray as xr
from .spatial import transform_180_to_360, transform_360_to_180, latlon_deg2m


def test_transform_180_to_360_bounds():
    coords = np.linspace(-180, 180, 20)

    new_coords = transform_180_to_360(coords)

    assert new_coords.min() >= 0.0
    assert new_coords.max() <= 360

    new_coords = transform_180_to_360(new_coords)

    assert new_coords.min() >= 0.0
    assert new_coords.max() <= 360


def test_transform_360_to_180_bounds():
    coords = np.linspace(0, 360, 20)

    new_coords = transform_360_to_180(coords)

    assert new_coords.min() >= -180
    assert new_coords.max() <= 180

    new_coords = transform_360_to_180(new_coords)

    assert new_coords.min() >= -180
    assert new_coords.max() <= 180

def test_latlon_deg2m():
    
    lon_coords = np.linspace(-65, -54, 50)
    lat_coords = np.linspace(32, 44, 50)
    
    da = xr.Dataset()
    da["lon"] = lon_coords
    da["lat"] = lat_coords
    
    da = latlon_deg2m(da, mean=False)
    
    # check minval is 0
    assert da["lon"].min() == 0
    assert da["lat"].min() == 0
    
    # check attributes
    assert da["lon"].attrs["units"] == "m"
    assert da["lat"].attrs["units"] == "m"
    
    
def test_latlon_deg2m_mean():
    
    lon_coords = np.linspace(-65, -54, 50)
    lat_coords = np.linspace(32, 44, 50)
    
    da = xr.Dataset()
    da["lon"] = lon_coords
    da["lat"] = lat_coords
    
    da = latlon_deg2m(da, mean=True)
    
    # check minval is 0
    assert da["lon"].min() == 0
    assert da["lat"].min() == 0
    
    # check attributes
    assert da["lon"].attrs["units"] == "m"
    assert da["lat"].attrs["units"] == "m"
    
    
    
    