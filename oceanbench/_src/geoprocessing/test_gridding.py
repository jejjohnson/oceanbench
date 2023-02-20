import pytest
import xarray as xr
import numpy as np
import pandas as pd
import oceanbench._src.geoprocessing.gridding as gridding


@pytest.fixture()
def test_domain():
    return dict(
       time=[pd.to_datetime('2020-01-01'), pd.to_datetime('2020-01-05')],
       lat=[40, 50],
       lon=[-60, -50],
    )

@pytest.fixture()
def simple_coord_based_ds_1d(test_domain):
    n_points = 100
    day =  np.arange(n_points) // 20
    time =  test_domain['time'][0] + day*pd.to_timedelta('1D') + (np.arange(n_points) % 20) * pd.to_timedelta('1H')
    lat =  test_domain['lat'][0] + (np.arange(n_points) % 20) * 0.1
    lon =  test_domain['lon'][0] + (np.arange(n_points) % 20) * 0.2
    return xr.Dataset(
        {
            'ssh': ('idx', np.sin(np.linspace(0, 3, 100))),
            'time': ('idx', time),
            'lat': ('idx', lat),
            'lon': ('idx', lon),
        },
        {'idx': np.arange(100)}
    ) 

@pytest.fixture()
def simple_grid_ds_12H_05(test_domain):
    dt = pd.to_timedelta('12H')
    dx = 0.5
    return xr.Dataset(
        coords = {
            'time': pd.date_range(*test_domain['time'], freq=dt),
            'lat': np.arange(*test_domain['lat'], dx),
            'lon': np.arange(*test_domain['lon'], dx),
        }
    ) 


def test_coord_based_to_grid(simple_grid_ds_12H_05, simple_coord_based_ds_1d):
    gridded = gridding.coord_based_to_grid(simple_coord_based_ds_1d, simple_grid_ds_12H_05)
    xr.testing.assert_allclose(
            gridded[['lat', 'lon']],
            simple_grid_ds_12H_05[['lat', 'lon']],
    )
    

def test_parse_regular_dim(simple_grid_ds_12H_05):
    assert  False

def test_bin_values(simple_grid_ds_12H_05):
    assert  False

def test_multi_groupby(simple_grid_ds_12H_05):
    assert  False
