import pytest
import xarray as xr
import numpy as np
import pandas as pd
import oceanbench._src.geoprocessing.gridding as gridding

@pytest.fixture()
def ssh_attrs():
    return dict( units='m', long_name='Sea Surface Height',)

@pytest.fixture()
def time_attrs():
    return dict(long_name='Date')


@pytest.fixture()
def lon_attrs():
    return dict(units='degrees_east', long_name='Longitude')

@pytest.fixture()
def lat_attrs():
    return dict(units='degrees_north', long_name='Latitude')

@pytest.fixture()
def test_domain():
    return dict(
       time=[pd.to_datetime('2020-01-01'), pd.to_datetime('2020-01-05')],
       lat=[40, 50],
       lon=[-60, -50],
    )

@pytest.fixture()
def simple_coord_based_ds_1d(test_domain, ssh_attrs, time_attrs, lon_attrs, lat_attrs):
    n_points = 100
    day =  np.arange(n_points) // 20
    time =  test_domain['time'][0] + day*pd.to_timedelta('1D') + (np.arange(n_points) % 20) * pd.to_timedelta('1H')
    lat =  test_domain['lat'][0] + (np.arange(n_points) % 20) * 0.1
    lon =  test_domain['lon'][0] + (np.arange(n_points) % 20) * 0.2
    return xr.Dataset(
        {
            'ssh': ('idx', np.ones_like(np.linspace(0, 3, 100)), ssh_attrs),
        },
        {
            'idx': np.arange(100),
            'time': ('idx', time, time_attrs),
            'lat': ('idx', lat, lat_attrs),
            'lon': ('idx', lon, lon_attrs),
        }

    ) 

def _mk_grid_ds(dt, dx, test_domain, ssh_attrs, time_attrs, lon_attrs, lat_attrs):
    return xr.Dataset(
        coords = {
            'time': ('time', pd.date_range(*test_domain['time'], freq=dt), time_attrs),
            'lat': ('lat', np.arange(*test_domain['lat'], dx), lat_attrs),
            'lon': ('lon', np.arange(*test_domain['lon'], dx), lon_attrs),
        }
    ).assign(ssh=lambda ds: (ds.dims, np.ones([*ds.dims.values()]), ssh_attrs))

@pytest.fixture()
def simple_grid_ds_12H_05(test_domain, ssh_attrs, time_attrs, lon_attrs, lat_attrs):
    dt = pd.to_timedelta('12H')
    dx = 0.5
    return _mk_grid_ds(dt, dx, test_domain, ssh_attrs, time_attrs, lon_attrs, lat_attrs)

@pytest.fixture()
def simple_grid_ds_24H_01(test_domain, ssh_attrs, time_attrs, lon_attrs, lat_attrs):
    dt = pd.to_timedelta('1D')
    dx = 0.1
    return _mk_grid_ds(dt, dx, test_domain, ssh_attrs, time_attrs, lon_attrs, lat_attrs)

def test_coord_based_to_grid(simple_grid_ds_12H_05, simple_coord_based_ds_1d):
    gridded = gridding.coord_based_to_grid(simple_coord_based_ds_1d, simple_grid_ds_12H_05)
    xr.testing.assert_allclose(
            gridded[['lat', 'lon']],
            simple_grid_ds_12H_05[['lat', 'lon']],
    )
    
    xr.testing.assert_allclose(
            gridded.time,
            simple_grid_ds_12H_05.time,
    )
    
    values = gridded.ssh.values
    msk = np.isfinite(values)
    assert msk.sum()>0, "should have finite values"
    np.testing.assert_almost_equal(values[msk], 1.), "Gridding ones should make ones"

    
def test_grid_to_coord_based(simple_grid_ds_12H_05, simple_coord_based_ds_1d):
    trackified = gridding.grid_to_coord_based(simple_grid_ds_12H_05, simple_coord_based_ds_1d)
    xr.testing.assert_allclose(
            trackified[['lat', 'lon']],
            simple_coord_based_ds_1d[['lat', 'lon']],
    )
    xr.testing.assert_allclose(
            trackified.time,
            simple_coord_based_ds_1d.time,
    )
    
    values = trackified.ssh.values
    msk = np.isfinite(values)
    assert msk.sum()>0, "should have finite values"
    np.testing.assert_almost_equal(values[msk], 1.), "Gridding ones should make ones"


def test_regular_grid_to_regular_grid(simple_grid_ds_12H_05, simple_grid_ds_24H_01):
    regridded = gridding.grid_to_regular_grid(simple_grid_ds_12H_05, simple_grid_ds_24H_01)
    print(regridded)
    xr.testing.assert_allclose(
            regridded[['lat', 'lon']],
            simple_grid_ds_24H_01[['lat', 'lon']],
    )
    xr.testing.assert_allclose(
            regridded.time,
            simple_grid_ds_12H_05.time,
    )
    
    values = regridded.ssh.values
    print(np.nanmin(values))
    print(np.nanmax(values))
    msk = np.isfinite(values)
    assert msk.sum()>0, "should have finite values"
    np.testing.assert_almost_equal(values[msk], 1.), "Gridding ones should make ones"


