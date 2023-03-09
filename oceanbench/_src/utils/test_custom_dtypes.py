import pytest
import oceanbench._src.utils.custom_dtypes as cdtypes
import numpy as np
import pandas as pd


@pytest.fixture
def x_coord():
    return dict(x_min=-10, x_max=10, dx=10)


@pytest.fixture
def lon_coord():
    return dict(lon_min=-180, lon_max=180, dlon=20)


@pytest.fixture
def lat_coord():
    return dict(lat_min=-90, lat_max=90, dlat=20)


@pytest.fixture
def time_coord():
    return dict(t_min="2012-01-01", t_max="2012-01-30", dt="1D")


def test_coordinate_axis(x_coord):

    data = np.arange(x_coord["x_min"], x_coord["x_max"] + x_coord["dx"], x_coord["dx"])

    coords = cdtypes.CoordinateAxis.init_from_bounds(x_coord["x_min"], x_coord["x_max"], x_coord["dx"])

    np.testing.assert_array_equal(coords.data, data)
    assert coords.data.dtype == data.dtype
    assert coords.ndim == (len(data))


def test_longitude_axis(lon_coord):
    data = np.arange(lon_coord["lon_min"], lon_coord["lon_max"] + lon_coord["dlon"], lon_coord["dlon"])

    coords = cdtypes.LongitudeAxis.init_from_bounds(lon_coord["lon_min"], lon_coord["lon_max"], lon_coord["dlon"])

    np.testing.assert_array_equal(coords.data, data)
    assert coords.name == "lon"
    assert coords.long_name == "Longitude"
    assert coords.standard_name == "longitude"
    assert coords.units == "degrees_east"
    assert coords.ndim == (len(data))


def test_latitude_axis(lat_coord):
    data = np.arange(lat_coord["lat_min"], lat_coord["lat_max"] + lat_coord["dlat"], lat_coord["dlat"])

    coords = cdtypes.LatitudeAxis.init_from_bounds(lat_coord["lat_min"], lat_coord["lat_max"], lat_coord["dlat"])

    np.testing.assert_array_equal(coords.data, data)
    assert coords.name == "lat"
    assert coords.long_name == "Latitude"
    assert coords.standard_name == "latitude"
    assert coords.units == "degrees_west"
    assert coords.ndim == (len(data))


def test_time_axis(time_coord):
    t_min = pd.to_datetime(time_coord["t_min"])
    t_max = pd.to_datetime(time_coord["t_max"])
    dt = pd.to_timedelta(time_coord["dt"]) 
    data = np.arange(t_min, t_max + dt, dt)

    coords = cdtypes.TimeAxis.init_from_bounds(time_coord["t_min"], time_coord["t_max"], time_coord["dt"])

    np.testing.assert_array_equal(coords.data, data)
    assert coords.name == "time"
    assert coords.long_name == "Date"
    assert coords.ndim == (len(data))


def test_grid2d(lat_coord, lon_coord):

    lon_coord = cdtypes.LongitudeAxis.init_from_bounds(lon_coord["lon_min"], lon_coord["lon_max"], lon_coord["dlon"])
    lat_coord = cdtypes.LatitudeAxis.init_from_bounds(lat_coord["lat_min"], lat_coord["lat_max"], lat_coord["dlat"])

    grid = cdtypes.Grid2D(lon=lon_coord, lat=lat_coord)
    manual_grid = np.meshgrid(lat_coord.data, lon_coord.data, indexing="ij")
    manual_grid = np.stack(manual_grid, axis=-1)

    assert grid.ndim == (lat_coord.ndim, lon_coord.ndim)
    np.testing.assert_array_equal(grid.grid, manual_grid)
    

def test_grid2dt(lat_coord, lon_coord, time_coord):

    lon_coord = cdtypes.LongitudeAxis.init_from_bounds(lon_coord["lon_min"], lon_coord["lon_max"], lon_coord["dlon"])
    lat_coord = cdtypes.LatitudeAxis.init_from_bounds(lat_coord["lat_min"], lat_coord["lat_max"], lat_coord["dlat"])
    time_coord = cdtypes.TimeAxis.init_from_bounds(time_coord["t_min"], time_coord["t_max"], time_coord["dt"])

    grid = cdtypes.Grid2DT(lon=lon_coord, lat=lat_coord, time=time_coord)
    # manual_grid = np.meshgrid(lat_coord.data, lon_coord.data, indexing="ij")
    # manual_grid = np.stack(manual_grid, axis=-1)

    assert grid.ndim == (time_coord.ndim, lat_coord.ndim, lon_coord.ndim)

def test_ssh_2d(lat_coord, lon_coord):

    lon_coord = cdtypes.LongitudeAxis.init_from_bounds(lon_coord["lon_min"], lon_coord["lon_max"], lon_coord["dlon"])
    lat_coord = cdtypes.LatitudeAxis.init_from_bounds(lat_coord["lat_min"], lat_coord["lat_max"], lat_coord["dlat"])

    ssh = cdtypes.SSH2D.init_from_axis(lon=lon_coord, lat=lat_coord)
    # manual_grid = np.meshgrid(lat_coord.data, lon_coord.data, indexing="ij")
    # manual_grid = np.stack(manual_grid, axis=-1)

    assert ssh.ndim == (lat_coord.ndim, lon_coord.ndim)
    np.testing.assert_array_equal(ssh.data, np.ones((lat_coord.ndim, lon_coord.ndim)))
    assert ssh.name == "ssh"
    assert ssh.standard_name == "sea_surface_height"
    assert ssh.long_name == "Sea Surface Height"
    assert ssh.units == "m"


    grid = cdtypes.Grid2D(lon=lon_coord, lat=lat_coord)
    ssh = cdtypes.SSH2D.init_from_grid(grid=grid)

    assert ssh.ndim == (lat_coord.ndim, lon_coord.ndim)
    np.testing.assert_array_equal(ssh.data, np.ones((lat_coord.ndim, lon_coord.ndim)))
    assert ssh.name == "ssh"
    assert ssh.standard_name == "sea_surface_height"
    assert ssh.long_name == "Sea Surface Height"
    assert ssh.units == "m"


def test_ssh_2dt(lat_coord, lon_coord, time_coord):

    lon_coord = cdtypes.LongitudeAxis.init_from_bounds(lon_coord["lon_min"], lon_coord["lon_max"], lon_coord["dlon"])
    lat_coord = cdtypes.LatitudeAxis.init_from_bounds(lat_coord["lat_min"], lat_coord["lat_max"], lat_coord["dlat"])
    time_coord = cdtypes.TimeAxis.init_from_bounds(time_coord["t_min"], time_coord["t_max"], time_coord["dt"])
    
    ssh = cdtypes.SSH2DT.init_from_axis(lon=lon_coord, lat=lat_coord, time=time_coord)
    # manual_grid = np.meshgrid(lat_coord.data, lon_coord.data, indexing="ij")
    # manual_grid = np.stack(manual_grid, axis=-1)

    assert ssh.ndim == (time_coord.ndim, lat_coord.ndim, lon_coord.ndim, )
    np.testing.assert_array_equal(ssh.data, np.ones((time_coord.ndim, lat_coord.ndim, lon_coord.ndim, )))
    assert ssh.name == "ssh"
    assert ssh.standard_name == "sea_surface_height"
    assert ssh.long_name == "Sea Surface Height"
    assert ssh.units == "m"


    grid = cdtypes.Grid2DT(lon=lon_coord, lat=lat_coord, time=time_coord)
    ssh = cdtypes.SSH2DT.init_from_grid(grid=grid)

    assert ssh.ndim == (time_coord.ndim, lat_coord.ndim, lon_coord.ndim)
    np.testing.assert_array_equal(ssh.data, np.ones((time_coord.ndim, lat_coord.ndim, lon_coord.ndim)))
    assert ssh.name == "ssh"
    assert ssh.standard_name == "sea_surface_height"
    assert ssh.long_name == "Sea Surface Height"
    assert ssh.units == "m"
