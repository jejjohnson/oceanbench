import typing as tp
import numpy as np
import pandas as pd
import pint
from dataclasses import dataclass
from xarray_dataclasses import Coordof, Dataof, Attr, Coord, Data, Name, AsDataArray, AsDataset, Dataof, asdataarray
from metpy.units import units
import oceanbench._src.geoprocessing.geostrophic as geocalc


TIME = tp.Literal["time"]
LAT = tp.Literal["lat"]
LON = tp.Literal["lon"]

@dataclass
class TimeAxis:
    data: Data[TIME, tp.Literal["datetime64[ns]"]]
    name: Name[str] = "time"
    long_name: Attr[str] = "Date"

@dataclass
class LatitudeAxis:
    data: Data[LAT, np.float32]
    name: Name[str] = "lat"
    standard_name: Attr[str] = "latitude"
    long_name: Attr[str] = "Latitude"
    units: Attr[str] = str(units.degree_north)

@dataclass
class LongitudeAxis:
    data: Data[LON, np.float32]
    name: Name[str] = "lon"
    standard_name: Attr[str] = "longitude"
    long_name: Attr[str] = "Longitude"
    units: Attr[str] = str(units.degree_east)


@dataclass
class SSH2DT:
    data: Data[tuple[TIME, LAT, LON], np.float32]
    time: Coordof[TimeAxis] = 0
    lat: Coordof[LatitudeAxis] = 0
    lon: Coordof[LongitudeAxis] = 0
    name: Name[str] = "ssh"
    units: Attr[str] = "m"
    standard_name: Attr[str] = "sea_surface_height"
    long_name: Attr[str] = "Sea Surface Height"
    
    
def get_ssh_data():
    lon = np.linspace(-65, -54, 10)
    lat = np.linspace(32, 44, 10)
    tmin, tmax = pd.to_datetime("2013-01-01"), pd.to_datetime("2013-01-10")
    dt = pd.to_timedelta(1, "D")
    time = np.arange(tmin, tmax, dt)
    rng = np.random.RandomState(seed=123)
    data = rng.randn(time.shape[0], lat.shape[0], lon.shape[0])
    ds = SSH2DT(data=data * units.meter, lat=lat, lon=lon, time=time)
    
    ds = asdataarray(ds).to_dataset()
    
    return ds


def test_streamfunction():
    ds = get_ssh_data()
    dims = ds.dims
    ds = geocalc.streamfunction(ds, "ssh")
    
    # check variables
    assert "psi" in ds
    assert "ssh" in ds
    # check attributes
    assert ds["psi"].attrs["long_name"] == "Stream Function"
    assert ds["psi"].attrs["standard_name"] == "stream_function"
    # check units
    assert ds["psi"].metpy.units == pint.Unit("meter2/second")
    # check dims
    assert dims == ds.dims
    
    
def test_velocities():
    ds = get_ssh_data()
    
    dims = ds.dims
    ds = geocalc.streamfunction(ds, "ssh")
    ds = geocalc.geostrophic_velocities(ds, variable="psi")
    
    print(ds["psi"].metpy.units, ds["ssh"].metpy.units, ds["u"].metpy.units)
    # check variables
    assert "psi" in ds
    assert "ssh" in ds
    # check attributes
    assert ds["u"].attrs["long_name"] == "Zonal Velocity"
    assert ds["u"].attrs["standard_name"] == "zonal_velocity"
    assert ds["v"].attrs["long_name"] == "Meridional Velocity"
    assert ds["v"].attrs["standard_name"] == "meridional_velocity"
    # check units
    assert ds["u"].metpy.units == pint.Unit("meter/second")
    assert ds["v"].metpy.units == pint.Unit("meter/second")
    # check dims
    assert dims == ds.dims
    
    
def test_kinetic_energy():
    ds = get_ssh_data()
    dims = ds.dims
    
    ds = geocalc.streamfunction(ds, "ssh")
    ds = geocalc.geostrophic_velocities(ds, variable="psi")
    ds = geocalc.kinetic_energy(ds, variables=["u", "v"])
    
    # check variables
    assert "psi" in ds
    assert "ssh" in ds
    assert "u" in ds
    assert "v" in ds
    assert "ke" in ds
    # check attributes
    assert ds["ke"].attrs["long_name"] == "Kinetic Energy"
    assert ds["ke"].attrs["standard_name"] == "kinetic_energy"
    # check units
    assert ds["ke"].metpy.units == pint.Unit("meter2/second2")
    # check dims
    assert dims == ds.dims
    
    
def test_relative_vorticity():
    ds = get_ssh_data()
    dims = ds.dims
    ds = geocalc.streamfunction(ds, "ssh")
    ds = geocalc.geostrophic_velocities(ds, variable="psi")
    ds = geocalc.relative_vorticity(ds, variables=["u", "v"])
    
    # check variables
    assert "psi" in ds
    assert "ssh" in ds
    assert "u" in ds
    assert "v" in ds
    assert "vort_r" in ds
    # check attributes
    assert ds["vort_r"].attrs["long_name"] == "Relative Vorticity"
    assert ds["vort_r"].attrs["standard_name"] == "relative_vorticity"
    # check units
    assert ds["vort_r"].metpy.units == pint.Unit("1/second")
    # check dims
    assert dims == ds.dims
    
    
def test_absolute_vorticity():
    ds = get_ssh_data()
    dims = ds.dims
    
    ds = geocalc.streamfunction(ds, "ssh")
    ds = geocalc.geostrophic_velocities(ds, variable="psi")
    ds = geocalc.absolute_vorticity(ds, variables=["u", "v"])
    
    # check variables
    assert "psi" in ds
    assert "ssh" in ds
    assert "u" in ds
    assert "v" in ds
    assert "vort_a" in ds
    # check attributes
    assert ds["vort_a"].attrs["long_name"] == "Absolute Vorticity"
    assert ds["vort_a"].attrs["standard_name"] == "absolute_vorticity"
    # check units
    assert ds["vort_a"].metpy.units == pint.Unit("1/second")
    # check dims
    assert dims == ds.dims
    
    
def test_divergence():
    ds = get_ssh_data()
    dims = ds.dims
    
    ds = geocalc.streamfunction(ds, "ssh")
    ds = geocalc.geostrophic_velocities(ds, variable="psi")
    ds = geocalc.divergence(ds, variables=["u", "v"])
    
    # check variables
    assert "psi" in ds
    assert "ssh" in ds
    assert "u" in ds
    assert "v" in ds
    assert "div" in ds
    # check attributes
    assert ds["div"].attrs["long_name"] == "Divergence"
    assert ds["div"].attrs["standard_name"] == "divergence"
    # check units
    assert ds["div"].metpy.units == pint.Unit("1/second")
    # check dims
    assert dims == ds.dims
    
    
def test_enstrophy():
    ds = get_ssh_data()
    
    dims = ds.dims
    ds = geocalc.streamfunction(ds, "ssh")
    ds = geocalc.geostrophic_velocities(ds, variable="psi")
    ds = geocalc.relative_vorticity(ds, variables=["u", "v"])
    ds = geocalc.enstrophy(ds, variable="vort_r")
    
    # check variables
    assert "psi" in ds
    assert "ssh" in ds
    assert "u" in ds
    assert "v" in ds
    assert "ens" in ds
    # check attributes
    assert ds["ens"].attrs["long_name"] == "Enstrophy"
    assert ds["ens"].attrs["standard_name"] == "enstrophy"
    # check units
    assert ds["ens"].metpy.units == pint.Unit("1/second2")
    # check dims
    assert dims == ds.dims
    
    
def test_shear_strain():
    ds = get_ssh_data()
    dims = ds.dims
    
    ds = geocalc.streamfunction(ds, "ssh")
    ds = geocalc.geostrophic_velocities(ds, variable="psi")
    ds = geocalc.shear_strain(ds, variables=["u", "v"])
    
    # check variables
    assert "psi" in ds
    assert "ssh" in ds
    assert "u" in ds
    assert "v" in ds
    assert "shear_strain" in ds
    # check attributes
    assert ds["shear_strain"].attrs["long_name"] == "Shear Strain"
    assert ds["shear_strain"].attrs["standard_name"] == "shear_strain"
    # check units
    assert ds["shear_strain"].metpy.units == pint.Unit("1/second")
    # check dims
    assert dims == ds.dims
    
    
def test_tensor_strain():
    ds = get_ssh_data()
    
    dims = ds.dims
    
    ds = geocalc.streamfunction(ds, "ssh")
    ds = geocalc.geostrophic_velocities(ds, variable="psi")
    ds = geocalc.tensor_strain(ds, variables=["u", "v"])
    
    # check variables
    assert "psi" in ds
    assert "ssh" in ds
    assert "u" in ds
    assert "v" in ds
    assert "tensor_strain" in ds
    # check attributes
    assert ds["tensor_strain"].attrs["long_name"] == "Tensor Strain"
    assert ds["tensor_strain"].attrs["standard_name"] == "tensor_strain"
    # check units
    assert ds["tensor_strain"].metpy.units == pint.Unit("1/second")
    # check dims
    assert dims == ds.dims
    
    
def test_strain_magnitude():
    ds = get_ssh_data()
    dims = ds.dims
    ds = geocalc.streamfunction(ds, "ssh")
    ds = geocalc.geostrophic_velocities(ds, variable="psi")
    ds = geocalc.strain_magnitude(ds, variables=["u", "v"])
    
    # check variables
    assert "psi" in ds
    assert "ssh" in ds
    assert "u" in ds
    assert "v" in ds
    assert "strain" in ds
    # check attributes
    assert ds["strain"].attrs["long_name"] == "Strain"
    assert ds["strain"].attrs["standard_name"] == "strain"
    # check units
    assert ds["strain"].metpy.units == pint.Unit("1/second")
    # check dims
    assert dims == ds.dims
    
    
def test_okubo_weiss_param():
    ds = get_ssh_data()
    
    dims = ds.dims
    ds = geocalc.streamfunction(ds, "ssh")
    ds = geocalc.geostrophic_velocities(ds, variable="psi")
    ds = geocalc.okubo_weiss_param(ds, variables=["u", "v"])
    
    # check variables
    assert "psi" in ds
    assert "ssh" in ds
    assert "u" in ds
    assert "v" in ds
    assert "ow" in ds
    # check attributes
    assert ds["ow"].attrs["long_name"] == "Okubo-Weiss Parameter"
    assert ds["ow"].attrs["standard_name"] == "okubo_weiss_parameter"
    # check units
    assert ds["ow"].metpy.units == pint.Unit("1/second2")
    # check dims
    assert dims == ds.dims
    
    
    