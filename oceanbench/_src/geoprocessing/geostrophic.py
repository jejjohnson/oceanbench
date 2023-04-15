from typing import Optional, List
import metpy
import metpy.calc as geocalc
import numpy as np
import xarray as xr
from pint import Quantity
from metpy.constants import earth_gravity as GRAVITY


def streamfunction(
    ds: xr.Dataset, 
    variable: str="ssh", 
    g: Optional[float]=None,
    f0: Optional[float]=None,
) -> xr.Dataset:
    """Calculates the stream function from SSH values
    
    Args:
        ds (xr.Dataset): an xr.Dataset with the variable
        g (float): the gravitational constant
    
    Returns:
        ds (xr.Dataset): the xr.Dataset with the stream function variable
    """
    
    ds = ds.copy()
    
    # extract dimensions
    dims = ds.dims
    
    if f0 is None:
        f0 = geocalc.coriolis_parameter(latitude=np.deg2rad(ds.lat)).mean()
    else:
        f0 = Quantity(f0, "1/s")
    
    
    if g is None:
        g = GRAVITY
    else:
        g *= GRAVITY.units
        
    psi = (g/f0) * ds[variable]
            
    ds["psi"] = psi
    ds["psi"].attrs["long_name"] = "Stream Function"
    ds["psi"].attrs["standard_name"] = "stream_function"
    
    return ds

def geostrophic_velocities(ds, variable: str="psi"):
    
    ds = ds.copy()
    
    dpsi_dx, dpsi_dy = geocalc.geospatial_gradient(
        f=ds[variable], latitude=ds.lat, longitude=ds.lon
    )
    
    ds["u"] = (("time", "lat", "lon"), - dpsi_dy)
    ds["u"].attrs["long_name"] = "Zonal Velocity"
    ds["u"].attrs["standard_name"] = "zonal_velocity"
    
    ds["v"] = (("time", "lat", "lon"), dpsi_dx)
    ds["v"].attrs["long_name"] = "Meridional Velocity"
    ds["v"].attrs["standard_name"] = "meridional_velocity"
    
    return ds


def kinetic_energy(ds, variables: List[str]=["u", "v"]):
    
    ds = ds.copy()
    
    ds["ke"] = 0.5 * (ds[variables[0]]**2 + ds[variables[1]]**2)
    
    ds["ke"].attrs["long_name"] = "Kinetic Energy"
    ds["ke"].attrs["standard_name"] = "kinetic_energy"
    
    return ds

def relative_vorticity(ds, variables: List[str]=["u", "v"]):
    
    ds = ds.copy()
    
    ds["vort_r"] = geocalc.vorticity(
        v=ds["v"], u=ds["v"], 
        latitude=ds.lat, longitude=ds.lon
    )
    
    ds["vort_r"].attrs["long_name"] = "Relative Vorticity"
    ds["vort_r"].attrs["standard_name"] = "relative_vorticity"
    
    return ds

def absolute_vorticity(ds, variables: List[str]=["u", "v"]):
    
    ds["vort_a"] = geocalc.absolute_vorticity(
        v=ds["v"], u=ds["v"], 
        latitude=ds.lat, longitude=ds.lon
    )
    
    ds["vort_a"].attrs["long_name"] = "Absolute Vorticity"
    ds["vort_a"].attrs["standard_name"] = "absolute_vorticity"
    
    return ds


def divergence(ds, variables: List[str]=["u", "v"]):
    
    ds = ds.copy()
    
    ds["div"] = geocalc.divergence(
        v=ds["v"], u=ds["v"], 
        latitude=ds.lat, longitude=ds.lon
    )
    
    ds["div"].attrs["long_name"] = "Divergence"
    ds["div"].attrs["standard_name"] = "divergence"
    
    
    return ds


def enstrophy(ds, variable: str="vort_r"):
    
    ds = ds.copy()
    
    ds["ens"] = 0.5 * (ds[variable]**2)
    
    ds["ens"].attrs["long_name"] = "Enstrophy"
    ds["ens"].attrs["standard_name"] = "enstrophy"
    
    return ds


def coriolis_normalized(ds, variable: str, f0: Optional[float]=None,):
    
    ds = ds.copy()
    
    if f0 is None:
        f0 = geocalc.coriolis_parameter(latitude=np.deg2rad(ds.lat)).mean()
    else:
        f0 = Quantity(f0, "1/s")
    
    ds[variable] = ds[variable] / f0
    
    return ds





def shear_strain(ds, variables: List[str]=["u", "v"]):
    """Sometimes called:
    * shearing deformation
    
    """
    
    ds = ds.copy()
    
    ds["shear_strain"] = geocalc.shearing_deformation(
        v=ds[variables[0]], u=ds[variables[1]], 
        latitude=ds.lat, longitude=ds.lon
    )
    
    ds["shear_strain"].attrs["long_name"] = "Shear Strain"
    ds["shear_strain"].attrs["standard_name"] = "shear_strain"
    
    return ds


def shear_strain(ds, variables: List[str]=["u", "v"]):
    """Sometimes called:
    * stretching deformation
    
    """
    
    ds = ds.copy()
    
    ds["shear_strain"] = geocalc.shearing_deformation(
        v=ds[variables[0]], u=ds[variables[1]], 
        latitude=ds.lat, longitude=ds.lon
    )
    
    ds["shear_strain"].attrs["long_name"] = "Shear Strain"
    ds["shear_strain"].attrs["standard_name"] = "shear_strain"
    
    return ds


def tensor_strain(ds, variables: List[str]=["u", "v"]):
    """Sometimes called:
    * stretching deformation
    
    """
    
    ds = ds.copy()
    
    ds["tensor_strain"] = geocalc.stretching_deformation(
        v=ds[variables[0]], u=ds[variables[1]], 
        latitude=ds.lat, longitude=ds.lon
    )
    
    ds["tensor_strain"].attrs["long_name"] = "Tensor Strain"
    ds["tensor_strain"].attrs["standard_name"] = "tensor_strain"
    
    return ds


def strain_magnitude(ds, variables: List[str]=["u", "v"]):
    """Sometimes called:
    * stretching deformation
    
    """
    
    ds = ds.copy()
    
    ds["strain"] = geocalc.total_deformation(
        v=ds[variables[0]], u=ds[variables[1]], 
        latitude=ds.lat, longitude=ds.lon
    )
    
    ds["strain"].attrs["long_name"] = "Strain"
    ds["strain"].attrs["standard_name"] = "strain"
    
    return ds


def okubo_weiss_param(ds, variables: List[str]=["u", "v"]):
    
    ds = ds.copy()
    
    sigma_n = tensor_strain(ds, variables=variables)["tensor_strain"]
    sigma_s = shear_strain(ds, variables=variables)["shear_strain"]
    vort_r = relative_vorticity(ds, variables=variables)["vort_r"]
    
    ds["ow"] = sigma_n**2 + sigma_s**2 - vort_r**2
    
    ds["ow"].attrs["long_name"] = "Okubo-Weiss Parameter"
    ds["ow"].attrs["standard_name"] = "okubo_weiss_parameter"
    
    return ds