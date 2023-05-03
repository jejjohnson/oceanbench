from typing import Optional, List
import metpy.calc as geocalc
import numpy as np
import xarray as xr
from pint import Quantity
from metpy.constants import earth_gravity as GRAVITY


def streamfunction(
    ds: xr.Dataset,
    variable: str = "ssh",
    g: Optional[float] = None,
    f0: Optional[float] = None,
) -> xr.Dataset:
    """Calculates the stream function from SSH values

    Eq:
        η = (g/f₀) Ψ

    Args:
        ds (xr.Dataset): the sea surface height [m]
        variable (str): the variable name to use for SSH
        f0 (Array|float): the coriolis parameter
        g (float): the acceleration due to gravity

    Returns:
        ds (xr.Dataset): the xr.Dataset with the stream function variable
            (psi)
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

    psi = (g / f0) * ds[variable]

    ds["psi"] = psi
    ds["psi"].attrs["long_name"] = "Stream Function"
    ds["psi"].attrs["standard_name"] = "stream_function"

    return ds


def geostrophic_velocities(ds, variable: str = "psi"):
    """Calculates the geostrophic velocities from the stream function.

    Eqn:
        u = -∂Ψ/∂y
        v =  ∂Ψ/∂x

    Args:
        ds (xr.Dataset): the xr.Dataset with the stream function (psi)
            variable
        variable (str): the variable, (default="psi")

    Return:
        ds (xr.Dataset): the xr.Dataset with the u,v variables [m/s^2]
    """
    ds = ds.copy()

    dpsi_dx, dpsi_dy = geocalc.geospatial_gradient(
        f=ds[variable], latitude=ds.lat, longitude=ds.lon
    )

    ds["u"] = (("time", "lat", "lon"), -dpsi_dy)
    ds["u"].attrs["long_name"] = "Zonal Velocity"
    ds["u"].attrs["standard_name"] = "zonal_velocity"

    ds["v"] = (("time", "lat", "lon"), dpsi_dx)
    ds["v"].attrs["long_name"] = "Meridional Velocity"
    ds["v"].attrs["standard_name"] = "meridional_velocity"

    return ds


def kinetic_energy(ds, variables: List[str] = ["u", "v"]):
    """Calculates the kinetic energy via an
    arbitrary magnitude of the u and v velocities

    Eqn:
        ke(u,v) = 0.5 * (u² + v²)

    Args:
        ds (xr.Dataset): the xr.Dataset with the u,v velocity variables
        variables (List[str]): the variable namse to use for the kinetic
            energy, default=["u","v"]

    Returns:
        ds (xr.Dataset): the xr.Dataset with the kinetic energy
            ("ke")
    """
    ds = ds.copy()

    ds["ke"] = 0.5 * (ds[variables[0]] ** 2 + ds[variables[1]] ** 2)

    ds["ke"].attrs["long_name"] = "Kinetic Energy"
    ds["ke"].attrs["standard_name"] = "kinetic_energy"

    return ds


def relative_vorticity(ds, variables: List[str] = ["u", "v"]):
    """Calculates the relative vorticity by using
    finite difference in the y and x direction for the
    u and v velocities respectively

    Eqn:
        ζ = ∂v/∂x - ∂u/∂y

    Args:
        ds (xr.Dataset): the xr.Dataset with the u,v velocity variables
        variables (List[str]): the variable namse to use for the relative
            vorticity, default=["u","v"]

    Returns:
        ds (xr.Dataset): the xr.Dataset with the relative vorticity
            ("vort_r")
    """
    ds = ds.copy()

    ds["vort_r"] = geocalc.vorticity(
        v=ds[variables[0]], u=ds[variables[1]], latitude=ds.lat, longitude=ds.lon
    )

    ds["vort_r"].attrs["long_name"] = "Relative Vorticity"
    ds["vort_r"].attrs["standard_name"] = "relative_vorticity"

    return ds


def absolute_vorticity(ds, variables: List[str] = ["u", "v"]):
    """Calculates the absolute vorticity by using
    finite difference in the y and x direction for the
    u and v velocities respectively

    Eqn:
        |ζ| = ∂v/∂x + ∂u/∂y

    Args:
        ds (xr.Dataset): the xr.Dataset with the u,v velocity variables
        variables (List[str]): the variable namse to use for the absolute
            vorticity, default=["u","v"]

    Returns:
        ds (xr.Dataset): the xr.Dataset with the absolute vorticity
            ("vort_a")
    """
    ds["vort_a"] = geocalc.absolute_vorticity(
        v=ds[variables[0]], u=ds[variables[1]], latitude=ds.lat, longitude=ds.lon
    )

    ds["vort_a"].attrs["long_name"] = "Absolute Vorticity"
    ds["vort_a"].attrs["standard_name"] = "absolute_vorticity"

    return ds


def divergence(ds, variables: List[str] = ["u", "v"]):
    """Calculates the divergence by using
    finite difference in the y and x direction for the
    u and v velocities respectively

    Eqn:
        div(u)= ∂u/∂x + ∂v/∂y

    Args:
        ds (xr.Dataset): the xr.Dataset with the u,v velocity variables
        variables (List[str]): the variable namse to use for the divergence
            default=["u","v"]

    Returns:
        ds (xr.Dataset): the xr.Dataset with the divergence
            ("div")
    """
    ds = ds.copy()

    ds["div"] = geocalc.divergence(
        v=ds[variables[0]], u=ds[variables[1]], latitude=ds.lat, longitude=ds.lon
    )

    ds["div"].attrs["long_name"] = "Divergence"
    ds["div"].attrs["standard_name"] = "divergence"

    return ds


def enstrophy(ds, variable: str = "vort_r"):
    """Calculates the potential energy via an
    arbitrary magnitude of the potential vorticity

    Eqn:
        pq(q) = 0.5 (q²)

    Args:
        ds (xr.Dataset): the xr.Dataset with the relative vorticity variable
        variables (List[str]): the variable name to use for the vorticity variable
            default=["vort_r"]

    Returns:
        ds (xr.Dataset): the xr.Dataset with the enstrophy
            ("ens")
    """
    ds = ds.copy()

    ds["ens"] = 0.5 * (ds[variable] ** 2)

    ds["ens"].attrs["long_name"] = "Enstrophy"
    ds["ens"].attrs["standard_name"] = "enstrophy"

    return ds


def coriolis_normalized(
    ds,
    variable: str,
    f0: Optional[float] = None,
):
    """This function normalizes an arbitrary variable
    with the Corilios parameter. This is common for the strain
    and the vorticity.

    Equation:
        f = variable/2Ω sinθ

    Args:
        ds (xr.Dataset): the xr.Dataset with the variable
        variables (List[str]): the variable name to use for the variable
        f0 (Array|float): the coriolis parameter

    Returns:
        ds (xr.Dataset): the xr.Dataset with the normalized variable
    """
    ds = ds.copy()

    if f0 is None:
        f0 = geocalc.coriolis_parameter(latitude=np.deg2rad(ds.lat)).mean()
    else:
        f0 = Quantity(f0, "1/s")

    ds[variable] = ds[variable] / f0

    return ds


def shear_strain(ds, variables: List[str] = ["u", "v"]):
    """
     Calculates the shear strain by using
    finite difference in the y and x direction for the
    u and v velocities respectively

    Eqn:
        Sₛ = ∂v/∂x + ∂u/∂y

    Args:
        ds (xr.Dataset): the xr.Dataset with the u,v velocity variables
        variables (List[str]): the variable namse to use for the shear
            strain, default=["u","v"]

    Returns:
        ds (xr.Dataset): the xr.Dataset with the shear strain
            ("shear_strain")
    """
    ds = ds.copy()

    ds["shear_strain"] = geocalc.shearing_deformation(
        v=ds[variables[0]], u=ds[variables[1]], latitude=ds.lat, longitude=ds.lon
    )

    ds["shear_strain"].attrs["long_name"] = "Shear Strain"
    ds["shear_strain"].attrs["standard_name"] = "shear_strain"

    return ds


def tensor_strain(ds, variables: List[str] = ["u", "v"]):
    """
     Calculates the tensor strain by using
    finite difference in the y and x direction for the
    u and v velocities respectively

    Eqn:
        Sₙ = ∂u/∂x - ∂v/∂y

    Args:
        ds (xr.Dataset): the xr.Dataset with the u,v velocity variables
        variables (List[str]): the variable namse to use for the tensor strain
            strain, default=["u","v"]

    Returns:
        ds (xr.Dataset): the xr.Dataset with the tensor strain
            ("tensor_strain")
    """

    ds = ds.copy()

    ds["tensor_strain"] = geocalc.stretching_deformation(
        v=ds[variables[0]], u=ds[variables[1]], latitude=ds.lat, longitude=ds.lon
    )

    ds["tensor_strain"].attrs["long_name"] = "Tensor Strain"
    ds["tensor_strain"].attrs["standard_name"] = "tensor_strain"
    return ds


def strain_magnitude(ds, variables: List[str] = ["u", "v"]):
    """Calculates the strain by using
    finite difference in the y and x direction for the
    u and v velocities respectively. Strain is the addition of the
    squared tensor strain and shear strain terms respectively.

    Eqn:
        σₛ = Sₙ² + Sₛ²

    Args:
        ds (xr.Dataset): the xr.Dataset with the u,v velocity variables
        variables (List[str]): the variable namse to use for the strain
            magnitude, default=["u","v"]

    Returns:
        ds (xr.Dataset): the xr.Dataset with the strain magnitude
            ("strain")
    """

    ds = ds.copy()

    ds["strain"] = geocalc.total_deformation(
        v=ds[variables[0]], u=ds[variables[1]], latitude=ds.lat, longitude=ds.lon
    )

    ds["strain"].attrs["long_name"] = "Strain"
    ds["strain"].attrs["standard_name"] = "strain"

    return ds


def okubo_weiss_param(ds, variables: List[str] = ["u", "v"]):
    """Calculates the Okubo-Weiss parameter by using
    finite difference in the y and x direction for the
    u and v velocities respectively. The Okubo-Weiss parameter
    is the difference between the strain and the divergence.

    Eqn:
        ow = σₛ - div(u)²

    Args:
        ds (xr.Dataset): the xr.Dataset with the u,v velocity variables
        variables (List[str]): the variable namse to use for the Okubo
            Weiss, default=["u","v"]

    Returns:
        ds (xr.Dataset): the xr.Dataset with the Okubo-Weiss
            ("ow")
    """
    ds = ds.copy()

    sigma_n = tensor_strain(ds, variables=variables)["tensor_strain"]
    sigma_s = shear_strain(ds, variables=variables)["shear_strain"]
    vort_r = relative_vorticity(ds, variables=variables)["vort_r"]

    ds["ow"] = sigma_n**2 + sigma_s**2 - vort_r**2

    ds["ow"].attrs["long_name"] = "Okubo-Weiss Parameter"
    ds["ow"].attrs["standard_name"] = "okubo_weiss_parameter"

    return ds
