import autoroot
import xarray as xr
import pint_xarray
from oceanbench._src.geoprocessing import geostrophic
from metpy.units import units


def calculate_physical_quantities(ds: xr.Dataset, variable: str):

    ds = ds.pint.dequantify()

    if variable == "sea_surface_height":
        return ds
    elif variable == "streamfunction":
        ds["ssh"] = ds.ssh * units.meters
        return geostrophic.streamfunction(ds, "ssh")

    elif variable == "kinetic_energy":
        ds["ssh"] = ds.ssh * units.meters
        ds = geostrophic.streamfunction(ds, "ssh")
        ds = geostrophic.geostrophic_velocities(ds, variable="psi")
        return geostrophic.kinetic_energy(ds, variables=["u", "v"])
    elif variable == "relative_vorticity":
        ds["ssh"] = ds.ssh * units.meters
        ds = geostrophic.streamfunction(ds, "ssh")
        ds = geostrophic.geostrophic_velocities(ds, variable="psi")
        ds = geostrophic.relative_vorticity(ds, variables=["u", "v"])
        ds = geostrophic.coriolis_normalized(ds, "vort_r")
        return ds

    elif variable == "strain":
        ds["ssh"] = ds.ssh * units.meters
        ds = geostrophic.streamfunction(ds, "ssh")
        ds = geostrophic.geostrophic_velocities(ds, variable="psi")
        ds = geostrophic.strain_magnitude(ds, variables=["u", "v"])
        ds = geostrophic.coriolis_normalized(ds, "strain")
        return ds

    else:
        raise ValueError(f"Unrecognized variable...: {variable}")