import xarray as xr


def remove_swath_dimension(ds: xr.Dataset, name: str="nC") -> xr.Dataset:
    """Removes the swath dimension fromt he SWOT
    simulations
    
    Args:
        ds (xr.Dataset): an xr.Dataset where we have the swath dimension
            "nC" and the time dimensions, "time"
        name (str): the name of the swath dimension, default="nC"
    
    Returns:
        ds (xr.Dataset): an xr.Dataset in the alongtrack format
    """
    
    msg = "mismatch in dimensions to collapse"
    msg += f"\nName: {name} | Dims: {ds.dims}"
    assert name in ds.dims, msg
    
    return ds.rename({"time": "z"}).stack(time=(name, "z")).set_index({"time": "z"}).reset_coords([name]).sortby("time")