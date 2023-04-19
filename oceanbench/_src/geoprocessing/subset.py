import xarray as xr

def where_slice(ds: xr.Dataset, variable: str, min_val: float, max_val: float) -> xr.Dataset:
    ds = ds.where(
        (ds[variable] >= min_val) & (ds[variable] <= max_val),
        drop=True
    )
        
    return ds