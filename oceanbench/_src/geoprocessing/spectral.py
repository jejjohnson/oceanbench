from typing import List
import xrft
import xarray as xr
from functools import reduce
import pint_xarray


def psd_spacetime(
    ds: xr.Dataset, 
    variable: str, 
    dims: List[str], 
    **kwargs
)-> xr.Dataset:
    
    name = f"{variable}_psd"
    
    # compute PSD err and PSD signal
    psd_signal = xrft.power_spectrum(
        ds[variable],
        dim=dims,
        detrend=kwargs.get("detrend", "linear"),
        window=kwargs.get("window", "tukey"),
        nfactor=kwargs.get("nfactor", 2),
        window_correction=kwargs.get("window_correction", True),
        true_amplitude=kwargs.get("true_amplitude", True),
        truncate=kwargs.get("truncate", True),
    )
    
    return psd_signal.to_dataset(name=variable)


def psd_isotropic(
    ds: xr.DataArray, 
    variable: str, 
    dims: List[str], 
    **kwargs
) -> xr.DataArray:
    name = f"{variable}_psd"
    
    # compute PSD err and PSD signal
    psd_signal = xrft.isotropic_power_spectrum(
        ds[variable],
        dim=dims,
        detrend=kwargs.get("detrend", "linear"),
        window=kwargs.get("window", "tukey"),
        nfactor=kwargs.get("nfactor", 2),
        window_correction=kwargs.get("window_correction", True),
        true_amplitude=kwargs.get("true_amplitude", True),
        truncate=kwargs.get("truncate", True),
    )
    

    original_dims = ds.dims
    psd_signal = psd_signal.to_dataset(name=variable)
    
    #psd_signal = psd_signal.pint.quantify(
    #    {variable: str(ds[variable].pint.units)})
    
    
    return psd_signal
