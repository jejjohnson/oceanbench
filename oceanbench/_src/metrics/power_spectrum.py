from typing import List, Optional
import xarray as xr
from oceanbench._src.preprocessing.mean import xr_cond_average
from typing import List
import xrft
import xarray as xr
from functools import reduce
import pint_xarray
import scipy


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
        scaling=kwargs.get("scaling", "density"),
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
        scaling=kwargs.get("scaling", "density"),
        detrend=kwargs.get("detrend", "linear"),
        window=kwargs.get("window", "tukey"),
        nfactor=kwargs.get("nfactor", 2),
        window_correction=kwargs.get("window_correction", True),
        true_amplitude=kwargs.get("true_amplitude", True),
        truncate=kwargs.get("truncate", True),
    )
    

    original_dims = ds.dims
    psd_signal = psd_signal.to_dataset(name=variable)
    
    
    return psd_signal


def psd_welch(
    da: xr.DataArray,
    variable: str,
    delta_x: float,
    nperseg=None,
    **kwargs
) -> xr.DataArray:
    
    da = da.copy()
    
    wavenumber, psd = scipy.signal.welch(
        da[variable].values.flatten(),
        fs=1./delta_x,
        nperseg=nperseg,
        scaling=kwargs.pop("scaling", "density"),
        noverlap=kwargs.pop("noverlap", 0)
    )
    
    ds = xr.DataArray(
        data=psd,
        coords={"wavenumber": wavenumber},
        name=variable
    )
    
    
    return ds.to_dataset()


def psd_welch_error(
    da: xr.DataArray,
    variable: str,
    variable_ref: str, 
    delta_x: float,
    nperseg=None,
    **kwargs
) -> xr.DataArray:
    
    da = da.copy()
        
    wavenumber, psd = scipy.signal.welch(
        da[variable].values.flatten() - da[variable_ref].values.flatten(),
        fs=1./delta_x,
        nperseg=nperseg,
        scaling=kwargs.pop("scaling", "density"),
        noverlap=kwargs.pop("noverlap", 0)
    )
    
    da = xr.DataArray(
        data=psd,
        coords={"wavenumber": wavenumber},
        name="error"
    )
    return da.to_dataset()


def psd_welch_score(
    da: xr.DataArray,
    variable: str,
    variable_ref: str, 
    delta_x: float,
    nperseg=None,
    **kwargs
) -> xr.DataArray:
    
    da = da.copy()
        
    
    # error
    ds = psd_welch_error(
        da=da, variable=variable,
        variable_ref=variable_ref, 
        delta_x=delta_x,
        nperseg=nperseg,
        **kwargs
    )
    
    # differ
    _, psd = scipy.signal.welch(
        da[variable_ref].values.flatten(),
        fs=1./delta_x,
        nperseg=nperseg,
        scaling=kwargs.pop("scaling", "density"),
        noverlap=kwargs.pop("noverlap", 0)
    )
    
    ds["score"] = (("wavenumber"), 1 - ds["error"].values / psd)
    
    return ds






def psd_isotropic_error(
    da: xr.DataArray, 
    da_ref: xr.DataArray, 
    variable: str,
    psd_dims: List[str],
    avg_dims: Optional[List[str]]=None,
) -> xr.DataArray:
    psd_error = psd_isotropic(ds=da_ref - da, variable=variable, dims=psd_dims)
    if avg_dims is not None:
        psd_error =  xr_cond_average(psd_error, dims=avg_dims, drop=True)
    return psd_error

def psd_spacetime_error(
    da: xr.DataArray,
    da_ref: xr.DataArray,
    variable: str,
    psd_dims: List[str],
    avg_dims: Optional[List[str]]=None,
) -> xr.DataArray:
    psd_error = psd_spacetime(ds=da_ref - da, variable=variable, dims=psd_dims)
    if avg_dims is not None:
        psd_error = xr_cond_average(psd_error, dims=avg_dims, drop=True)
    return psd_error

def psd_isotropic_score(
    da: xr.DataArray, 
    da_ref: xr.DataArray, 
    variable: str,
    psd_dims: List[str],
    avg_dims: List[str]=None,
) -> xr.DataArray:

    
    # error
    score = psd_isotropic_error(
        da=da, da_ref=da_ref, variable=variable, psd_dims=psd_dims, avg_dims=avg_dims
    )
    
    # reference signal
    psd_ref = psd_isotropic(ds=da_ref, variable=variable, dims=psd_dims)
    
    if avg_dims is not None:
        psd_ref = xr_cond_average(psd_ref,dims=avg_dims)
    
    if score[variable].shape != psd_ref[variable].shape:
        psd_ref = psd_ref.interp_like(score)
    

    # normalized score
    score = 1.0 - (score / psd_ref)
    
    return score


def psd_spacetime_score(
    da: xr.DataArray, 
    da_ref: xr.DataArray, 
    variable: str,
    psd_dims: List[str],
    avg_dims: List[str]=None,
) -> xr.DataArray:

    
    # error
    score = psd_spacetime_error(
        da=da, da_ref=da_ref, variable=variable, psd_dims=psd_dims, avg_dims=avg_dims
    )
    
    # reference signal
    psd_ref = psd_spacetime(ds=da_ref, variable=variable, dims=psd_dims)
    
    if avg_dims is not None:
        psd_ref = xr_cond_average(psd_ref,dims=avg_dims)
    
    if score[variable].shape != psd_ref[variable].shape:
        psd_ref = psd_ref.interp_like(score)

    # normalized score
    score = 1.0 - (score / psd_ref)
    
    return score