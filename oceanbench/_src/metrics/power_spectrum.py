from typing import List, Optional
import xarray as xr
from oceanbench._src.preprocessing.mean import xr_cond_average
from oceanbench._src.metrics.utils import find_intercept_1D, find_intercept_2D
from typing import List
import xrft
import xarray as xr
from functools import reduce
import pint_xarray
import scipy


def psd_spacetime(
    ds: xr.Dataset, variable: str, dims: List[str], **kwargs
) -> xr.Dataset:
    """Calculates the PSD with arbitrary dimensions

    PSD_SCORE = PSD(x)

    Args:
        ds (xr.Dataset): the xarray dataset with dimensions
        variable (str): the variable we wish to do the PSD
        dims (List[str]): the dimensions for the PSD
        **kwargs: all key word args for the xrft.power_spectrum

    Returns:
        ds (xr.Dataset): the xr.Dataset with the new frequency dimensions.

    Example:
    >>> psd_spacetime_score(
        da,                 # ssh map
        "ssh",              # variable
        ["time", "lon"],    # dimensions for power spectrum
        )
    """
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
    ds: xr.Dataset, variable: str,
    dims: List[str], **kwargs
) -> xr.Dataset:
    """Calculates the isotropic PSD with arbitrary dimensions

    PSD_SCORE = isoPSD(x)

    Args:
        ds (xr.Dataset): the xarray dataset with dimensions
        variable (str): the variable we wish to do the PSD
        dims (List[str]): the dimensions for the PSD
        **kwargs: all key word args for the xrft.power_spectrum

    Returns:
        ds (xr.Dataset): the xr.Dataset with the new frequency dimensions.

    Example:
    >>> psd_spacetime_score(
        da,                 # ssh map
        "ssh",              # variable
        ["time", "lon"],    # dimensions for isotropic power spectrum
        )
    """
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
    ds: xr.Dataset, variable: str, delta_x: float, nperseg=None, **kwargs
) -> xr.Dataset:
    ds = ds.copy()

    wavenumber, psd = scipy.signal.welch(
        ds[variable].values.flatten(),
        fs=1.0 / delta_x,
        nperseg=nperseg,
        scaling=kwargs.pop("scaling", "density"),
        noverlap=kwargs.pop("noverlap", 0),
    )

    da = xr.DataArray(data=psd, coords={"wavenumber": wavenumber}, name=variable)

    return da.to_dataset()


def psd_welch_error(
    ds: xr.Dataset,
    variable: str,
    variable_ref: str,
    delta_x: float,
    nperseg=None,
    **kwargs,
) -> xr.Dataset:
    ds = ds.copy()

    diff = ds[variable].values.flatten() - ds[variable_ref].values.flatten()

    wavenumber, psd = scipy.signal.welch(
        diff,
        fs=1.0 / delta_x,
        nperseg=nperseg,
        scaling=kwargs.pop("scaling", "density"),
        noverlap=kwargs.pop("noverlap", 0),
    )

    da = xr.DataArray(data=psd, coords={"wavenumber": wavenumber}, name="error")
    return da.to_dataset()


def psd_welch_score(
    ds: xr.Dataset,
    variable: str,
    variable_ref: str,
    delta_x: float,
    nperseg=None,
    **kwargs,
) -> xr.Dataset:
    ds = ds.copy()

    # error
    ds = psd_welch_error(
        ds=ds,
        variable=variable,
        variable_ref=variable_ref,
        delta_x=delta_x,
        nperseg=nperseg,
        **kwargs,
    )

    # differ
    ds_ = psd_welch(
        ds=ds,
        variable=variable_ref,
        delta_x=delta_x,
        nperseg=nperseg,
        **kwargs
    )

    ds["score"] = (("wavenumber"), 1 - (ds["error"].values / ds_[variable_ref].values))

    return ds


def psd_isotropic_error(
    ds: xr.Dataset,
    ref_variable: str,
    study_variable: str,
    psd_dims: List[str],
    avg_dims: Optional[List[str]] = None,
    **kwargs,
) -> xr.Dataset:
    """Calculates the isotropic PSD error with arbitrary dimensions

    PSD_SCORE = isoPSD(x - y)

    Args:
        ds (xr.Dataset): the xarray dataset with dimensions
        ds_ref (xr.Dataset): the xarray dataset with the dimensions
        variable (str): the variable we wish to do the PSD
        psd_dims (List[str]): the dimensions for the PSD
        avg_dims (List[str]): the dimensions for the conditional average
        **kwargs: all key word args for the xrft.power_spectrum

    Returns:
        ds (xr.Dataset): the xr.Dataset with the new frequency dimensions.

    Example:
    >>> psd_spacetime_score(
        da,                 # ssh map
        da_ref,             # ssh reference dataset
        "ssh",              # variable
        ["time", "lon"],    # dimensions for isotropic power spectrum
        "lat")              # dimensions for the average
    """
    da_ref, da = ds[ref_variable], ds[study_variable]
    psd_error = psd_isotropic(
        ds=(da_ref - da).to_dataset(name='error'), variable='error', dims=psd_dims, **kwargs
    )
    if avg_dims is not None:
        psd_error = xr_cond_average(psd_error, dims=avg_dims, drop=True)
    return psd_error


def psd_spacetime_error(
    ds: xr.Dataset,
    ref_variable: str,
    study_variable: str,
    psd_dims: List[str],
    avg_dims: Optional[List[str]] = None,
    **kwargs,
) -> xr.Dataset:
    """Calculates the PSD error with arbitrary dimensions

    PSD_SCORE = PSD(x - y)

    Args:
        ds (xr.Dataset): the xarray dataset with dimensions
        ds_ref (xr.Dataset): the xarray dataset with the dimensions
        variable (str): the variable we wish to do the PSD
        psd_dims (List[str]): the dimensions for the PSD
        avg_dims (List[str]): the dimensions for the conditional average
        **kwargs: all key word args for the xrft.power_spectrum

    Returns:
        ds (xr.Dataset): the xr.Dataset with the new frequency dimensions.

    Example:
    >>> psd_spacetime_score(
        da,                 # ssh map
        da_ref,             # ssh reference dataset
        "ssh",              # variable
        ["time", "lon"],    # dimensions for power spectrum
        "lat")              # dimensions for the average
    """
    da_ref, da = ds[ref_variable], ds[study_variable]
    psd_error = psd_spacetime(
        ds=(da_ref - da).to_dataset(name='error'),
        variable='error', dims=psd_dims, **kwargs
    )
    if avg_dims is not None:
        psd_error = xr_cond_average(psd_error, dims=avg_dims, drop=True)
    return psd_error


def psd_isotropic_score(
    ds: xr.Dataset,
    ref_variable: str,
    study_variable: str,
    psd_dims: List[str],
    avg_dims: Optional[List[str]] = None,
    **kwargs,
) -> tuple[xr.Dataset, float]:
    """Calculates the isotropic PSD score with arbitrary dimensions

    PSD_SCORE = 1 - PSD(x - y) / PSD(y)

    Args:
        ds (xr.Dataset): the xarray dataset with dimensions
        ds_ref (xr.Dataset): the xarray dataset with the dimensions
        variable (str): the variable we wish to do the PSD
        psd_dims (List[str]): the dimensions for the PSD
        avg_dims (List[str]): the dimensions for the conditional average
        **kwargs: all key word args for the xrft.power_spectrum

    Returns:
        ds (xr.Dataset): the xr.Dataset with the new frequency dimensions.

    Example:
    >>> psd_spacetime_score(
        da,                 # ssh map
        da_ref,             # ssh reference dataset
        "ssh",              # variable
        ["time", "lon"],    # dimensions for isotropic power spectrum
        "lat")              # dimensions for the average
    """
    # error
    psd_err = psd_isotropic_error(
        ds=ds,
        ref_variable=ref_variable,
        study_variable=study_variable,
        psd_dims=psd_dims,
        avg_dims=avg_dims,
        **kwargs,
    )

    # reference signal
    psd_ref = psd_isotropic(ds=ds, variable=ref_variable, dims=psd_dims, **kwargs)

    if avg_dims is not None:
        psd_ref = xr_cond_average(psd_ref, dims=avg_dims)

    # normalized score
    score = (1.0 - (psd_err.error / psd_ref.ref)).to_dataset(name='score')

    space_rs = find_intercept_1D(
        y=1.0 / score.freq_r.values,
        x=score.score.values,
        level=0.5,
    )
    return score, space_rs


def psd_spacetime_score(
    ds: xr.Dataset,
    ref_variable: str,
    study_variable: str,
    psd_dims: List[str],
    avg_dims: Optional[List[str]] = None,
    **kwargs,
) -> tuple[xr.Dataset, float, float]:
    """Calculates the PSD score with arbitrary dimensions

    PSD_SCORE = 1 - PSD(x - y) / PSD(y)

    Args:
        ds (xr.Dataset): the xarray dataset with dimensions
        ds_ref (xr.Dataset): the xarray dataset with the dimensions
        variable (str): the variable we wish to do the PSD
        psd_dims (List[str]): the dimensions for the PSD
        avg_dims (List[str]): the dimensions for the conditional average
        **kwargs: all key word args for the xrft.power_spectrum

    Returns:
        ds (xr.Dataset): the xr.Dataset with the new frequency dimensions.

    Example:
    >>> psd_spacetime_score(
        da,                 # ssh map
        da_ref,             # ssh reference dataset
        "ssh",              # variable
        ["time", "lon"],    # dimensions for power spectrum
        "lat")              # dimensions for the average
    """
    # error
    psd_err = psd_spacetime_error(
        ds=ds,
        ref_variable=ref_variable,
        study_variable=study_variable,
        psd_dims=psd_dims,
        avg_dims=avg_dims,
        **kwargs,
    )

    # reference signal
    psd_ref = psd_spacetime(ds=ds, variable=ref_variable, dims=psd_dims, **kwargs)

    if avg_dims is not None:
        psd_ref = xr_cond_average(psd_ref, dims=avg_dims)

    # normalized score
    score = (1.0 - (psd_err.error / psd_ref[ref_variable])).to_dataset(name="score")

    lon_rs, time_rs = find_intercept_2D(
        x=1.0 / score.freq_lon.values,
        y=1.0 / score.freq_time.values,
        z=score.score.values,
        levels=0.5,
    )
    return score, lon_rs.item(), time_rs.item()
