import numpy as np
from oceanbench._src.geoprocessing.interpolate import fillnan_gauss_seidel
from oceanbench._src.geoprocessing.test_geostrophic import get_ssh_data
import pint_xarray


def test_fillnans_border():
    ds = get_ssh_data()
    
    # # set nans on lat axis
    # ds.ssh[:, 0] = np.nan
    # assert np.sum(np.isnan(ds.ssh)) > 0
    
    # # CASE I - fill nans on wrong axes
    # ds_filled = fillnans(
    #     ds, dims=["lon"], method="slinear", fill_value="extrapolate"
    # )
    # assert np.sum(np.isnan(ds_filled.ssh.values)) > 0
    
    # # CASE II - fill nans on correct axes
    # ds_filled = fillnans(
    #     ds, dims=["lat"], method="slinear", fill_value="extrapolate"
    # )
    # assert np.sum(np.isnan(ds_filled.ssh.values)) == 0
    
    # CASE III - fill nans on all axes
    ds_filled = fillnan_gauss_seidel(
        ds.pint.dequantify(), "ssh", 
    )
    assert np.sum(np.isnan(ds_filled.ssh.values)) == 0
    