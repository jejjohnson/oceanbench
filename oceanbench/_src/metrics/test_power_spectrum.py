import numpy as np
from oceanbench._src.metrics.power_spectrum import psd_spacetime, psd_isotropic
from oceanbench._src.geoprocessing.test_geostrophic import get_ssh_data


def test_psd_isotropic():
    
    ds = get_ssh_data()
    
    ds_psd = psd_isotropic(ds, "ssh", ["lat", "lon"])
    dims = set(["freq_r", "time"])
    assert dims == set(ds_psd.dims.keys())
    
    ds_psd = psd_isotropic(ds, "ssh", ["time", "lon"])
    dims = set(["freq_r", "lat"])
    assert dims == set(ds_psd.dims.keys())
    
    ds_psd = psd_isotropic(ds, "ssh", ["time", "lat"])
    dims = set(["freq_r", "lon"])
    assert dims == set(ds_psd.dims.keys())
    
    
def test_psd_spacetime():
    
    ds = get_ssh_data()
    
    # CASES
    ds_psd = psd_spacetime(ds, "ssh", ["time",])
    dims = set(["freq_time", "lat", "lon"])
    assert dims == set(ds_psd.dims.keys())
    
    ds_psd = psd_spacetime(ds, "ssh", ["lon"])
    dims = set(["time", "freq_lon", "lat"])
    assert dims == set(ds_psd.dims.keys())
    
    ds_psd = psd_spacetime(ds, "ssh", ["lat", ])
    dims = set(["freq_lat", "lon", "time"])
    assert dims == set(ds_psd.dims.keys())
    
    # 2D CASES
    ds_psd = psd_spacetime(ds, "ssh", ["time", "lat"])
    dims = set(["freq_time", "freq_lat", "lon"])
    assert dims == set(ds_psd.dims.keys())
    
    ds_psd = psd_spacetime(ds, "ssh", ["time", "lon"])
    dims = set(["freq_time", "freq_lon", "lat"])
    assert dims == set(ds_psd.dims.keys())
    
    ds_psd = psd_spacetime(ds, "ssh", ["lat", "lon"])
    dims = set(["freq_lat", "freq_lon", "time"])
    assert dims == set(ds_psd.dims.keys())
    
    