---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: oceanbench
  language: python
  name: oceanbench
---

# Regridding Data

```{code-cell} ipython3
cd ..
```

```{code-cell} ipython3
from omegaconf import OmegaConf
import hydra
import xarray as xr

import oceanbench._src.geoprocessing.validation as geoval
```

```{code-cell} ipython3
import importlib
importlib.reload(geoval)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
raw_natl = xr.open_dataset('../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc')
```

```{code-cell} ipython3
raw_natl
```

## Preprocessing steps
- set time units "seconds since 2012-10-01"
- decode times to datetime
- add longitude and latitude units
- add sea surface height unit
- select domain

+++

**Decode time**

```{code-cell} ipython3
print('Before: ', raw_natl.time)
print()
natl = geoval.decode_cf_time(raw_natl, units="seconds since 2012-10-01")
print('After: ', natl.time)
                      
```

**Validate lat lon coordinates**

```{code-cell} ipython3
print('Before: ', natl.lon.attrs, natl.lat.attrs)
print()
natl = geoval.validate_latlon(natl)
print('After: ', natl.lon.attrs, natl.lat.attrs)
```

**Validate ssh variable**

```{code-cell} ipython3
print('Before: ', natl.ssh.attrs)
print()
natl = geoval.validate_ssh(natl)
print('After: ', natl.ssh.attrs)

```

```{code-cell} ipython3
print('Before: ', natl.dims)
print()
final_natl = natl.sel(lat=slice(32, 44), lon=slice(-66, -54), time=slice('2013-01-10', '2013-03-10'))
print('After: ', final_natl.dims)
```

```{code-cell} ipython3
final_natl
```

## Using configuration for processing

```{code-cell} ipython3
import yaml
from IPython.display import Markdown, display

def disp_config(cfg):
    display(Markdown("""```yaml\n""" +yaml.dump(OmegaConf.to_container(cfg), default_flow_style=None, indent=2)+"""\n```"""))
```

```{code-cell} ipython3

```

```{code-cell} ipython3
data_cfg = OmegaConf.load('config/data/gridded.yaml')
data = hydra.utils.call(data_cfg)
disp_config(data_cfg)
```

```{code-cell} ipython3
key = 'natl'
OmegaConf.resolve(data_cfg[key])
disp_config(data_cfg[key])
data[key]()
```

```{code-cell} ipython3
key = 'oi'
OmegaConf.resolve(data_cfg[key])
disp_config(data_cfg[key])
data[key]()
```

```{code-cell} ipython3
key = 'obs'
OmegaConf.resolve(data_cfg[key])
disp_config(data_cfg[key])
data[key]()
```

```{code-cell} ipython3
hydra.utils.call(data).natl
```

```{code-cell} ipython3
hydra.utils.call(data_cfg).oi()
```

```{code-cell} ipython3

```
