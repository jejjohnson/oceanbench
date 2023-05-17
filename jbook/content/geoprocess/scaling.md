---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python [conda env:miniconda3-jejeqx]
  language: python
  name: conda-env-miniconda3-jejeqx-py
---

# Scaling Domains

+++

> This demo notebook showcases some key functions for scaling coordinates. 
> We often need to use semantic coordinate values that are actually useful for computation. 
> So for example, lat/lon coordinates are often in the spherical domain. But for many physical problems, we need them in meters. 
> Similarly with time coordinates: we often have `np.datetime64` coordinate values but we need them in seconds.

**Note**: There are some more meaningful rescaling that involve the mean and variance (`StandardScaler`) or specific min/max values (`MinMaxScaler`) but this is an arbitrary transformation that isn't specific to coordinate values.

```{code-cell} ipython3
import autoroot
import typing as tp
from dataclasses import dataclass
import numpy as np
import pandas as pd
import xarray as xr
import einops
import xarray_dataclasses as xrdataclass
from oceanbench._src.datasets.base import XRDABatcher

%load_ext autoreload
%autoreload 2
```

```{code-cell} ipython3
file = "/gpfswork/rech/cli/uvo53rl/projects/jejeqx/data/natl60/NATL60-CJM165_GULFSTREAM_ssh_y2013.1y.nc"
!ls $file
```

```{code-cell} ipython3
da = xr.open_dataset(file, decode_times=False)
da["time"] = pd.to_datetime(da.time)
da = da.sortby("time")
```

## Spatial Rescaling

+++

#### Degrees to Meters

```{code-cell} ipython3
from oceanbench._src.geoprocessing.spatial import latlon_deg2m
```

```{code-cell} ipython3
da_scaled = latlon_deg2m(da, mean=False)
da_scaled
```

## Temporal Rescaling

+++

#### DateTime 2 Seconds

```{code-cell} ipython3
from oceanbench._src.geoprocessing.temporal import time_rescale
import pandas as pd
```

```{code-cell} ipython3
t0 = "2012-12-15"
freq_dt = 1
freq_unit = "D"


da_scale = time_rescale(da_scaled, freq_dt=freq_dt, freq_unit=freq_unit, t0=t0)

da_scale
```

```{code-cell} ipython3

```
