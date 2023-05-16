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

+++ {"user_expressions": []}

# XArrayDataset

```{code-cell} ipython3
import autoroot
import typing as tp
from dataclasses import dataclass
import numpy as np
import einops
import xarray_dataclasses as xrdataclass
from oceanbench._src.datasets.base import XRDABatcher

%load_ext autoreload
%autoreload 2
```

This tutorial walks through some of the nice features of the custom `XRDABatcher` class.
This is a custom class that slices and dices through an `xr.DataArray` where a user can specify explicitly the patch dimensions and the strides.
We preallocated the *slices* and then we can arbitrarily call the slices at will.
This is very similar to the *torch.utils.data* object except we are only working with `xr.DataArray`'s directly.


There have been other previous attempts at this, e.g. `xBatcher`.
However, we found the API very cumbersome and non-intuitive.
This is our attempt to design an API that we are comfortable with and that we find easy to use.

Below, we have outlined a few use-cases that users may be interested in. 
These use cases are:

* Chunking a 1-Dimensional Time Series
* Patch-ify a 2D Grid
* Cube-ify a 3D Volume
* Cube-ify a 2D+T Spatio-Temporal Field
* Reconstructing Multiple Variables
* Choosing Specific Dimensions for Reconstructions

We will walk through each of these and highlight how this can be achieved with the custom `XRDABatcher` class.

+++

## Case I: Chunking a 1D TS

```{code-cell} ipython3
TIME = tp.Literal["time"]

@dataclass
class TimeAxis:
    data: xrdataclass.Data[TIME, tp.Literal["datetime64[ns]"]]
    name: xrdataclass.Name[str] = "time"
    long_name: xrdataclass.Attr[str] = "Date"

@dataclass
class Variable1D:
    data: xrdataclass.Data[tuple[TIME], np.float32]
    time: xrdataclass.Coordof[TimeAxis] = 0
    name: xrdataclass.Attr[str] = "var"
```

```{code-cell} ipython3

t = np.arange(1, 360+1, 1)
rng = np.random.RandomState(seed=123)
ts = np.sin(t)

ts = Variable1D(data=ts, time=t, name="var")

da = xrdataclass.asdataarray(ts)

da
```

```{code-cell} ipython3
# da.plot()
```

In this first example, we are going to do a non-overlapping style.
We will take a 30 day window with a 30 day stride.
This will give us exactly 12 patches (like 12 months).

```{code-cell} ipython3
patches = {"time": 30}
strides = {"time": 30}
domain_limits = None#{"lat": slice(-10, 10)}
check_full_scan = True

xrda_batches = XRDABatcher(
    da=da,
    patches=patches,
    strides=strides,
    check_full_scan=check_full_scan
)

print(xrda_batches)
print(f"Dataset(size): {len(xrda_batches)}")
```



+++

In this example, we will incorporate overlapping windows.
We will do a 30 day window but we will have a 15 day stride.
So, we have a 15 day overlap when creating the patches.
We can do the mental calculation already because it's quite simple:

$$
\text{Patches} = \frac{360 \text{ days total } - 30 \text{ day patches }}{15 \text{ day stride }} + 1
$$

If this is nicely divisible, we wont have any problems. 
However, often times it's not so we might have to use the `floor` operator to ensure we get integers
Our method will give a warning (optional) which lets the user know there is an issue.

```{code-cell} ipython3
patches = {"time": 30}
strides = {"time": 15}
domain_limits = None#{"lat": slice(-10, 10)}
check_full_scan = True

xrda_batches = XRDABatcher(
    da=da,
    patches=patches,
    strides=strides,
    check_full_scan=check_full_scan
)

print(xrda_batches)
print(f"Dataset(size): {len(xrda_batches)}")
```

## Case II: Patchify a 2D Grid

```{code-cell} ipython3
TIME = tp.Literal["time"]
X = tp.Literal["x"]
Y = tp.Literal["y"]

@dataclass
class TimeAxis:
    data: xrdataclass.Data[TIME, tp.Literal["datetime64[ns]"]]
    name: xrdataclass.Name[str] = "time"
    long_name: xrdataclass.Attr[str] = "Date"

@dataclass
class XAxis:
    data: xrdataclass.Data[X, np.float32]
    name: xrdataclass.Name[str] = "x"

@dataclass
class YAxis:
    data: xrdataclass.Data[Y, np.float32]
    name: xrdataclass.Name[str] = "y"

@dataclass
class Variable2D:
    data: xrdataclass.Data[tuple[X, Y], np.float32]
    x: xrdataclass.Coordof[XAxis] = 0
    y: xrdataclass.Coordof[YAxis] = 0
    name: xrdataclass.Attr[str] = "var"
```

```{code-cell} ipython3

x = np.linspace(-1, 1, 128)
y = np.linspace(-2, 2, 128)
rng = np.random.RandomState(seed=123)

data = rng.randn(x.shape[0], y.shape[0])

grid = Variable2D(data=data, x=x, y=y, name="var")

da = xrdataclass.asdataarray(grid)


da
```

```{code-cell} ipython3
# da.T.plot.imshow()
```

We will have a `[20,20]` patch with no overlap, `[20,20]`

```{code-cell} ipython3
patches = {"x": 8, "y": 8}
strides = {"x": 8, "y": 8}
domain_limits = None#{"lat": slice(-10, 10)}
check_full_scan = True

xrda_batches = XRDABatcher(
    da=da,
    patches=patches,
    strides=strides,
    check_full_scan=check_full_scan
)

print(xrda_batches)
print(f"Dataset(size): {len(xrda_batches)}")
```

We will have a `[20,20]` patch with some overlap, like the boundaries of 2, `[2,2]`

```{code-cell} ipython3
patches = {"x": 8, "y": 8}
strides = {"x": 2, "y": 2}
domain_limits = None#{"lat": slice(-10, 10)}
check_full_scan = True

xrda_batches = XRDABatcher(
    da=da,
    patches=patches,
    strides=strides,
    check_full_scan=check_full_scan
)

print(xrda_batches)
print(f"Dataset(size): {len(xrda_batches)}")
```

## Case III: Cube-ify a 3D Volume

```{code-cell} ipython3
TIME = tp.Literal["time"]
X = tp.Literal["x"]
Y = tp.Literal["y"]
Z = tp.Literal["z"]
@dataclass
class TimeAxis:
    data: xrdataclass.Data[TIME, tp.Literal["datetime64[ns]"]]
    name: xrdataclass.Name[str] = "time"
    long_name: xrdataclass.Attr[str] = "Date"

@dataclass
class XAxis:
    data: xrdataclass.Data[X, np.float32]
    name: xrdataclass.Name[str] = "x"

@dataclass
class YAxis:
    data: xrdataclass.Data[Y, np.float32]
    name: xrdataclass.Name[str] = "y"

@dataclass
class ZAxis:
    data: xrdataclass.Data[Z, np.float32]
    name: xrdataclass.Name[str] = "z"

@dataclass
class Variable3D:
    data: xrdataclass.Data[tuple[X, Y, Z], np.float32]
    x: xrdataclass.Coordof[XAxis] = 0
    y: xrdataclass.Coordof[YAxis] = 0
    z: xrdataclass.Coordof[ZAxis] = 0
    name: xrdataclass.Attr[str] = "var"
```

```{code-cell} ipython3

x = np.linspace(-1, 1, 128)
y = np.linspace(-2, 2, 128)
z = np.linspace(-5, 5, 128)
rng = np.random.RandomState(seed=123)

data = rng.randn(x.shape[0], y.shape[0], z.shape[0])

grid = Variable3D(data=data, x=x, y=y, z=z, name="var")

da = xrdataclass.asdataarray(grid)

da
```

We will have a `[20,20]` patch with no overlap, `[20,20]`

```{code-cell} ipython3
patches = {"x": 8, "y": 8, "z": 8}
strides = {"x": 8, "y": 8, "z": 8}
domain_limits = None#{"lat": slice(-10, 10)}
check_full_scan = True

xrda_batches = XRDABatcher(
    da=da,
    patches=patches,
    strides=strides,
    check_full_scan=check_full_scan
)

print(xrda_batches)
print(f"Dataset(size): {len(xrda_batches)}")
```

We will have a `[20,20]` patch with some overlap, like the boundaries of 2, `[2,2]`

```{code-cell} ipython3
patches = {"x": 8, "y": 8, "z": 8}
strides = {"x": 2, "y": 2, "z": 2}
domain_limits = None#{"lat": slice(-10, 10)}
check_full_scan = True

xrda_batches = XRDABatcher(
    da=da,
    patches=patches,
    strides=strides,
    check_full_scan=check_full_scan
)

print(xrda_batches)
print(f"Dataset(size): {len(xrda_batches)}")
```

## Case IV: Cube-ify a 2D+T Spatio-Temporal Field

```{code-cell} ipython3
TIME = tp.Literal["time"]
X = tp.Literal["x"]
Y = tp.Literal["y"]
Z = tp.Literal["z"]

@dataclass
class TimeAxis:
    data: xrdataclass.Data[TIME, tp.Literal["datetime64[ns]"]]
    name: xrdataclass.Name[str] = "time"
    long_name: xrdataclass.Attr[str] = "Date"

@dataclass
class XAxis:
    data: xrdataclass.Data[X, np.float32]
    name: xrdataclass.Name[str] = "x"

@dataclass
class YAxis:
    data: xrdataclass.Data[Y, np.float32]
    name: xrdataclass.Name[str] = "y"

@dataclass
class ZAxis:
    data: xrdataclass.Data[Z, np.float32]
    name: xrdataclass.Name[str] = "z"

@dataclass
class Variable2DT:
    data: xrdataclass.Data[tuple[TIME, X, Y], np.float32]
    x: xrdataclass.Coordof[XAxis] = 0
    y: xrdataclass.Coordof[YAxis] = 0
    time: xrdataclass.Coordof[TimeAxis] = 0
    name: xrdataclass.Attr[str] = "var"
```

```{code-cell} ipython3

x = np.linspace(-1, 1, 200)
y = np.linspace(-2, 2, 200)
t = np.arange(1, 360+1, 1)
rng = np.random.RandomState(seed=123)

data = rng.randn(t.shape[0], x.shape[0], y.shape[0])

grid = Variable2DT(data=data, x=x, y=y, time=t, name="var")

da = xrdataclass.asdataarray(grid)

da
```

Now, this is a rather big field.
Let's say we want to use some ML method with a CNN to learn how to predict ...
However, ingesting this large patch would be very difficult.
So instead, we will use the standard size for many CNNs, which is `[64,64]`.
In addition, we will use a temporal window of 15 days. 
So the patch will be `[15,64,64]`. 

As will the above examples, we will also account for the overlap in the spatial borders with `[4,4]` strides.
And lastly, we will have a 5 day overlap for the time steps.
So the final strides will be `[5,4,4]`

```{code-cell} ipython3
patches = {"x": 64, "y": 64, "time": 15}
strides = {"x": 4, "y": 4, "time": 5}
domain_limits = None#{"lat": slice(-10, 10)}
check_full_scan = True

xrda_batches = XRDABatcher(
    da=da,
    patches=patches,
    strides=strides,
    check_full_scan=check_full_scan
)

print(xrda_batches)
print(f"Dataset(size): {len(xrda_batches)}")
```

All of the sudden, we have a LOT of data if we do things in a patch-wise manner, more than 85K samples!
However, we know from statistics that perhaps this isn't the greatest idea because there are a lot of overlap.
So we can be clever and use a training dataset with less overlap. 
However, we can create a different dataset for predictions where we reduce the strides considerably so that we take a weighted average over the predictions!

```{code-cell} ipython3
patches = {"x": 64, "y": 64, "time": 15}
strides = {"x": 1, "y": 1, "time": 1}
domain_limits = None#{"lat": slice(-10, 10)}
check_full_scan = True

xrda_batches = XRDABatcher(
    da=da,
    patches=patches,
    strides=strides,
    check_full_scan=check_full_scan
)

print(xrda_batches)
print(f"Dataset(size): {len(xrda_batches)}")
```

So this will cover use because we can take a weighted average of all of the predictions!

+++

## Case V: Reconstructing with multiple variables

+++

In this example, we look at how we can do reconstructions with multiple variables.
This may occur when we have used different methods to make predictions and we want to reconstruct all of them.

Another example is when we have some sort of latent variable representation and we would like to reconstruct each of the latent variable representations.

```{code-cell} ipython3

t = np.arange(1, 360+1, 1)
rng = np.random.RandomState(seed=123)
ts = np.sin(t)

ts = Variable1D(data=ts, time=t, name="var")

da = xrdataclass.asdataarray(ts)
```

```{code-cell} ipython3
patches = {"time": 30}
strides = {"time": 30}
domain_limits = None#{"lat": slice(-10, 10)}
check_full_scan = True

xrda_batches = XRDABatcher(
    da=da,
    patches=patches,
    strides=strides,
    check_full_scan=check_full_scan
)

print(xrda_batches)
print(f"Dataset(size): {len(xrda_batches)}")
```

```{code-cell} ipython3
all_batches = list(map(lambda x: x.data, xrda_batches))
all_batches_latent = list(map(lambda x: einops.repeat(x, "... -> ... N", N=5), all_batches)) 
```

```{code-cell} ipython3
dims_labels = ["time", "z"]
weight = np.ones((patches["time"]))
rec_da = xrda_batches.reconstruct(all_batches_latent, dims_labels=dims_labels, weight=weight)
rec_da
```

## Case VI: Choosing a Specific Dimension for Reconstruction

```{code-cell} ipython3

x = np.linspace(-1, 1, 50)
y = np.linspace(-2, 2, 50)
t = np.arange(1, 30+1, 1)
rng = np.random.RandomState(seed=123)

data = rng.randn(t.shape[0], x.shape[0], y.shape[0])

grid = Variable2DT(data=data, x=x, y=y, time=t, name="var")

da = xrdataclass.asdataarray(grid)

da
```

Now, this is a rather big field.
Let's say we want to use some ML method with a CNN to learn how to predict ...
However, ingesting this large patch would be very difficult.
So instead, we will use the standard size for many CNNs, which is `[64,64]`.
In addition, we will use a temporal window of 15 days. 
So the patch will be `[15,64,64]`. 

As will the above examples, we will also account for the overlap in the spatial borders with `[4,4]` strides.
And lastly, we will have a 5 day overlap for the time steps.
So the final strides will be `[5,4,4]`

```{code-cell} ipython3
patches = {"x": 10, "y": 10, "time": 5}
strides = {"x": 8, "y": 8, "time": 1}
domain_limits = None#{"lat": slice(-10, 10)}
check_full_scan = True

xrda_batches = XRDABatcher(
    da=da,
    patches=patches,
    strides=strides,
    check_full_scan=check_full_scan
)

print(xrda_batches)
print(f"Dataset(size): {len(xrda_batches)}")
```

Here, we can reconstruct just the time series. 
So in this case, we will take the mean of all of the spatial values x,y and just have the time series. 

```{code-cell} ipython3
all_batches = list(map(lambda x: x.mean(dim=["x", "y"]).data, xrda_batches))
```

But we still want to reconstruct! 
So we can pass these through the reconstruction but pay careful attention to the dimension we wish to reconstruct.

```{code-cell} ipython3
dims_labels = ["time"]
weight = np.ones((patches["time"]))
rec_da = xrda_batches.reconstruct(all_batches, dims_labels=dims_labels, weight=weight)
rec_da
```

Here, we can reconstruct just the x,y patches. 
So in this case, we will take the mean of all of the temporal coordinates and just have the spatial patches. 

```{code-cell} ipython3
all_batches = list(map(lambda x: x.mean(dim=["time"]).data, xrda_batches))
```

Again, we still want to reconstruct! 
So, like above, we can simply pass the correct dimensions to the reconstruction.

```{code-cell} ipython3
dims_labels = ["x", "y"]
weight = np.ones((patches["x"], patches["y"]))
rec_da = xrda_batches.reconstruct(all_batches, dims_labels=dims_labels, weight=weight)
rec_da
```

```{code-cell} ipython3

```
