from typing import Literal, Tuple
import pytest
from .base import XRDABatcher
import pandas as pd
import numpy as np
from xarray_dataclasses import Data, Name, Coord, asdataarray
from dataclasses import dataclass
from oceanbench._src.datasets.utils import list_product
from einops import repeat

X = Literal["x"]
Y = Literal["y"]
Z = Literal["z"]


@dataclass
class Variable1D:
    data: Data[tuple[X], np.float32]
    x: Coord[X, np.float32] = 0
    name: Name[str] = "var"


@dataclass
class Variable2D:
    data: Data[tuple[X, Y], np.float32]
    x: Coord[X, np.float32] = 0
    y: Coord[Y, np.float32] = 0
    name: Name[str] = "var"


@dataclass
class Variable3D:
    data: Data[tuple[X, Y, Z], np.float32]
    x: Coord[X, np.float32] = 0
    y: Coord[Y, np.float32] = 0
    z: Coord[Z, np.float32] = 0
    name: Name[str] = "var"


@pytest.fixture
def axis_1d():
    return np.arange(-10, 10, 1)

@pytest.fixture
def axis_2d(axis_1d):
    axis2 = np.arange(-20, 20, 1)
    return axis_1d, axis2


@pytest.fixture
def axis_3d(axis_2d):
    axis1, axis2 = axis_2d
    axis3 = np.arange(-30, 30, 1)
    return axis1, axis2, axis3


@pytest.fixture
def variable_1d(axis_1d):
    ones = np.ones((len(axis_1d),))
    var = Variable1D(ones, axis_1d)
    return asdataarray(var)


@pytest.fixture
def variable_2d(axis_2d):
    axis1, axis2 = axis_2d
    ones = np.ones((len(axis1),len(axis2)))
    var = Variable2D(ones, axis1, axis2)
    return asdataarray(var)

@pytest.fixture
def variable_3d(axis_3d):
    axis1, axis2, axis3 = axis_3d
    ones = np.ones((len(axis1),len(axis2), len(axis3)))
    var = Variable3D(ones, axis1, axis2, axis3)
    return asdataarray(var)


# TODO: Test Transformations

@pytest.mark.parametrize(
        "patch,stride,domain_limits,datasize",
        [
    (None, None, None, 1),
    (1, None, None, 20),
    (1, 1, None, 20),
    (1, None, {"x": slice(-5,5)}, 11),
    (5, None, None, 16),
    (4, 2, None, 9),
    (5, 2, {"x": slice(-5,5)}, 4)
    ])
def test_xrda_patcher_check_1d(variable_1d, patch, stride,domain_limits, datasize):
    patches = {"x": patch} if patch is not None else None
    strides = {"x": stride} if stride is not None else None
    check_full_scan = True

    ds = XRDABatcher(
        da=variable_1d,
        patches=patches,
        strides=strides,
        domain_limits=domain_limits,
        check_full_scan=check_full_scan,
    )

    msg = f"Patches: {ds.patches} | Strides: {ds.strides} | Dims: {ds.da_size}"
    assert ds.strides == {"x": 1} if strides is None else strides, msg
    assert ds.patches == {"x": 20} if patches is None else patches, msg
    assert ds.da_size == {"x": datasize}, msg
    assert ds[0].shape == (patch,) if patch is not None else (1,), msg
    assert len(ds) == datasize
    
    all_batches = list(map(lambda x: x, ds))
    dims_label = ["x",]

    rec_da = ds.reconstruct([all_batches], dims_label)
    np.testing.assert_array_almost_equal(rec_da.data, ds.da, decimal=5)


    all_batches = list(map(lambda x: repeat(x, "... -> ... N", N=5), all_batches)) 
    dims_label = ["x", "z"]
    rec_da = ds.reconstruct([all_batches], dims_label)
    np.testing.assert_array_almost_equal(rec_da.isel(z=0), ds.da, decimal=5)


@pytest.mark.parametrize(
        "patch,stride,domain_limits,datasize",
        [
    (None, None, None, (1,1)),
    ((1,1), None, None, (20,40)),
    ((1,1), (1,1), None, (20,40)),
    ((1,1), None, {"x": slice(-5,5)}, (11,40)),
    ((1,1), None, {"y": slice(-10,10)}, (20,21)),
    ((1,1), None, {"x": slice(-5,5), "y": slice(-10,10)}, (11,21)),
    ((5,1), None, None, (16,40)),
    ((1,5), None, None, (20,36)),
    ((5,5), None, None, (16,36)),
    ((10,20), (2,1), None, (6,21)),
    ((10,20), (1,2), None, (11,11)),
    ((10,4), (2,4), None, (6,10)),
    ])
def test_xrda_patcher_check_2d(variable_2d, patch, stride,domain_limits, datasize):
    patches = {"x": patch[0], "y": patch[1]} if patch is not None else None
    strides = {"x": stride[0], "y": stride[1]} if stride is not None else None
    check_full_scan = True

    ds = XRDABatcher(
        da=variable_2d,
        patches=patches,
        strides=strides,
        domain_limits=domain_limits,
        check_full_scan=check_full_scan,
    )

    msg = f"Patches: {ds.patches} | Strides: {ds.strides} | Dims: {ds.da_size}"
    assert ds.strides == {"x": 1, "y": 1} if strides is None else strides, msg
    assert ds.patches == {"x": 20, "y": 40} if patches is None else patches, msg
    assert ds.da_size == {"x": datasize[0], "y": datasize[1]}, msg
    assert ds[0].shape == (patch[0], patch[1]) if patch is not None else (1, 1), msg
    assert len(ds) == list_product(list(datasize))

    all_batches = list(map(lambda x: x, ds))
    dims_label = ["x", "y"]

    rec_da = ds.reconstruct([all_batches], dims_label)
    np.testing.assert_array_almost_equal(rec_da.data, ds.da, decimal=5)

    all_batches = list(map(lambda x: repeat(x, "... -> ... N", N=5), all_batches)) 
    dims_label = ["x", "y", "z"]
    rec_da = ds.reconstruct([all_batches], dims_label)
    np.testing.assert_array_almost_equal(rec_da.isel(z=0), ds.da, decimal=5)

