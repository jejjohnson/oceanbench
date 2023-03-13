from typing import Literal, Tuple
import pytest
from .base import XArrayDataset
import pandas as pd
import numpy as np
from oceanbench._src.utils.custom_dtypes import CoordinateAxis, Bounds
from oceanbench._src.geoprocessing.gridding import create_coord_grid
from xarray_dataclasses import Data, Name, Coordof, asdataarray
from dataclasses import dataclass

X = Literal["x"]
Y = Literal["y"]
Z = Literal["z"]

@dataclass
class Variable1D:
    data: Data[X, np.ndarray]
    x: Coordof[CoordinateAxis] = 0
    name: Name[str] = "var"


@dataclass
class Variable2D:
    data: Data[Tuple[X, Y], np.ndarray]
    x: Coordof[CoordinateAxis] = 0
    y: Coordof[CoordinateAxis] = 0
    name: Name[str] = "var"


@dataclass
class Variable3D:
    data: Data[Tuple[X, Y, Z], np.ndarray]
    x: Coordof[CoordinateAxis] = 0
    y: Coordof[CoordinateAxis] = 0
    z: Coordof[CoordinateAxis] = 0
    name: Name[str] = "var"


@pytest.fixture
def bounds_1d():
    return Bounds(-5, 5, name="x")


@pytest.fixture
def bounds_2d(bounds_1d):
    return [bounds_1d, Bounds(-10, 10, name="y")]


@pytest.fixture
def bounds_3d(bounds_1d, bounds_2d):
    return [bounds_1d, bounds_2d, Bounds(-15, 15, name="z")]


@pytest.fixture
def axis_1d():
    return CoordinateAxis.init_from_limits(-10, 10, 1)


@pytest.fixture
def axis_2d(axis_1d):
    axis2 = CoordinateAxis.init_from_limits(-20, 20, 1)
    return axis_1d, axis2


@pytest.fixture
def axis_3d(axis_2d):
    axis_1d, axis_2d = axis_2d
    axis3 = CoordinateAxis.init_from_limits(-30, 30, 1)
    return axis_2d, axis_2d, axis3


@pytest.fixture
def variable_1d(axis_1d):
    ones = np.arange(0, axis_1d.ndim, 1)
    var = Variable1D(ones, axis_1d)
    return asdataarray(var)


@pytest.fixture
def variable_2d(axis_2d):
    axis1, axis2 = axis_2d
    ones = create_coord_grid(axis1.ndim, axis2.ndim)
    ones = np.ones((axis1.ndim, axis2.ndim))
    var = Variable2D(ones, axis1, axis2)
    return asdataarray(var)


def test_xarraydataset_check_case_1d_null(variable_1d):
    patch_dims = {}
    strides = {}
    domain_limits = None
    check_full_scan = True
    check_dim_order = True
    transforms = None

    ds = XArrayDataset(
        da=variable_1d,
        patch_dims=patch_dims,
        strides=strides,
        domain_limits=domain_limits,
        check_full_scan=check_full_scan,
        check_dim_order=check_dim_order,
        transforms=transforms
    )
    assert patch_dims == {}
    assert strides == {}
    assert domain_limits == None
    assert ds[0].shape == (len(variable_1d.data),)
    np.testing.assert_array_equal(ds[0], np.arange(0, len(variable_1d.x), 1))


@pytest.mark.parametrize(
        "patch,stride,xlims, batchsize",
        [(1, None,None, 21), (1, None,(-5,5), 11), (1, 1,None, 21),  ])
def test_xarraydataset_check_1d(variable_1d, patch, stride,xlims, batchsize):
    patch_dims = {"x": patch} if patch is not None else None
    strides = {"x": stride} if stride is not None else None
    # create bounds object
    if xlims is not None:
        domain_limits = Bounds(val_min=xlims[0], val_max=xlims[1], name="x")
    else:
        domain_limits = None
    check_full_scan = True
    check_dim_order = True
    transforms = None

    ds = XArrayDataset(
        da=variable_1d,
        patch_dims=patch_dims,
        strides=strides,
        domain_limits=domain_limits,
        check_full_scan=check_full_scan,
        check_dim_order=check_dim_order,
        transforms=transforms
    )
    assert patch_dims == ds.patch_dims
    assert ds.strides == {} if strides is None else strides
    assert ds.da_size == {"x": batchsize}
    assert ds[0].shape == (patch,)
    # assert ds[0].shape == (len(variable_1d.data),)
    # np.testing.assert_array_equal(ds[0], np.arange(0, len(variable_1d.x), 1))


# def test_xarraydataset_check_null_case_2d(variable_2d):
#     patch_dims = {}
#     strides = {}
#     domain_limits = None
#     check_full_scan = True
#     check_dim_order = True
#     transforms = None

#     ds = XArrayDataset(
#         da=variable_2d,
#         patch_dims=patch_dims,
#         strides=strides,
#         domain_limits=domain_limits,
#         check_full_scan=check_full_scan,
#         check_dim_order=check_dim_order,
#         transforms=transforms
#     )
#     assert patch_dims == {}
#     assert strides == {}
#     assert domain_limits == None
#     assert ds[0].shape == (len(variable_2d.data),)
#     # np.testing.assert_array_equal(ds[0], np.arange(0, len(variable_2d.x), 1))



# def test_xdataset_check_null_case_1dt():
#     pass


# def test_xdataset_check_null_case_2d():
#     pass


# def test_xdataset_check_null_case_2dt():
#     pass


# def test_xrdataset_check_full_scan_1d():
#     pass


# def test_xrdataset_check_full_scan_1dt():
#     pass


# def test_xrdataset_check_full_scan_2d():
#     pass


# def test_xrdataset_check_full_scan_2dt():
#     pass

# def test_xrdataset_check_dim_order():
#     pass


# def test_xrdataset_iter():
#     pass


# def test_xrdatset_len():
#     pass


# def test_xrdataset_get_coords():
#     pass


# def test_xrdataset_get_item():
#     pass


# def test_xrdataset_reconstruct():
#     pass


# def test_xrdataset_reconstruct_from_items():
#     pass


# def test_xrconcatdataset_reconstruct():
#     pass