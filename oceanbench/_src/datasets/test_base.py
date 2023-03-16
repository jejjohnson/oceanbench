from typing import Literal, Tuple
import pytest
from .base import XArrayDataset
import pandas as pd
import numpy as np
from oceanbench._src.utils.custom_dtypes import CoordinateAxis, Bounds
from oceanbench._src.geoprocessing.gridding import create_coord_grid
from xarray_dataclasses import Data, Name, Coord, asdataarray
from dataclasses import dataclass

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
    return np.arange(-10, 10+1, 1)

@pytest.fixture
def axis_2d(axis_1d):
    axis2 = np.arange(-20, 20+1, 1)
    return axis_1d, axis2


@pytest.fixture
def axis_3d(axis_2d):
    axis_1d, axis_2d = axis_2d
    axis3 = np.arange(-30, 30+1, 1)
    return axis_1d, axis_2d, axis3


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

# TODO: Test Domain Limits
# TODO: Test Patch Dims
# TODO: Test Strides
# TODO: Test Transformations

@pytest.mark.parametrize(
        "patch,stride,limits,datasize",
        [
    (None, None, None, 21),
    (1, None, None, 21),
    (1, 1, None, 21),
    (1, None, {"x": slice(-5,5)}, 11),
    (5, None, None, 17),
    (5, 2, None, 9),
    (5, 2, {"x": slice(-5,5)}, 4)
    ])
def test_xarraydataset_check_1d(variable_1d, patch, stride,limits, datasize):
    patches = {"x": patch} if patch is not None else None
    strides = {"x": stride} if stride is not None else None
    # create bounds object
    domain_limits = None if limits is None else limits
    check_full_scan = True
    check_dim_order = True
    transforms = None

    ds = XArrayDataset(
        da=variable_1d,
        patches=patches,
        strides=strides,
        domain_limits=domain_limits,
        check_full_scan=check_full_scan,
        check_dim_order=check_dim_order,
        transforms=transforms
    )
    assert ds.strides == {"x": 1} if strides is None else strides
    assert ds.patches == {"x": 1} if patches is None else patches
    assert ds.da_size == {"x": datasize}
    assert ds[0].shape == (patch,) if patch is not None else (1,)


@pytest.mark.parametrize(
        "patch,stride,limits,datasize",
        [
    (None, None, None, (21,41)),
    ((1,1), None, None, (21,41)),
    ((1,1), (1,1), None, (21,41)),
    ((1,1), None, {"x": slice(-5,5)}, (11,41)),
    ((1,1), None, {"y": slice(-10,10)}, (21,21)),
    ((1,1), None, {"x": slice(-5,5), "y": slice(-10,10)}, (11,21)),
    ((5,1), None, None, (17,41)),
    ((1,5), None, None, (21,37)),
    ((5,5), None, None, (17,37)),
    ((1,1), (2,1), None, (11,41)),
    ((1,1), (1,3), None, (21,14)),
    ((1,1), (3,4), None, (7,11)),
    ])
def test_xarraydataset_check_2d(variable_2d, patch, stride,limits, datasize):
    patches = {"x": patch[0], "y": patch[1]} if patch is not None else None
    strides = {"x": stride[0], "y": stride[1]} if stride is not None else None
    # create bounds object
    domain_limits = None if limits is None else limits
    check_full_scan = False
    check_dim_order = False
    transforms = None

    ds = XArrayDataset(
        da=variable_2d,
        patches=patches,
        strides=strides,
        domain_limits=domain_limits,
        check_full_scan=check_full_scan,
        check_dim_order=check_dim_order,
        transforms=transforms
    )

    assert ds.strides == {"x": 1, "y": 1} if strides is None else strides
    assert ds.patches == {"x": 1, "y": 1} if patches is None else patches
    assert ds.da_size == {"x": datasize[0], "y": datasize[1]}
    assert ds[0].shape == (patch[0],patch[1]) if patch is not None else (1,1)



# def test_xarraydataset_check_case_2d_null(variable_2d):
#     patch_dims = None
#     strides = None
#     domain_limits = None
#     check_full_scan = True
#     check_dim_order = True
#     transforms = None

#     ds = XArrayDataset(
#         da=variable_2d,
#         patches=patch_dims,
#         strides=strides,
#         domain_limits=domain_limits,
#         check_full_scan=check_full_scan,
#         check_dim_order=check_dim_order,
#         transforms=transforms
#     )
#     assert ds.patches == {"x": 1, "y": 1}
#     assert ds.strides == {} if strides is None else strides
#     assert domain_limits == None
#     assert ds[0].shape == (1,1,)


# @pytest.mark.parametrize(
#         "patch_x,patch_y,stride,xlims,datasize",
#         [
#     (1, None, None, None, 21),
#     (1, None, 1, None, 21),
#     (1, None, None, (-5,5), 11),
#     (5, None, None, None, 17),
#     (5, None, 2, None, 9),
#     (5, None, 2, (-5,5), 4)
#     ])
# def test_xarraydataset_check_2d(variable_1d, patch_x, stride,xlims, datasize):
#     patch_dims = {"x": patch_x} if patch_x is not None else None
#     strides = {"x": stride} if stride is not None else None
#     # create bounds object
#     if xlims is not None:
#         domain_limits = Bounds(val_min=xlims[0], val_max=xlims[1], name="x")
#     else:
#         domain_limits = None
#     check_full_scan = True
#     check_dim_order = True
#     transforms = None

#     ds = XArrayDataset(
#         da=variable_1d,
#         patch_dims=patch_dims,
#         strides=strides,
#         domain_limits=domain_limits,
#         check_full_scan=check_full_scan,
#         check_dim_order=check_dim_order,
#         transforms=transforms
#     )
#     assert patch_dims == ds.patch_dims
#     assert ds.strides == {} if strides is None else strides
#     assert ds.da_size == {"x": datasize}
#     assert ds[0].shape == (patch_x,)


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