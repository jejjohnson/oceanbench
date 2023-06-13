from typing import Literal
import pytest
from .base import XRDAPatcher
import numpy as np
from xarray_dataclasses import Data, Name, Coord, asdataarray
from dataclasses import dataclass
from einops import repeat


RNG = np.random.RandomState(seed=123)

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
    ones = np.ones((len(axis1), len(axis2)))
    var = Variable2D(ones, axis1, axis2)
    return asdataarray(var)


@pytest.fixture
def variable_3d(axis_3d):
    axis1, axis2, axis3 = axis_3d
    ones = np.ones((len(axis1), len(axis2), len(axis3)))
    var = Variable3D(ones, axis1, axis2, axis3)
    return asdataarray(var)


@pytest.mark.parametrize(
    "patch,stride,domain_limits,datasize",
    [
        (None, None, None, 1),
        (1, None, None, 20),
        (1, 1, None, 20),
        (1, None, {"x": slice(-5, 5)}, 11),
        (5, None, None, 16),
        (4, 2, None, 9),
        (5, 2, {"x": slice(-5, 5)}, 4),
    ],
)
def test_xrda_patcher_1d(variable_1d, patch, stride, domain_limits, datasize):
    patches = {"x": patch} if patch is not None else None
    strides = {"x": stride} if stride is not None else None
    check_full_scan = True

    ds = XRDAPatcher(
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


@pytest.mark.parametrize(
    "patch, stride",
    [
        (None, None),
        (30, None),
        (None, 5),
        (30, 5),
        (60, 10),
        (60, 60),
    ],
)
def test_xrda_patcher_1d_reconstruct(patch, stride):
    # initialize coordinates, data

    coordinate = np.arange(1, 360 + 1, 1)
    data = RNG.randn(*coordinate.shape)

    da = Variable1D(data=data, x=coordinate, name="ssh")
    da = asdataarray(da)

    patches = {"x": patch} if patch is not None else None
    strides = {"x": stride} if stride is not None else None
    check_full_scan = True

    xrda_batcher = XRDAPatcher(
        da=da, patches=patches, strides=strides, check_full_scan=check_full_scan
    )

    # collect all items
    all_items = list(map(lambda x: x.data, xrda_batcher))

    # =================================
    # CASE I - No Weight | No Label
    # =================================
    dims_labels = None
    weight = None
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    np.testing.assert_array_almost_equal(rec_da.data, xrda_batcher.da)
    assert list(rec_da.coords.keys()) == ["x"]
    assert rec_da.dims == tuple("x")

    # ====================================
    # CASE II - No Weight | Exact Label
    # ====================================
    dims_labels = ["x"]
    weight = None
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    np.testing.assert_array_almost_equal(rec_da.data, xrda_batcher.da)
    assert list(rec_da.coords.keys()) == ["x"]
    assert rec_da.dims == tuple("x")

    # ====================================
    # CASE III - Weight | No Label
    # ====================================
    dims_labels = None
    weight = np.ones((xrda_batcher.patches["x"],))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    np.testing.assert_array_almost_equal(rec_da.data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == set(["x"])
    assert rec_da.dims == tuple("x")

    # ====================================
    # CASE IV - Weight | Exact Label
    # ====================================
    dims_labels = ["x"]
    weight = np.ones((xrda_batcher.patches["x"],))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    np.testing.assert_array_almost_equal(rec_da.data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == set(["x"])
    assert rec_da.dims == tuple("x")


@pytest.mark.parametrize(
    "patch, stride",
    [
        (None, None),
        (30, None),
        (None, 5),
        (30, 5),
        (60, 10),
        (60, 60),
    ],
)
def test_xrda_patcher_1d_reconstruct_latent(patch, stride):
    # initialize coordinates, data

    coordinate = np.arange(1, 360 + 1, 1)
    data = RNG.randn(*coordinate.shape)

    da = Variable1D(data=data, x=coordinate, name="ssh")
    da = asdataarray(da)

    patches = {"x": patch} if patch is not None else None
    strides = {"x": stride} if stride is not None else None
    check_full_scan = True

    xrda_batcher = XRDAPatcher(
        da=da, patches=patches, strides=strides, check_full_scan=check_full_scan
    )

    ######################################
    # LATENT SPACES
    ######################################
    # aggregate all items
    all_items = list(map(lambda x: x.data, xrda_batcher))
    all_items = list(map(lambda x: repeat(x, "... -> ... N", N=5), all_items))

    # =================================
    # CASE I - No Weight | No Label
    # =================================
    dims_labels = None
    weight = None
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    np.testing.assert_array_almost_equal(rec_da.isel(v1=0).data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == set(["x"])
    assert set(rec_da.dims) == set(tuple(["x", "v1"]))

    # ====================================
    # CASE II - No Weight | Exact Label
    # ====================================
    dims_labels = ["x", "z"]
    weight = None
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    np.testing.assert_array_almost_equal(rec_da.isel(z=0).data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == set(["x"])
    assert set(rec_da.dims) == set(tuple(["x", "z"]))

    # ====================================
    # CASE III - Weight | No Label
    # ====================================
    dims_labels = None
    weight = np.ones((xrda_batcher.patches["x"],))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    np.testing.assert_array_almost_equal(rec_da.isel(v1=0).data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == set(["x"])
    assert set(rec_da.dims) == set(tuple(["x", "v1"]))

    # ====================================
    # CASE IV - Weight | Exact Label
    # ====================================
    dims_labels = ["x", "z"]
    weight = np.ones((xrda_batcher.patches["x"],))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    np.testing.assert_array_almost_equal(rec_da.isel(z=0).data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == set(["x"])
    assert set(rec_da.dims) == set(tuple(["x", "z"]))


@pytest.mark.parametrize(
    "patch,stride,domain_limits,datasize",
    [
        (None, None, None, (1, 1)),
        ((1, 1), None, None, (20, 40)),
        ((1, 1), (1, 1), None, (20, 40)),
        ((1, 1), None, {"x": slice(-5, 5)}, (11, 40)),
        ((1, 1), None, {"y": slice(-10, 10)}, (20, 21)),
        ((1, 1), None, {"x": slice(-5, 5), "y": slice(-10, 10)}, (11, 21)),
        ((5, 1), None, None, (16, 40)),
        ((1, 5), None, None, (20, 36)),
        ((5, 5), None, None, (16, 36)),
        ((10, 20), (2, 1), None, (6, 21)),
        ((10, 20), (1, 2), None, (11, 11)),
        ((10, 4), (2, 4), None, (6, 10)),
    ],
)
def test_xrda_patcher_2d(variable_2d, patch, stride, domain_limits, datasize):
    patches = {"x": patch[0], "y": patch[1]} if patch is not None else None
    strides = {"x": stride[0], "y": stride[1]} if stride is not None else None
    check_full_scan = True

    ds = XRDAPatcher(
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
    assert len(ds) == np.prod(list(datasize))


@pytest.mark.parametrize(
    "patch, stride",
    [
        (None, None),
        ((10, 10), None),
        (None, (10, 10)),
        ((10, 10), (10, 10)),
        ((10, 10), (6, 6)),
        ((8, 8), (4, 4)),
    ],
)
def test_xrda_patcher_2d_reconstruct(patch, stride):
    # initialize coordinates, data

    lon_axis = np.arange(10, 30, 0.5)
    lat_axis = np.arange(-80, -60, 0.5)
    data = RNG.randn(lat_axis.shape[0], lon_axis.shape[0])

    da = Variable2D(data=data, x=lon_axis, y=lat_axis, name="ssh")
    da = asdataarray(da)

    patches = {"x": patch[0], "y": patch[1]} if patch is not None else None
    strides = {"x": stride[0], "y": stride[1]} if stride is not None else None
    check_full_scan = True

    xrda_batcher = XRDAPatcher(
        da=da, patches=patches, strides=strides, check_full_scan=check_full_scan
    )

    # aggregate all items
    all_items = list(map(lambda x: x.data, xrda_batcher))

    ###################################
    # EXACT LABELS
    ###################################

    # =================================
    # CASE I - No Weight | No Label
    # =================================
    dims_labels = None
    weight = None
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    np.testing.assert_array_almost_equal(rec_da.data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == set(["x", "y"])
    assert rec_da.dims == tuple(["x", "y"])

    # ====================================
    # CASE II - No Weight | Exact Label
    # ====================================
    dims_labels = ["x", "y"]
    weight = None
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    np.testing.assert_array_almost_equal(rec_da.data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == set(["x", "y"])
    assert rec_da.dims == tuple(["x", "y"])

    # ====================================
    # CASE III - Weight | No Label
    # ====================================
    dims_labels = None
    weight = np.ones((xrda_batcher.patches["x"], xrda_batcher.patches["y"]))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    np.testing.assert_array_almost_equal(rec_da.data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == set(["x", "y"])
    assert rec_da.dims == tuple(["x", "y"])

    # ====================================
    # CASE IV - Weight | Exact Label
    # ====================================
    dims_labels = ["x", "y"]
    weight = np.ones((xrda_batcher.patches["x"], xrda_batcher.patches["y"]))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    np.testing.assert_array_almost_equal(rec_da.data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == set(["x", "y"])
    assert rec_da.dims == tuple(["x", "y"])

    ###################################
    # MIXED LABELS
    ###################################

    # ====================================
    # CASE V - Weight | Mixed Label I
    # ====================================
    dims_labels = ["x"]
    weight = np.ones((xrda_batcher.patches["x"],))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    assert list(rec_da.coords.keys()) == ["x"]
    assert rec_da.dims == tuple(["x", "v1"])

    # ====================================
    # CASE VI - Weight | Mixed Label II
    # ====================================
    dims_labels = ["y"]
    weight = np.ones((xrda_batcher.patches["y"],))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    assert list(rec_da.coords.keys()) == ["y"]
    assert rec_da.dims == tuple(["y", "v1"])

    # ====================================
    # CASE V - No Weight | Mixed Label I
    # ====================================
    dims_labels = ["x"]
    weight = None
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    assert list(rec_da.coords.keys()) == ["x"]
    assert rec_da.dims == tuple(["x", "v1"])

    # ====================================
    # CASE VI - No Weight | Mixed Label II
    # ====================================
    dims_labels = ["y"]
    weight = None
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    assert list(rec_da.coords.keys()) == ["y"]
    assert rec_da.dims == tuple(["y", "v1"])


@pytest.mark.parametrize(
    "patch, stride",
    [
        (None, None),
        ((10, 10), None),
        (None, (10, 10)),
        ((10, 10), (10, 10)),
        ((10, 10), (6, 6)),
        ((8, 8), (4, 4)),
    ],
)
def test_xrda_patcher_2d_reconstruct_latent(patch, stride):
    # initialize coordinates, data

    lon_axis = np.arange(10, 30, 0.5)
    lat_axis = np.arange(-80, -60, 0.5)
    data = RNG.randn(lat_axis.shape[0], lon_axis.shape[0])

    da = Variable2D(data=data, x=lon_axis, y=lat_axis, name="ssh")
    da = asdataarray(da)

    patches = {"x": patch[0], "y": patch[1]} if patch is not None else None
    strides = {"x": stride[0], "y": stride[1]} if stride is not None else None
    check_full_scan = True

    xrda_batcher = XRDAPatcher(
        da=da, patches=patches, strides=strides, check_full_scan=check_full_scan
    )

    # aggregate all items
    all_items = list(map(lambda x: x.data, xrda_batcher))
    all_items = list(map(lambda x: repeat(x, "... -> ... N", N=5), all_items))

    ###################################
    # EXACT LABELS
    ###################################

    # =================================
    # CASE 1 - No Weight | No Label
    # =================================
    dims_labels = None
    weight = None
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    np.testing.assert_array_almost_equal(rec_da.isel(v1=0).data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == set(["x", "y"])
    assert rec_da.dims == tuple(["x", "y", "v1"])

    # ====================================
    # CASE 2 - No Weight | Exact Label
    # ====================================
    dims_labels = ["x", "y", "z"]
    weight = None
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    np.testing.assert_array_almost_equal(rec_da.isel(z=0).data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == set(["x", "y"])
    assert rec_da.dims == tuple(["x", "y", "z"])

    # ====================================
    # CASE 3 - Weight | No Label
    # ====================================
    dims_labels = None
    weight = np.ones((xrda_batcher.patches["x"], xrda_batcher.patches["y"]))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    np.testing.assert_array_almost_equal(rec_da.isel(v1=0).data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == set(["x", "y"])
    assert rec_da.dims == tuple(["x", "y", "v1"])

    # ====================================
    # CASE 4 - Weight | Exact Label
    # ====================================
    dims_labels = ["x", "y", "z"]
    weight = np.ones((xrda_batcher.patches["x"], xrda_batcher.patches["y"]))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    np.testing.assert_array_almost_equal(rec_da.isel(z=0).data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == set(["x", "y"])
    assert rec_da.dims == tuple(["x", "y", "z"])

    ###################################
    # MIXED LABELS
    ###################################

    # ====================================
    # CASE 1 - Weight | Mixed Label 1
    # ====================================
    dims_labels = ["x"]
    weight = np.ones((xrda_batcher.patches["x"],))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    assert list(rec_da.coords.keys()) == ["x"]
    assert rec_da.dims == tuple(["x", "v1", "v2"])

    # ====================================
    # CASE 1 - Weight | Mixed Label 2
    # ====================================
    dims_labels = ["y"]
    weight = np.ones((xrda_batcher.patches["y"],))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    assert list(rec_da.coords.keys()) == ["y"]
    assert rec_da.dims == tuple(["y", "v1", "v2"])

    # ====================================
    # CASE 1 - Weight | Mixed Label 3
    # ====================================
    dims_labels = ["x", "z"]
    weight = np.ones((xrda_batcher.patches["x"],))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    assert list(rec_da.coords.keys()) == ["x"]
    assert rec_da.dims == tuple(["x", "z", "v1"])

    # ====================================
    # CASE 1 - Weight | Mixed Label 4
    # ====================================
    dims_labels = ["y", "z"]
    weight = np.ones((xrda_batcher.patches["y"],))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    assert list(rec_da.coords.keys()) == ["y"]
    assert rec_da.dims == tuple(["y", "z", "v1"])

    # ====================================
    # CASE 1 - Weight | Mixed Label 5
    # ====================================
    dims_labels = ["z", "x"]
    weight = np.ones((xrda_batcher.patches["x"],))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    assert list(rec_da.coords.keys()) == ["x"]
    assert rec_da.dims == tuple(["z", "x", "v1"])

    # ====================================
    # CASE 1 - Weight | Mixed Label 6
    # ====================================
    dims_labels = ["z", "y"]
    weight = np.ones((xrda_batcher.patches["y"],))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    assert list(rec_da.coords.keys()) == ["y"]
    assert rec_da.dims == tuple(["z", "y", "v1"])

    # ====================================
    # CASE 2 - No Weight | Mixed Label 1
    # ====================================
    dims_labels = [
        "x",
    ]
    weight = None
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    assert list(rec_da.coords.keys()) == ["x"]
    assert rec_da.dims == tuple(["x", "v1", "v2"])

    # ====================================
    # CASE 2 - No Weight | Mixed Label 2
    # ====================================
    dims_labels = ["y"]
    weight = None
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    assert list(rec_da.coords.keys()) == ["y"]
    assert rec_da.dims == tuple(["y", "v1", "v2"])

    # ====================================
    # CASE 2 - No Weight | Mixed Label 3
    # ====================================
    dims_labels = ["x", "z"]
    weight = None
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    assert list(rec_da.coords.keys()) == ["x"]
    assert rec_da.dims == tuple(["x", "z", "v1"])

    # ====================================
    # CASE 2 - No Weight | Mixed Label 4
    # ====================================
    dims_labels = ["y", "z"]
    weight = None
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    assert list(rec_da.coords.keys()) == ["y"]
    assert rec_da.dims == tuple(["y", "z", "v1"])

    # ====================================
    # CASE 2 - No Weight | Mixed Label 5
    # ====================================
    dims_labels = ["z", "x"]
    weight = None
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    assert list(rec_da.coords.keys()) == ["x"]
    assert rec_da.dims == tuple(["z", "x", "v1"])

    # ====================================
    # CASE 2 - No Weight | Mixed Label 6
    # ====================================
    dims_labels = ["z", "y"]
    weight = None
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    assert list(rec_da.coords.keys()) == ["y"]
    assert rec_da.dims == tuple(["z", "y", "v1"])

    ###################################
    # RESHAPED LABELS
    ###################################

    all_items = list(map(lambda x: repeat(x, "x y z -> y z x"), all_items))
    dims_labels = ["y", "z", "x"]
    weight = np.ones((xrda_batcher.patches["y"], xrda_batcher.patches["x"]))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=dims_labels, weight=weight)
    assert set(rec_da.coords.keys()) == set(["x", "y"])
    assert rec_da.dims == tuple(["y", "z", "x"])
