import pytest
from .base import XRDataset, XRConcatDataset
import pandas as pd
import numpy as np


def test_xrdataset_check_full_scan():
    n_points = 100
    day = np.arange(n_points) // 2
    time = pd.to_datetime('2020-01-01') + day * pd.to_timedelta("1D") + (np.arange(n_points) % 20) * pd.to_timedelta('1H')
    print(time.min(), time.max())
    pass

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