import pytest
import numpy as np
from .spatial import transform_180_to_360, transform_360_to_180


def test_transform_180_to_360_bounds():

    coords = np.linspace(-180, 180, 20)

    new_coords = transform_180_to_360(coords)

    assert new_coords.min() >= 0.0
    assert new_coords.max() <= 360

    new_coords = transform_180_to_360(coords)

    assert new_coords.min() >= 0.0
    assert new_coords.max() <= 360

def test_transform_360_to_180_bounds():

    coords = np.linspace(0, 360, 20)

    new_coords = transform_360_to_180(coords)

    assert new_coords.min() >=-180
    assert new_coords.max() <= 180

    new_coords = transform_360_to_180(coords)

    assert new_coords.min() >= -180
    assert new_coords.max() <= 180