import numpy as np


def transform_360_to_180(coord: np.ndarray) -> np.ndarray:
    """This converts the coordinates that are bounded from
    [-180, 180] to coordinates bounded by [0, 360]

    Args:
        coord (np.ndarray): the input array of coordinates

    Returns:
        coord (np.ndarray): the output array of coordinates
    """
    return (coord % 360) - 180


def transform_180_to_360(coord: np.ndarray) -> np.ndarray:
    """This converts the coordinates that are bounded from
    [0, 360] to coordinates bounded by [-180, 180]

    Args:
        coord (np.ndarray): the input array of coordinates

    Returns:
        coord (np.ndarray): the output array of coordinates
    """
    return coord % 360