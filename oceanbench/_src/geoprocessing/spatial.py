import numpy as np


def convert_lon_360_180(lon: np.ndarray) -> np.ndarray:
    """This converts the longitude coordinates from
    0:360 to -180:180
    Args:
        lon (np.ndarray): the longitude coordinates (0, 360)
    Returns:
        np.ndarray: the longitude coordinates (0, 360)
    """
    return ((lon + 180) % 360) - 180


def convert_lon_180_360(lon: np.ndarray) -> np.ndarray:
    """This converts the longitude coordinates from
    0:360 to -180:180
    Args:
        lon (np.ndarray): the longitude coordinates (0, 360)
    Returns:
        np.ndarray: the longitude coordinates (0, 360)
    """
    return lon % 360