from typing import Union, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def find_intercept_1D(
    x: np.ndarray, 
    y: np.ndarray, 
    level: float=0.5
) -> np.ndarray:

    f = interp1d(x, y)

    try:
        ynew = f(level)
    except ValueError:
        msg = "The interpolated value is outside the range. "
        text = msg + f"{level}|{x.min():.2f}-{x.max():.2f}"
        warnings.warn(text)
        if level < x.min():
            ynew = f(x.min())
        else:
            ynew = f(x.max())

    return ynew


def find_intercept_2D(
    x: np.ndarray, 
    y: np.ndarray, 
    z: np.ndarray, 
    levels: Union[float, List[float]]=0.5
) -> Tuple[float, float]:
    
    x_shape = x.shape[0]
    y_shape = y.shape[0]
    z = z.reshape((y_shape, x_shape))
    
    if not isinstance(levels, list):
        levels = [levels]
        
    cs = plt.contour(x, y, z, levels=levels)
    try:
        x_level, y_level = cs.collections[0].get_paths()[0].vertices.T
    except IndexError:
        x_level, y_level = np.inf, np.inf
    plt.close()
        
    return np.min(x_level), np.min(y_level)