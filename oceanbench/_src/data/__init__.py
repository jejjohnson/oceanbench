from omegaconf import OmegaConf
import pint_xarray
from loguru import logger


def pipe(inp, fns):
    out = inp
    for fn in fns:
        assert callable(fn), OmegaConf.to_yaml(fn)
        out = fn(out)
    return out

