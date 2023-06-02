# import autoroot
import hydra
from loguru import logger
from pathlib import Path
from omegaconf import OmegaConf

import xarray as xr
import pint_xarray
import pandas as pd

pd.set_option("display.precision", 4)
from oceanbench._src.geoprocessing.gridding import grid_to_regular_grid

METRICS_SUMMARY = []
import hvplot
import hvplot.xarray
hvplot.extension('matplotlib')

def hvp(inp, **kw):  return inp.hvplot(**kw)
def opts(plot, **kw):  return plot.opts(**kw)
def cols(plot, ncol):  return plot.cols(ncol)

def merge_ds(study_ds, ref_ds, var='ssh'):
    da = grid_to_regular_grid(
        src_grid_ds=study_ds.pint.dequantify()[var],
        tgt_grid_ds=ref_ds.pint.dequantify()[var],
        keep_attrs=True,
    )
    da["time"] = ref_ds["time"]

    return xr.Dataset(dict(
        ref = ref_ds[var],
        study = da,
    ), coords=ref_ds.coords)


def add_units(da, units={}):
    da = da.pint.quantify(units)
    return da.pint.dequantify()

@hydra.main(config_path="config", config_name="main", version_base="1.2")
def main(cfg):
    eval = hydra.utils.instantiate(cfg.evaluation)

    logger.info(f"Building and preprocessing eval dataset...")
    ds = eval.build_eval_ds().pipe(eval.preprocessing)

    logger.info(f"plotting results...")
    ds_plots = {
        plot: plot_fn(ds) for plot, plot_fn in eval.plots.items()
    }

    logger.info(f"computing metrics...")
    metrics_data = {
        metric: metric_fn(ds) for metric, metric_fn in eval.metrics.items()
    }

    logger.info(f"plotting metrics...")
    metrics_plots = {
        plot: plot_fn(metrics_dict) for plot, plot_fn in eval.metrics_plots.items()
    }

    logger.info(f"metrics summary...")
    metrics_summary = pd.Series(
        {
            name: (value(metrics_data) if callable(value) else value)
            for name, value in eval.summary.items()
        }
    )
    logger.info(metrics_summary.to_frame(name='DC2020a_OSSE').T.to_markdown())
    METRICS_SUMMARY.append(metrics_summary)

    return metrics_summary
    




if __name__ == "__main__":
    # main()
    main()
    print(pd.concat(METRICS_SUMMARY, axis=1).T.to_markdown())
