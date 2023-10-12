from operator import itemgetter as igetter, methodcaller as caller
import numpy as np
import pandas as pd
import hydra_zen
from pandas._libs.lib import item_from_zerodim
import toolz
import xarray as xr
import ocn_tools._src.geoprocessing.validation as ocnval
import ocn_tools._src.geoprocessing.gridding as ocngri
import ocn_tools._src.geoprocessing.interpolate as ocnint
import ocn_tools._src.geoprocessing.spatial as ocnspa
import ocn_tools._src.geoprocessing.geostrophic as ocngeo
import ocn_tools._src.utils.data as ocnuda
import ocn_tools._src.data as ocndat
import ocn_tools._src.geoprocessing.temporal as ocntem
import ocn_tools._src.metrics.stats as ocnmst
import ocn_tools._src.metrics.power_spectrum as ocnmps
import operator
import oceanbench._src.utils.hydra
from dataclasses import asdict, dataclass
import dataclasses
import matplotlib.ticker
###  Reasoning: Better composability and updatability:
"""
all pipelines are described with dict[str, callable]
The idea is that the keys should be sortable with available extension before and after each key
this way we can:
    change the parameter of a step by accessing key.parameter
    adding intermediary steps at arbitrary positions
    delete a recipe step by changing the key to identity

the piping function takes the resulting dict and sort the functions depending on the key
"""





###
# Hydra utils
pb = hydra_zen.make_custom_builds_fn(zen_partial=True)
b = hydra_zen.make_custom_builds_fn(zen_partial=False)
pbc = hydra_zen.make_custom_builds_fn(hydra_convert='all', zen_partial=True)
bc = hydra_zen.make_custom_builds_fn(hydra_convert='all')

def zen_compose(d):
    return toolz.compose_left(*(d[k] for k in sorted(d)))

def from_recipe(rec):
    return pb(toolz.apply, b(zen_compose, asdict(rec)))

def from_recipe_and_inp(rec, inp):
    return pb(toolz.apply, b(zen_compose, asdict(rec)), inp)

def packed_caller(name, args=[], kwargs={}):
    return caller(name, *args, **kwargs)

def join_apply(d, dfunc):
    return {k: dfunc.get(k, toolz.identity)(v) for k, v in d.items()}

def cmocmap(name):
    import cmocean.cm
    import holoviews
    return holoviews.plotting.util.process_cmap(getattr(cmocean.cm, name))


p_yaml = lambda cfg: print(hydra_zen.to_yaml(cfg, sort_keys=True))
I = hydra_zen.instantiate
## Cfgs

## Recipes
### Grid Prepro
base_prepro = hydra_zen.make_config(
    _0=pb(toolz.identity),
    _1=pb(ocnval.validate_latlon),
    _2=pb(ocnval.validate_time),
    _3=pb(ocnval.validate_ssh),
    _4=b(igetter, 'ssh'),
)
# p_yaml(base_prepro)
    
grid_ssh_prepro = hydra_zen.make_config(
    _01=pb(ocnval.decode_cf_time, units='seconds since 2012-10-01 00:00:00'),
    bases=(base_prepro,),
)
# p_yaml(grid_ssh_prepro)


grid_sst_prepro = hydra_zen.make_config(
    _01=pb(ocnval.decode_cf_time, units='seconds since 2012-10-01 00:00:00'),
    _3=pb(ocndat.add_units, units=dict(sst='celsius')),
    _4=b(igetter, 'sst'),
    bases=(base_prepro,),
)

grid_obs_prepro = hydra_zen.make_config(
    _01=b(caller, 'rename', ssh_mod='ssh'),
    _02=pb(ocnval.decode_cf_time, units='days since 2012-10-01'),
    bases=(base_prepro,),
)
# p_yaml(grid_obs_prepro)


### Results Postpro

base_osse_postpro = hydra_zen.make_config(
    _5=b(
        caller, 'sel',
        lat=b(slice, '${task.domain.lat.0}', '${task.domain.lat.1}'),
        lon=b(slice, '${task.domain.lon.0}', '${task.domain.lon.1}',), 
        # time=b(slice, '${task.splits.test.0}', '${task.splits.test.1}'),
    ),
    _61=b(caller,'resample', time='1D'),
    _62=b(caller,'mean'),
    _7=b(
        caller, 'sel',
        time=b(slice, '${task.splits.test.0}', '${task.splits.test.1}'),
    ),
    _71=b(caller,'to_dataset', name='ssh'),
    _72=pb(ocngri.grid_to_regular_grid, tgt_grid_ds='${task.eval_grid}'),
    _8=b(igetter, 'ssh'),
)


# p_yaml(base_osse_postpro)

results_prepostpro=hydra_zen.make_config(
    _01=b(toolz.identity),
    bases=(base_prepro, base_osse_postpro)
)
# p_yaml(results_prepostpro)

ref_postpro=hydra_zen.make_config(
    _499=b(caller, '__call__'),
    bases=(base_osse_postpro,)
)
# p_yaml(ref_postpro)

data_natl60 = hydra_zen.make_config(
    ssh = from_recipe_and_inp(grid_ssh_prepro(),
        b(xr.open_dataset, '../oceanbench-data-registry/osse_natl60/grid/natl_mod_ssh_daily.nc')
    ),
    sst = from_recipe_and_inp(grid_sst_prepro(),
        b(xr.open_dataset, '../oceanbench-data-registry/osse_natl60/grid/natl_mod_sst.nc')
    ),
    nadir_gridded = from_recipe_and_inp(grid_obs_prepro(),
        b(xr.open_dataset, '../oceanbench-data-registry/osse_natl60/grid/natl_obs_nadir.nc')
    ),
    swot_gridded = from_recipe_and_inp(grid_obs_prepro(),
        b(xr.open_dataset, '../oceanbench-data-registry/osse_natl60/grid/natl_obs_nadirswot.nc')
    )
)
# p_yaml(data_natl60)

### EvalDs
## Tasks:
osse_nadir = hydra_zen.make_config(
    name='DC2020 OSSE Gulfstream Nadir',
    domain=dict(lat=[33, 43], lon=[-65, -55]),
    data=dict(obs='${data.nadir_gridded}', ssh='${data.ssh}'),
    splits=dict(
        test=['2012-10-22', '2012-12-02'],
        trainval=['2013-01-01', '2013-09-30'],
    ),
    eval_grid=b(
        xr.Dataset,
        coords=dict(
            lat=b(np.arange, '${task.domain.lat.0}', '${task.domain.lat.1}', 0.05),
            lon=b(np.arange, '${task.domain.lon.0}', '${task.domain.lon.1}', 0.05),
            time=b(pd.date_range, '${task.splits.test.0}', '${task.splits.test.1}', freq='1D'),
        )
    ),
)
# p_yaml(osse_nadir)

starter_recipe = results_prepostpro(_01=b(caller, 'rename', out='ssh'))
ost_recipe = results_prepostpro(_01=b(caller, 'rename', gssh='ssh'))
osse_nadir_results = hydra_zen.make_config(
    ref = from_recipe_and_inp(ref_postpro(), '${task.data.ssh}'),
    methods = {
        '4dvarnet': from_recipe_and_inp(starter_recipe,
            b(xr.open_dataset, '../oceanbench-data-registry/results/osse_gf_nadir/4dvarnet.nc')
        ),
        'miost': from_recipe_and_inp(ost_recipe,
            b(xr.open_dataset, '../oceanbench-data-registry/results/osse_gf_nadir/miost.nc')
        ),
    },
)
# p_yaml(osse_nadir_results)



build_eval_ds = hydra_zen.make_config(
    _0=pb(toolz.identity),
    _1=pb(toolz.assoc, dict(ref='${results.ref}'), 'study'),
    _2=pb(ocnuda.stack_dataarrays, ref_var='ref'),
    _3=b(caller, 'to_dataset', dim='variable'),
    _4=pb(ocndat.add_units, units=dict(study='meter', ref='meter')),
)
    

# eval_ds =
    ### Metrics

nrmse = hydra_zen.make_config(
    _1=pb(ocnmst.nrmse_ds, reference='ref', target='study', dim=['lat', 'lon', 'time']),
    _2=pb(float),
)
_psd_prepro=hydra_zen.make_config(
    _1=b(caller,'to_dataset', name='ssh'),
    _2=pb(ocnint.fillnan_gauss_seidel, variable='ssh'),
    _3=pb(ocnspa.latlon_deg2m),
    _4=pb(ocntem.time_rescale, freq_dt=1, freq_unit='day', t0='2012-10-22T12:00:00'),
    _5=b(igetter, 'ssh'),
)
psd_prepro = from_recipe(_psd_prepro())
base_lambda = hydra_zen.make_config(
    _1=b(caller, 'map', func=psd_prepro),
)

psd_kws = dict( ref_variable='ref', study_variable='study',
        detrend='constant', window='tukey', nfactor=2,
          window_correction=True, true_amplitude=True, truncate=True
)

lambda_x_isotrop = hydra_zen.make_config(
    _2=pb(ocnmps.psd_isotropic_score, psd_dims=['lon', 'lat'], avg_dims=['time'], **psd_kws),
    _3=b(igetter, 1),
    bases=(base_lambda,)
)
lambda_x_spacetime = hydra_zen.make_config(
    _2=pb(ocnmps.psd_spacetime_score, psd_dims=['time', 'lon'], avg_dims=['lat'], **psd_kws),
    _3=b(igetter, 1),
    bases=(base_lambda,)
)
lambda_t_spacetime = hydra_zen.make_config(
    _2=pb(ocnmps.psd_spacetime_score, psd_dims=['time', 'lon'], avg_dims=['lat'], **psd_kws),
    _3=b(igetter, 2),
    bases=(base_lambda,)
)
metrics = hydra_zen.make_config(
    nrmse=from_recipe(nrmse()),
    lambda_x_isotrop=from_recipe(lambda_x_isotrop()),
    lambda_x_spacetime=from_recipe(lambda_x_spacetime()),
    lambda_t_spacetime=from_recipe(lambda_t_spacetime()),
)

summary=hydra_zen.make_config(
    _1=pb(caller, '__call__'),
    _2=pb(toolz.valmap, d='${metrics}'),
)

# p_yaml(osse_nadir_results)

lambda_x_fmt = hydra_zen.make_config(
    _1=pb(operator.mul, 1e-3),
    _2=pb(caller, 'format'),
    _3=b(toolz.flip, toolz.apply, '{:.2f} km'),
)

lambda_t_fmt = hydra_zen.make_config(
    _2=pb(caller, 'format'),
    _3=b(toolz.flip, toolz.apply, '{:.2f} days'),
)
nrmse_fmt = hydra_zen.make_config(
    _2=pb(caller, 'format'),
    _3=b(toolz.flip, toolz.apply, '{:.3f}'),
)
metrics_fmt=hydra_zen.make_config(
    nrmse=from_recipe(nrmse_fmt()),
    lambda_x_isotrop=from_recipe(lambda_x_fmt()),
    lambda_x_spacetime=from_recipe(lambda_x_fmt()),
    lambda_t_spacetime=from_recipe(lambda_t_fmt()),
)


def minmax_cfg(v):
    return b(toolz.compose_left, 
        b(igetter, v),  # x -> x[v]
        b(toolz.juxt, pb(np.min), pb(np.max)), # x -> (min(x), max(x)) 
        pb(map, pb(float)), # [x1, x2] -> [float(x1), float(x2)]
        pb(tuple),  # [x1, x2] -> (x1, x2)
    )
### Plots

strain_pp = hydra_zen.make_config(
    _1=b(caller, 'to_dataset', name='ssh'),
    _12=pb(ocnval.validate_ssh),
    _2=pb(ocngeo.geostrophic_velocities),
    _3=pb(ocngeo.strain_magnitude),
    _4=pb(ocngeo.coriolis_normalized, variable='strain'),
    _5=b(operator.attrgetter, 'strain'),
)

get_plot_data = hydra_zen.make_config(
    _1=b(igetter, 'study'),
    _2=b(caller, 'isel', time=10),
)
levels= hydra_zen.make_config(
    _1=minmax_cfg('study'), # x -> (min(x[study]), max(x[study]))
    _2=pb(packed_caller, 'tick_values'), # [vmin, vmax]->  (x -> x.tick_values(vmin, vmax))
    _3=b(caller, '__call__', b(matplotlib.ticker.MaxNLocator, 5)), 
)
linestyles= hydra_zen.make_config(
    _4=pb(operator.le, 0),
    _5=b(oceanbench._src.utils.hydra.rpartial, pb(np.where) ,'-', '--'),
    bases=(levels,),
)

hvplot_contour = hydra_zen.make_config(
    _2=pb(caller, '__call__'), # x -> (f -> f(x))
    _3=pb(toolz.valmap, d=dict(  # f -> {k: f(v) for k,v..}
            levels=from_recipe(levels()),
            linestyle=from_recipe(linestyles())
    )),
    _4=pb(toolz.merge, dict( # d -> merge(d, ...)
        kind='contour',
        colorbar=False,
        aspect=1, x='lon', y='lat',
        alpha=0.5, linewidth=2,
        color='black',
    )),
    _5=pb(packed_caller, 'hvplot', []) # d: (x -> x.hvplot(**d))
)

contour_plot = hydra_zen.make_config(
    _1=b(toolz.juxt,  # x -> [(x-> x.hvplot(**d), x_plt)]
        pb(toolz.apply, b(zen_compose, asdict(hvplot_contour()))),
        pb(toolz.apply, b(zen_compose, asdict(get_plot_data()))),
    ),
    _2=pb(packed_caller, '__call__' ), # [f, inp] -> (x -> x(f, inp))
    _3=b(caller, '__call__', pb(toolz.apply)), # g -> g(toolz.apply) = apply(f, inp) = f(inp)
)

hvplot_image = hydra_zen.make_config(
    _2=pb(caller, '__call__'), # x -> (f -> f(x))
    _3=pbc(toolz.valmap, d=dict(  # f -> {k: f(v) for k,v..}
        xlim=minmax_cfg('lon'), ylim=minmax_cfg('lat'),
    )),
    _4=pbc(toolz.merge,
          dict( # d -> merge(d, ...)
        kind='image', cmap=cmocmap('speed'), clim=b(tuple, (0,30)),
        aspect=1, x='lon', y='lat',
    )
          ),
    _5=pb(packed_caller, 'hvplot', []), # d: (x -> x.hvplot(**d))
)
image_plot = hydra_zen.make_config(zen_dataclass=dict(cls_name='ImagePlot'),
    _1=b(toolz.juxt,  # x -> [(x-> x.hvplot(**d), x_plt)]
        from_recipe(hvplot_image()),
        from_recipe(get_plot_data()),
    ),
    _2=pb(packed_caller, '__call__' ), # [f, inp] -> (x -> x(f, inp))
    _3=b(caller, '__call__', pb(toolz.apply)), # g -> g(toolz.apply) = apply(f, inp) = f(inp)
)


strain_plt = hydra_zen.make_config(
    _1=b(toolz.juxt,  # x -> [(x-> x.hvplot(**d), x_plt)]
        from_recipe(image_plot()),
        from_recipe(contour_plot()),
    ),
    _2=pb(packed_caller, '__call__' ), # [f, inp] -> (x -> x(f, inp))
    _3=b(caller, '__call__', pb(operator.mul)), # g -> g(toolz.apply) = apply(f, inp) = f(inp)
)



plots = hydra_zen.make_config(
    strain=dict(
        pp=b(caller, 'map', from_recipe(strain_pp())),
        plt=from_recipe(strain_plt())
    )
)
leaderboard = hydra_zen.make_config(
    build_diag_ds=from_recipe(build_eval_ds()),
    plots=plots,
    metrics=metrics,
    metrics_fmt=metrics_fmt,
    summary_fmt=pb(join_apply, dfunc='${metrics_fmt}'),
    summary=pb(toolz.apply, b(zen_compose, asdict(summary()))),
    task=osse_nadir,
    results=osse_nadir_results,
    data=data_natl60,
)

il = I(leaderboard)
eval_ds = il.build_diag_ds(il.results.methods.miost())
summ = il.summary(eval_ds)
print(il.summary_fmt(summ))
pp, plt = il.plots.strain.pp, il.plots.strain.plt
plt(eval_ds.pipe(pp))
# st = eval_ds.map(strainfn)
### Plots


## Pipelines
### DataLoading

