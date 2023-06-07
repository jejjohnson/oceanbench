import itertools
from omegaconf import OmegaConf

def assign_var_attrs(ds, var_attrs):
    """
    Functional assignment of attributes of variables for xarray datasets
     ds = assign_var_attrs(ds, dict(var1=dict(units='m') var2=dict(long_name='toto')
    is equivalent to

    ds["var1"] = ds.var1.assign_attrs(units='m')
    ds["var2"] = ds.var2.assign_attrs(long_name='toto')
    """
    for v, attrs in var_attrs.items():
        ds[v].attrs = {**ds[v].attrs, **attrs}
    return ds

def concatdicts(*dicts):
    """
    Merge list of dicts (only shallowest layer)
    """
    return dict(
        zip(
            itertools.chain(*[d.keys() for d in dicts]),
            itertools.chain(*[d.values() for d in dicts]),
        )
    )

def rpartial(func, *args, **kwargs):
    """
    create partial function using right most positional arguments
    Useful for functions that take positional only arguments
    """
    def _f(*_args, **_kwargs):
        return func(*_args, *args, **concatdicts(kwargs, _kwargs))
    return _f

def call_if_callable(obj, inp=None):
    if not callable(obj): return obj
    if inp is None: return obj()
    return obj(inp)

def unpack_to(args=None, kwargs=None, to=None):
    args = args if args is not None else []
    kwargs = kwargs if kwargs is not None else {}
    return to(*args, **kwargs) #noqa

def const_fn(const):
    def _f(*args, **kwargs):
        return const

    return _f

## Workflow tools inspired by padl library  :
"""
pipe: function composition
rollout: dispatch of one input to multiple functions
parallel: match sequence of inputs with sequence of functions
"""
def pipe(inp, fns):
    out = inp
    for fn in fns:
        if callable(fn):
            out = fn(out)
        elif 'askw' in fn and 'fn' in fn:
            out = fn['fn'](**{fn['askw']: out})
        else:
            raise ValueError(f'{fn} should be callable or have "fn" and "askw" keys, instead has {fn.keys()}')


    return out


def rollout(inp, fns=[]):
    return [f(inp) if f is not None else inp for f in fns]

def parallel(inps, fns=[]):
    return [f(inp) if f is not None else inp for f, inp in zip(fns, inps)]


def rolloutdict(inp, dict_fns=[]):
    return {k: call_if_callable(f, inp) for k, f in dict_fns.items()}
