import xarray as xr


def regridstack_dataarrays(dataarrays, ref_grid_var=None):
    import ocn_tools._src.geoprocessing.gridding

    dataarrays = {k: v() if callable(v) else v for k, v in dataarrays.items()}

    ref_grid_var, ref_grid = (
        (ref_grid_var, dataarrays[ref_grid_var])
        if ref_grid_var is not None
        else next(iter(dataarrays.items()))
    )

    dataarrays = {
        k: ocn_tools._src.geoprocessing.gridding.grid_to_regular_grid(
            src_grid_ds=v,
            tgt_grid_ds=ref_grid,
            keep_attrs=True,
        )
        if k != ref_grid_var
        else v
        for k, v in dataarrays.items()
    }

    dataarrays = {
        k: v.assign_coords(time=ref_grid.time) if k != ref_grid_var else v
        for k, v in dataarrays.items()
    }
    # print(ref_grid, dataarrays)

    return xr.Dataset(dataarrays).to_array()


def stack_dataarrays(dataarrays, ref_var=None):
    dataarrays = {k: v() if callable(v) else v for k, v in dataarrays.items()}

    ref_var, ref_grid = (
        (ref_var, dataarrays[ref_var])
        if ref_var is not None
        else next(iter(dataarrays.items()))
    )

    dataarrays = {
        k: v.assign_coords(lat=ref_grid.lat, lon=ref_grid.lon).assign_coords(
            time=ref_grid.time
        )
        if k != ref_var
        else v
        for k, v in dataarrays.items()
    }

    return xr.Dataset(dataarrays).to_array()
