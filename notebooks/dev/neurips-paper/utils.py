import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from oceanbench._src.metrics.utils import find_intercept_1D, find_intercept_2D
#from oceanbench._src.metrics.power_spectrum import plot_psd_isotropic


def plot_psd_isotropic_score(
    da: xr.DataArray, 
    scale="km", 
    ax=None, 
    color: str="k", 
    name: str="model", 
    **kwargs
):
    
    if scale == "km":
        factor = 1e3
    elif scale == "m":
        factor = 1.0
        rfactor = 1
        
    else:
        raise ValueError(f"Unrecognized scale")
    
    fig, ax, secax = plot_psd_isotropic(
        da=da, scale=scale, ax=ax, **kwargs
    )
    
    ax.set(ylabel="PSD Score", yscale="linear")
    ax.set_ylim((0,1.0))
    ax.set_xlim((
        np.ma.min(np.ma.masked_invalid(da.freq_r.values * factor)),
        np.ma.max(np.ma.masked_invalid(da.freq_r.values * factor)),
    ))
    resolved_scale = factor / da.attrs["resolved_scale_space"]
    
    ax.vlines(
        x=resolved_scale, ymin=0, ymax=0.5, color=color, linewidth=2, 
        linestyle="--",
    )
    ax.hlines(
        y=0.5,
        xmin=np.ma.min(np.ma.masked_invalid(da.freq_r.values * factor)),
        xmax=resolved_scale, color=color,
        linewidth=2, linestyle="--",
    )
    
    ax.set_aspect("equal", "box")
    
    label = f"{name}: {1/resolved_scale:.0f} {scale} "
    ax.scatter(
        resolved_scale, 0.5, 
        color=color, marker=".", linewidth=5, label=label,
        zorder=3
    )
    
    return fig, ax, secax


def plot_psd_spacetime_wavenumber_score(
    da: xr.DataArray, 
    space_scale: str=None,
    psd_units: str=None,
    ax=None
):
    
    if space_scale == "km":
        space_scale = 1e3
        xlabel = "Wavenumber [cycles/km]"
    elif space_scale == "m":
        space_scale = 1.0
        xlabel = "Wavenumber [cycles/m]"
    elif space_scale is None:
        space_scale = 1.0
        xlabel = "Wavenumber [k]"
    else:
        raise ValueError(f"Unrecognized scale: {space_scale}")
        
    if psd_units is None:
        cbar_label = "PSD"
    else:
        cbar_label = f"PSD [{psd_units}]"
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
    
    pts = ax.contourf(
        1/(da.freq_lon*space_scale),
        1/da.freq_time, 
        da.transpose("freq_time", "freq_lon"), 
        extend="both",
        cmap="RdBu", 
        levels=np.arange(0, 1.1, 0.1)
    )

    ax.set(
        yscale="log",
        xscale="log",
        xlabel=xlabel,
        ylabel="Frequency [cycles/days]",
    )
    # colorbar

    cbar = fig.colorbar(
        pts,
        pad=0.02,
    )
    cbar.ax.set_ylabel(cbar_label)

    plt.grid(which="both", linestyle="--", linewidth=1, color="black", alpha=0.2)

    pts_middle = ax.contour(
        1/(da.freq_lon*space_scale),
        1/da.freq_time, 
        da.transpose("freq_time", "freq_lon"), 
        levels=[0.5], 
        linewidths=2, 
        colors="k"
    )

    cbar.add_lines(pts_middle)

    return fig, ax, cbar


def plot_psd_spacetime_score_wavelength(
    da, space_scale="km", psd_units=None, ax=None
):
    
    if space_scale == "km":
        xlabel = "Wavelength [km]"
    elif space_scale == "m":
        xlabel = "Wavelength [m]"
    elif space_scale is None:
        xlabel = "Wavelength k"
    else:
        raise ValueError(f"Unrecognized scale: {space_scale}")

    fig, ax, cbar = plot_psd_spacetime_wavenumber_score(
        da, space_scale=space_scale, psd_units=psd_units, ax=ax)

    ax.set(yscale="log", xscale="log", xlabel=xlabel, ylabel="Period [days]")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.yaxis.set_major_formatter("{x:.0f}")

    return fig, ax, cbar

class PlotPSDIsotropic:
        
    def init_fig(self, ax=None, figsize=None):
        if ax is None:
            figsize = (5,4) if figsize is None else figsize
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.ax = ax
            self.fig = plt.gcf()
        
    def plot_wavenumber(self, da,freq_scale=1.0, units=None, **kwargs):
        
        if units is not None:
            xlabel = f"Wavenumber [cycles/{units}]"
        else:
            xlabel = f"Wavenumber"
            
        dim = list(da.dims)[0]
        
        self.ax.plot(da[dim] * freq_scale, da, **kwargs)

        self.ax.set(
            yscale="log", xscale="log",
            xlabel=xlabel,
            ylabel=f"PSD [{da.name}]",
            xlim=[10**(-3) - 0.00025, 10**(-1) +0.025]
        )

        self.ax.legend()
        self.ax.grid(which="both", alpha=0.5)
        
    def plot_wavelength(self, da, freq_scale=1.0, units=None, **kwargs):
        
        if units is not None:
            xlabel = f"Wavelength [{units}]"
        else:
            xlabel = f"Wavelength"
        
        self.ax.plot(1/(da[dim] * freq_scale), da, **kwargs)
        
        self.ax.set(
            yscale="log", xscale="log",
            xlabel=xlabel,
            ylabel=f"PSD [{da.name}]"
        )

        self.ax.xaxis.set_major_formatter("{x:.0f}")
        self.ax.invert_xaxis()
        
        self.ax.legend()
        self.ax.grid(which="both", alpha=0.5)
                
    def plot_both(self, da, freq_scale=1.0, units=None, **kwargs):
        
        if units is not None:
            xlabel = f"Wavelength [{units}]"
        else:
            xlabel = f"Wavelength"
        
        self.plot_wavenumber(da=da, units=units, freq_scale=freq_scale, **kwargs)
        
        self.secax = self.ax.secondary_xaxis(
            "top", functions=(lambda x: 1 / (x + 1e-20), lambda x: 1 / (x + 1e-20))
        )
        self.secax.xaxis.set_major_formatter("{x:.0f}")
        self.secax.set(xlabel=xlabel)
        

class PlotPSDScoreIsotropic(PlotPSDIsotropic):
    
    def _add_score(
        self,
        da,
        freq_scale=1.0, 
        units=None, 
        threshhold: float=0.5, 
        threshhold_color="k",
        name=""
):
        
        dim = da.dims[0]
        self.ax.set(ylabel="PSD Score", yscale="linear")
        self.ax.set_ylim((0,1.0))
        self.ax.set_xlim((
            10**(-3) - 0.00025,
            10**(-1) +0.025,
        ))
        
        resolved_scale = freq_scale / find_intercept_1D(
            x=da.values, y=1./da[dim].values, level=threshhold
        )        
        self.ax.vlines(
            x=resolved_scale, 
            ymin=0, ymax=threshhold, 
            color=threshhold_color,
            linewidth=2, linestyle="--",
        )
        self.ax.hlines(
            y=threshhold,
            xmin=10**(-3) - 0.00025,
            #xmin=np.ma.min(np.ma.masked_invalid(da[dim].values * freq_scale)),
            xmax=resolved_scale, color=threshhold_color,
            linewidth=2, linestyle="--"
        )        
        label = f"{name}: {1/resolved_scale:.0f} {units} "
        self.ax.scatter(
            resolved_scale, threshhold,
            color=threshhold_color, marker=".",
            linewidth=5, label=label,
            zorder=3
        )
        
        
    def plot_score(
        self, 
        da, 
        freq_scale=1.0, 
        units=None, 
        threshhold: float=0.5, 
        threshhold_color="k",
        name="",
        **kwargs
    ):
        
        self.plot_both(da=da, freq_scale=freq_scale, units=units, **kwargs)
        self._add_score(
            da=da, 
            freq_scale=freq_scale,
            units=units,
            threshhold=threshhold, 
            threshhold_color=threshhold_color,
            name=name
        )
        
        
class PlotPSDSpaceTime:
    def init_fig(self, ax=None, figsize=None):
        if ax is None:
            figsize = (5,4) if figsize is None else figsize
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.ax = ax
            self.fig = plt.gcf()
        
    def plot_wavenumber(
        self, 
        da, 
        space_scale: float=1.0,
        space_units: str=None,
        time_units: str=None,
        psd_units: float=None,
        **kwargs):
        
        if space_units is not None:
            xlabel = f"Wavenumber [cycles/{space_units}]"
        else:
            xlabel = f"Wavenumber"
        if time_units is not None:
            ylabel = f"Frequency [cycles/{time_units}]"
        else:
            ylabel = f"Frequency"

        if psd_units is None:
            cbar_label = "PSD"
        else:
            cbar_label = f"PSD [{psd_units}]"
            
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        
        locator = ticker.LogLocator()
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        
        
        pts = self.ax.contourf(
            1/(da.freq_lon*space_scale),
            1/da.freq_time, 
            da.transpose("freq_time", "freq_lon"), 
            norm=norm, 
            locator=locator, 
            cmap=kwargs.pop("cmap", "RdYlGn"), 
            extend=kwargs.pop("extend", "both"),
            vmin=vmin, vmax=vmax,
            **kwargs
        )

        self.ax.set(
            yscale="log",
            xscale="log",
            xlabel=xlabel,
            ylabel=ylabel,
            
        )
        # colorbar
        fmt = ticker.LogFormatterMathtext(base=10)
        cbar = plt.colorbar(
            pts,
            ax=self.ax,
            pad=0.02,
            format=fmt,
            extend=True,
            norm=norm
            
        )
        cbar.ax.set_ylabel(cbar_label)
        self.ax.invert_xaxis()
        self.ax.invert_yaxis()
        self.ax.grid(which="both", linestyle="--", linewidth=1, color="black", alpha=0.2)


    def plot_wavelength(        
        self, 
        da, 
        space_scale: float=1.0,
        space_units: str=None,
        time_units: str=None,
        psd_units: float=None,
        **kwargs
    ):
    
        if space_units is not None:
            xlabel = f"Wavelength [{space_units}]"
        else:
            xlabel = f"Wavelength"
            
        if time_units is not None:
            ylabel = f"Period [{time_units}]"
        else:
            ylabel = f"Period"
            
        if psd_units is None:
            cbar_label = "PSD"
        else:
            cbar_label = f"PSD [{psd_units}]"
            
        self.plot_wavenumber(
            da=da, space_scale=space_scale, 
            space_units=space_units, time_units=time_units,
            psd_units=psd_units, 
            **kwargs
        )

        self.ax.set(
            xlabel=xlabel, 
            ylabel=ylabel,
        )
        self.ax.xaxis.set_major_formatter("{x:.0f}")
        self.ax.yaxis.set_major_formatter("{x:.0f}")
        
        
class PlotPSDSpaceTimeScore:
    def init_fig(self, ax=None, figsize=None):
        if ax is None:
            figsize = (5,4) if figsize is None else figsize
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.ax = ax
            self.fig = plt.gcf()
        
    def plot_wavenumber(
        self, 
        da, 
        space_scale: float=1.0,
        space_units: str=None,
        time_units: str=None,
        psd_units: float=None,
        threshhold: float=0.5,
        **kwargs):
        
        if space_units is not None:
            xlabel = f"Wavenumber [cycles/{space_units}]"
        else:
            xlabel = f"Wavenumber"
        if time_units is not None:
            ylabel = f"Frequency [cycles/{time_units}]"
        else:
            ylabel = f"Frequency"

        if psd_units is None:
            cbar_label = "PSD Score"
        else:
            cbar_label = f"PSD Score [{psd_units}]"
        
        
        pts = self.ax.contourf(
            1/(da.freq_lon*space_scale),
            1/da.freq_time, 
            da.transpose("freq_time", "freq_lon"), 
            cmap=kwargs.pop("cmap", "RdBu"), 
            extend=kwargs.pop("extend", "both"),
            levels=np.arange(0, 1.1, 0.1),
            **kwargs
        )

        self.ax.set(
            yscale="log",
            xscale="log",
            xlabel=xlabel,
            ylabel=ylabel,
        )
        # colorbar
        self.cbar = plt.colorbar(
            pts,
            ax=self.ax,
            pad=0.02,
        )
        self.cbar.ax.set_ylabel(cbar_label)

        self.ax.grid(which="both", linestyle="--", linewidth=1, color="black", alpha=0.2)
        
        pts_middle = self.ax.contour(
            1/(da.freq_lon * space_scale),
            1/da.freq_time,
            da.transpose("freq_time", "freq_lon"),
            levels=[threshhold],
            linewidths=2,
            colors="k",
        )
        
        self.cbar.add_lines(pts_middle)
        
        self.ax.invert_xaxis()
        self.ax.invert_yaxis()


    def plot_wavelength(        
        self, 
        da, 
        space_scale: float=1.0,
        space_units: str=None,
        time_units: str=None,
        psd_units: float=None,
        threshhold: float=0.5,
        **kwargs
    ):
    
        if space_units is not None:
            xlabel = f"Wavelength [{space_units}]"
        else:
            xlabel = f"Wavelength"
            
        if time_units is not None:
            ylabel = f"Period Score [{time_units}]"
        else:
            ylabel = f"Period Score"
            
        if psd_units is None:
            cbar_label = "PSD"
        else:
            cbar_label = f"PSD [{psd_units}]"

        self.plot_wavenumber(
            da=da, space_scale=space_scale, 
            space_units=space_units, time_units=time_units,
            psd_units=psd_units, threshhold=threshhold
        )

        self.ax.set(
            xlabel=xlabel, 
            ylabel=ylabel
        )
        self.ax.xaxis.set_major_formatter("{x:.0f}")
        self.ax.yaxis.set_major_formatter("{x:.0f}")


def plot_psd_isotropic_wavenumber(
    da: xr.DataArray, scale: str="km", units: str=None, ax=None, **kwargs):
    
    if scale == "km":
        scale = 1e3
        xlabel="Wavenumber [cycles/km]"
    else:
        scale = 1.0
        xlabel="Wavenumber [cycles/m]"
        
    if units is None:
        ylabel = "PSD"
    else:
        ylabel = f"PSD [{units}]"

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    else:
        fig = plt.gcf()

    ax.plot(da.freq_r * scale, da, **kwargs)

    ax.set(
        yscale="log",
        xscale="log",
        xlabel=xlabel,
        ylabel=ylabel,
    )

    ax.legend()
    ax.grid(which="both", alpha=0.5)

    return fig, ax


def plot_psd_isotropic_wavelength(
    da: xr.DataArray, scale: str="km", units: str=None, ax=None, **kwargs
):
    
    
    if scale == "km":
        xlabel="Wavenumber [km]"
    else:
        xlabel="Wavenumber [m]"
        
    if units is None:
        ylabel = "PSD"
    else:
        ylabel = f"PSD [{units}]"

    fig, ax = plot_psd_isotropic_wavenumber(
        da, 
        ax=ax, 
        scale=scale, 
        units=units,
        **kwargs
    )

    ax.set(
        yscale="log",
        xscale="log",
        xlabel=xlabel,
        ylabel=ylabel,
    )

    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.invert_xaxis()

    return fig, ax


def plot_psd_isotropic(
    da: xr.DataArray, scale: str="km", units: str=None, ax=None, **kwargs
):
    
    if scale == "km":
        xlabel="Wavenumber [km]"
    else:
        xlabel="Wavenumber [m]"

    fig, ax = plot_psd_isotropic_wavenumber(
        da=da, ax=ax, scale=scale, units=units, **kwargs
    )

    secax = ax.secondary_xaxis(
        "top", functions=(lambda x: 1 / (x + 1e-20), lambda x: 1 / (x + 1e-20))
    )
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.set(xlabel=xlabel)

    return fig, ax, secax


def plot_psd_spacetime_wavenumber(
    da: xr.DataArray, 
    space_scale: str=None,
    psd_units: str=None,
    ax=None
):
    
    if space_scale == "km":
        space_scale = 1e3
        xlabel = "Wavenumber [cycles/km]"
    elif space_scale == "m":
        space_scale = 1.0
        xlabel = "Wavenumber [cycles/m]"
    elif space_scale is None:
        space_scale = 1.0
        xlabel = "Wavenumber [k]"
    else:
        raise ValueError(f"Unrecognized scale: {space_scale}")
        
    if psd_units is None:
        cbar_label = "PSD"
    else:
        cbar_label = f"PSD [{psd_units}]"
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    locator = ticker.LogLocator()
    norm = colors.LogNorm()
    

    pts = ax.contourf(
        1/(da.freq_lon*space_scale),
        1/da.freq_time, 
        da.transpose("freq_time", "freq_lon"), 
        norm=norm, 
        locator=locator, 
        cmap="RdYlGn", 
        extend="both"
    )

    ax.set(
        yscale="log",
        xscale="log",
        xlabel=xlabel,
        ylabel="Frequency [cycles/days]",
    )
    # colorbar
    fmt = ticker.LogFormatterMathtext(base=10)
    cbar = plt.colorbar(
        pts,
        ax=ax,
        pad=0.02,
        format=fmt,
    )
    cbar.ax.set_ylabel(cbar_label)
    
    ax.grid(which="both", linestyle="--", linewidth=1, color="black", alpha=0.2)

    return fig, ax, cbar


def plot_psd_spacetime_wavelength(da, space_scale="km", psd_units=None, ax=None):
    
    if space_scale == "km":
        xlabel = "Wavelength [km]"
    elif space_scale == "m":
        xlabel = "Wavelength [m]"
    elif space_scale is None:
        xlabel = "Wavelength k"
    else:
        raise ValueError(f"Unrecognized scale: {space_scale}")
        
    fig, ax, cbar = plot_psd_spacetime_wavenumber(
        da, space_scale=space_scale, psd_units=psd_units, ax=ax)
    ax.set(yscale="log", xscale="log", xlabel=xlabel, ylabel="Period [days]")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.yaxis.set_major_formatter("{x:.0f}")

    return fig, ax, cbar