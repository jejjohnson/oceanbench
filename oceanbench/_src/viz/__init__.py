import numpy as np 
import matplotlib.ticker as ticker

def add_top_wavelength_hook(plot, element, xlabel=f"Wavelength [km]", fmt="{x:.0f}"):
    ax = plot.handles['axis']
    secax = ax.secondary_xaxis(
        "top", functions=(lambda x: 1 / (x + 1e-20), lambda x: 1 / (x + 1e-20))
    )

    secax.xaxis.set_major_formatter(fmt)
    secax.set(xlabel=xlabel)

def wavenumber_to_wavelength(plot, element, xlabel=f"Wavelength [km]", fmt="{x:.0f}"):
    ax = plot.handles['axis']
    secax = ax.secondary_xaxis(
        "bottom", functions=(lambda x: 1 / (x + 1e-20), lambda x: 1 / (x + 1e-20))
    )

    secax.xaxis.set_major_formatter(fmt)
    secax.set(xlabel=xlabel)

def frequency_to_period(plot, element, xlabel=f"Period [days]", fmt="{x:.0f}"):
    ax = plot.handles['axis']
    secax = ax.yaxis(
        "left", functions=(lambda x: 1 / (x + 1e-20), lambda x: 1 / (x + 1e-20))
    )

    secax.xaxis.set_major_formatter(fmt)
    secax.set(xlabel=xlabel)



def xformatter(plot, el):
    plot.handles['axis'].xaxis.set_major_locator(ticker.LogLocator())
    plot.handles['axis'].xaxis.set_minor_locator(ticker.LogLocator(subs='all'))
    plot.handles['axis'].grid(True, 'both', 'x')
    plot.handles['axis'].xaxis.set_major_formatter(lambda x, _: f'{x:.0f}')

def ygrid_major(plot, el):
    plot.handles['axis'].yaxis.set_major_locator(ticker.LogLocator(numticks=100))
    plot.handles['axis'].grid(True, 'major', 'y')
    plot.handles['axis'].grid(False, 'minor', 'y')
