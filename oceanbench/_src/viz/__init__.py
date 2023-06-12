def add_top_wavelength_hook(plot, element, xlabel=f"Wavelength [km]", fmt="{x:.0f}"):
    ax = plot.handles['axis']
    secax = ax.secondary_xaxis(
        "top", functions=(lambda x: 1 / (x + 1e-20), lambda x: 1 / (x + 1e-20))
    )

    secax.xaxis.set_major_formatter(fmt)
    secax.set(xlabel=xlabel)
