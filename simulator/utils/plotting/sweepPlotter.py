import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings


def getFigureData() -> (
    tuple[
        dict[str, str], dict[str, str], dict[str, str], dict[str, str], dict[str, str]
    ]
):
    """
    Convenience function to remove the axis data etc. used for the plot from the
    main script, which allows just calling this function to get all mappings.

    Returns
    -------
    5 dictionaries with strings as both keys and values. In order:
    1. The type of plot depending on which variable is plotted.
    2. Whether or not centralized and local solutions can be added to the plot
    3. The way in which the averaging should happen
    4. The labels for each variable.
    5. The x labels for each plot.
    6. The y labels for each plot depending on which variable is plotted.
    7. The limits for the y-axis for each plot
    8. The title for each plot depending on which variable is plotted.
    """
    types: dict[str, str] = {
        "LS_cost": "semilogy",
        "SINR": "regular",
        "MSE_w": "semilogy",
        "MSE_d": "semilogy",
        "STOI": "regular",
    }
    nonNetworkData: dict[str, bool] = {
        "LS_cost": True,
        "SINR": True,
        "MSE_w": False,
        "MSE_d": False,
        "STOI": True,
    }
    averageMethods: dict[str, str] = {
        "LS_cost": "normal",
        "SINR": "dB",
        "MSE_w": "normal",
        "MSE_d": "normal",
        "STOI": "normal",
    }
    labels: dict[str, str] = {
        "lFFT": r"$n_{FFT}$",
        "deltaUpdate": r"$\Delta_{update}$",
        "lmbd": r"$\lambda$",
        "GEVD": r"GEVD",
        "Gamma": r"$\Gamma$",
        "mu": r"$\mu$",
        "sequential": "sequential",
        "SRO": "SRO",
        "t60": r"$T_{60}$",
    }
    xlabels: dict[str, str] = {
        "LS_cost": "samples",
        "SINR": "samples",
        "MSE_w": "samples",
        "MSE_d": "samples",
        "STOI": "samples",
    }
    ylabels: dict[str, str] = {
        "LS_cost": "LS cost",
        "SINR": "SINR [dB]",
        "MSE_w": r"$MSE_W$",
        "MSE_d": r"$MSE_d$",
        "STOI": "STOI",
    }
    ylims: dict[str, list[float]] = {
        "LS_cost": [1e-3, 1e1],
        "SINR": [0, 25],
        "MSE_w": [1e-3, 1e1],
        "MSE_d": [1e-3, 1e1],
        "STOI": [0, 1],
    }
    titles: dict[str, str] = {
        "LS_cost": r"Progression of LS cost over time",
        "SINR": "Progression of SINR over time",
        "MSE_w": r"Progression of $MSE_w$ over time",
        "MSE_d": r"Progression of $MSE_d$ over time",
        "STOI": "Progression of STOI over time",
    }
    return (
        types,
        nonNetworkData,
        averageMethods,
        labels,
        xlabels,
        ylabels,
        ylims,
        titles,
    )


def getMaps(nColors: int) -> tuple[list[str], dict[str, list[str]]]:
    """
    yet another convenience function to avoid cluttering the main script. This
    time the maps (main colormap, to be sliced; markers; linestyles) are returned.

    Returns
    -------
    A list of strings indicating the sequencing of how to cycle through the
    styles alongside a dictionary with keys the styles to cycle through and
    the corresponding values as a list.
    """
    plasmaData = mpl._cm_listed._plasma_data
    plasmaData = plasmaData[: 5 * len(plasmaData) // 6]  # cut out brightest part
    if nColors == 1:
        colorMap = [plasmaData[0]]
    else:
        colorMap = [None for _ in range(nColors)]
        for i in range(len(colorMap)):
            idx = int(len(plasmaData) / (nColors - 1) * i)
            idx = np.minimum(np.maximum(idx, 0), len(plasmaData) - 1)
            colorMap[i] = plasmaData[idx]

    linestyleMap: list[str] = ["-", "--", "-."]
    markerMap: list[str] = ["o", "x", "+", ">", "<", "^", "v"]
    styleMaps: dict[str, list[str]] = {
        "color": colorMap,
        "marker": markerMap,
        "linestyle": linestyleMap,
    }
    styles = ["color", "marker", "linestyle"]

    return styles, styleMaps


def updatePlot(
    timestamps: np.ndarray,
    nwData: np.ndarray,
    label: str,
    linestyle: dict[str, str] = dict(),
    fig: plt.Figure | None = None,
    type: str = "regular",
) -> plt.Figure:
    """
    Update the passed in figure `fig` or create a new one with the passed in
    data: `timestamps` should be giving the x-axis data, `data` is the data to
    be plotted.

    Parameters
    ----------
    timestamps: np.ndarray
        The x coordinates of points to plot

    data: np.ndarray
        The y coordinates of points to plot

    locData: np.ndarray | None
        The metric for the localized solution or `None`

    centrData: np.ndarray | None
        The metric for the centralized solution or `None`

    label: str
        The label to give to the plot

    linestyle: str, optional
        The linestyle to use, if not passed in, the default matplotlib cycle is
        used (solid lines, no markers, default colorcycle)

    fig: plt.Figure, optional
        If a figure is passed in, its most recent axis is updated (the function
        if only designed for figures with just one axis). If nothing is passed
        in, a new figure is created that is returned.

    type: str, optional
        The type of plot to make: for now only "regular" and "semilogy" are
        supported. Defaults to "regular". If an unknown value is passed in this
        function returns without plotting anything, just a figure is returned
        (which could be potentially empty if either an empty figure was passed
        in or `None` was passed in).

    Returns
    -------
    A matplotlib figure that is potentially augmented with new data.
    """
    if fig == None:
        fig, ax = plt.subplots()
        fig.set_size_inches(8.5, 5.5)
    else:
        ax = fig.gca()

    # use the correct function for plotting depending on `type`
    match type:
        case "regular":
            plotFnct = ax.plot
        case "semilogy":
            plotFnct = ax.semilogy
        case _:
            warnings.warn("Type not recognized, just returning the passed in figure!")
            return fig

    if len(linestyle.keys()) != 0:
        plotFnct(
            timestamps,
            nwData,
            fillstyle="none",
            markevery=0.1,
            linewidth=2,
            label=label,
            **linestyle,
        )
    else:
        plotFnct(timestamps, nwData, linewidth=2, label=label)

    ax.grid(True)
    ax.autoscale(tight=True, axis="x")
    ax.legend()
    fig.tight_layout()
    return fig


def addNonNetworkData(
    fig: plt.Figure, localValue: float, centralValue: float, params: dict[str, object]
) -> plt.Figure:
    """
    Given a figure and metrics for the local and centralized solutions, add
    horizontal lines to the figure and annotate them.
    """
    ax = fig.gca()

    valLabel = ""
    for param, val in params.items():
        valLabel += f"{param} = {val}, "
    valLabel = valLabel[:-2]  # remove trailing ", "

    ax.axhline(localValue, linestyle="--", color="k")
    ax.axhline(centralValue, linestyle="--", color="k")
    ax.annotate(f"local: {valLabel}", xy=(0, localValue), fontsize=8)
    ax.annotate(f"central: {valLabel}", xy=(0, centralValue), fontsize=8)

    return fig


def finalizePlot(
    fig: plt.Figure, xlabel: str, ylabel: str, ylim: list[float], title: str
) -> plt.Figure:
    """
    Add labels to the figure and perform some final formatting
    """
    ax = fig.gca()
    ax.grid(True, which="both")
    ax.legend(fontsize=10, loc="upper right")
    ax.autoscale(tight=True, axis="x")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.set(ylim=ylim)
    fig.tight_layout()
    return fig
