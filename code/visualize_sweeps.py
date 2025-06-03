# Purpose of script:
# Visualize the results of the parameter sweeps performed by "main_sweep.py".
# Allows to select which parameters to visualize exactly.
#
# Workflow:
# The user should fill in the key-value pairs to fix and vary specific values
# such that those can be visualized. This should happen in a separate file
# called `variables.py`. This folder also contains the folder to extract the
# data from as well as which metrics to plot (in a list of strings) and which
# nodes to use for the computations/averaging (as a list of integers).
#
# Context:
# Msc thesis DANSE, parameter sweeps to determine the optimal ones, visualization
#
# (c) Basil Liekens

import ast  # needed to convert the string representations of the csv back to actual lists
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from parameter_sweep import parameterGenerator, sweepParameters
from signal_generation import Parameters
import sys
import utils
import visualization_variables as vars  # avoid having to push all data all the time


def main():
    folderName = vars.folderName  # the folder with the results to visualize
    variables = vars.variables
    coupling = vars.coupling
    metrics = vars.metrics
    nodes = vars.nodes

    ## setup
    maxColors = 6  # predefined value to avoid cluttering the graph
    maxNonSingletons = 3  # predefined value to ensure everything gets an "id"
    centrVariables = [
        "lFFT",
        "GEVD",
        "Gamma",
        "mu",
    ]  # vars that lead to != centr solutions, SRO could be included if desired

    ## dataloading
    basePath = os.path.join("simulator", "output", "sweeps")
    folder = os.path.join(basePath, folderName)
    df = pd.read_csv(os.path.join(folder, "data.csv"))
    cfg = Parameters().load_from_yaml(os.path.join(folder, "cfg.yml"))

    df, variables, metrics, nodes = cleanInputs(df, variables, metrics, nodes, cfg)
    keys = list(variables.keys())

    ## process user input
    nNonSingletons = np.sum([0 if len(variables[key]) == 1 else 1 for key in keys])
    if nNonSingletons > maxNonSingletons:
        raise RuntimeError(
            f"More than {maxNonSingletons} encountered, impossible to visualize!"
        )

    # start by building up some stuff
    lengths = [len(variables[key]) for key in keys]
    idcs = np.argsort(lengths)[::-1][:nNonSingletons]  # flip then select non-singleton

    nColors = np.maximum(1, len(variables[keys[idcs[0]]]))
    if nColors > maxColors:
        raise RuntimeError(
            f"Too many colors needed: max number={maxColors}, received {nColors}."
        )

    styles, styleMaps = utils.plotting.getMaps(nColors)
    types, nonNetworkData, methods, labels, xlabels, ylabels, ylims, titles = (
        utils.plotting.getFigureData()
    )

    df = groupRows(df, variables, methods, coupling)

    # obtain the rows that will be used for centralized and local solutions
    # there are a set of rows that all have the same set of values for the
    # variables that make a difference for the centralized and local solutions,
    # hence it is sufficient to only keep track of one of them and add that one
    # to the plot.
    # filter the unneeded keys
    centrVariables = list(filter(lambda x: x in keys, centrVariables))

    genVals = []
    params = sweepParameters()  # not the cleanest solution, but it works.
    for var in centrVariables:
        genVals.append((var, list(variables[var])))

    combinations = parameterGenerator(params, genVals, coupling=coupling)
    centrRows = []
    for combination in combinations():
        tmpDf = df
        for var in centrVariables:
            tmpDf = tmpDf[tmpDf[var] == combination.__getattribute__(var)]

        centrRows.append(tmpDf.index[0])  # all centr values equal => use "0"

    # assign identifiers: will be identified by the variable (first dict), then
    # a tuple with str saying which element of the style is addressed, and a
    # second dict with all possible values of the variable that map to the
    # identifier.
    identifiers: dict[str, tuple[str, dict[object, str]]] = dict()
    for i, idx in enumerate(idcs):
        styleName = styles[i]
        variable = keys[idx]  # which variable will determine the style element
        values = variables[variable]
        mapping = dict()

        # Use a string as a key to avoid problems with lists as keys
        # This also requires transforming the keys in the plotting itself!
        for j, value in enumerate(values):
            mapping[str(value)] = styleMaps[styleName][j]

        identifiers[variable] = (styleName, mapping)

    ## plotting
    for metric in metrics:
        fig, ax = plt.subplots()
        # iterate over all rows of the dataframe
        for i in df.index:
            row: pd.DataFrame = df.loc[i]

            # build up linestyle and legend
            linestyle = {
                "color": styleMaps["color"][0],
                "marker": "o",
                "linestyle": "-",
            }
            legend = ""
            for variable in identifiers.keys():
                styleName, mapping = identifiers[variable]
                value = row.loc[variable]
                legend += f"{labels[variable]}={value}, "

                # mapping[variable] returns the tuple containing the name of the
                # modifier for the line and a dictionary with keys the relevant values
                linestyle[styleName] = mapping[str(value)]

            # gather necessary data to be able to plot
            data = gatherData(row, metric, nodes, averageMethod=methods[metric])
            fig = utils.plotting.updatePlot(
                data["timestamps"],
                data["nw"],
                legend[:-2],  # remove traling ", "
                linestyle,
                fig,
                type=types[metric],
            )

            # add centralized and local solutions if possible
            if i in centrRows and nonNetworkData[metric]:
                params = dict()
                for variable in centrVariables:
                    params[variable] = row[variable]
                fig = utils.plotting.addNonNetworkData(
                    fig, data["loc"], data["centr"], params
                )

        ## take care of good layout
        fig = utils.plotting.finalizePlot(
            fig,
            xlabel=xlabels[metric],
            ylabel=ylabels[metric],
            ylim=ylims[metric],
            title=titles[metric],
        )
    print(cfg)
    plt.show(block=True)


def cleanInputs(
    df: pd.DataFrame,
    variables: dict[str, list[object]],
    metrics: list[str],
    nodes: list[int],
    p: Parameters,
) -> tuple[pd.DataFrame, dict[str, list[object]], list[str], list[int]]:
    """
    Perform cleanup of the dataframe: remove rows that don't correspond to a
    combination of interest, convert the columns to the correct format for
    further handling, ...

    Parameters
    -----------
        df: pd.DataFrame
            The read-in csv to clean up

        variables: dict[str, list[object]]
            The variables of interest

        metrics: list[str]
            The metrics to plot

        nodes: list[int]
            The nodes of interest, the rest will be filtered out of the dataframe

        p: Parameters
            The parameters for the sweeps

    Returns
    --------
        The cleaned up dataframe + the adjusted set of variables (e.g. the
        blanks are filled in.)
    """
    keys = list(variables.keys())

    # remove unnecessary columns from the dataframe & unused variables etc.
    nValidHeaders = len(keys) + len(metrics) + 1  # + 1 for timestamps
    masks = np.zeros((nValidHeaders, len(df.columns)), dtype=bool)

    for idx, header in enumerate(keys):  # pedantic checking for variables
        masks[idx, :] = list(map(lambda x: header == x, df.columns))
    for idx, header in enumerate(metrics):  # metrics have _idx, so be more lenient
        masks[idx + len(keys), :] = list(map(lambda x: header in x, df.columns))
    masks[-1, :] = list(map(lambda x: "timestamps" in x, df.columns))

    colMasks = np.any(masks, axis=0)
    df = df.loc[:, colMasks]

    # clean up the variables and the metrics, use np arrays to use boolean masks
    headerMasks = np.any(masks, axis=1)
    variablesMask = headerMasks[: len(variables.keys())]
    metricMask = headerMasks[len(variables.keys()) : -1]

    cleanedVariables = dict()
    maskedKeys = np.array(keys)[variablesMask]
    for key in maskedKeys:
        cleanedVariables[str(key)] = variables[key]  # convert back to string
    variables = cleanedVariables

    metrics = list(np.array(metrics)[metricMask])

    keys = list(variables.keys())
    # slice of relevant rows of the dataframe
    for key in keys:
        # expand the slices
        if variables[key] == []:
            variables[key] = list(pd.unique(df[key]))

        # if the variables are lists, convert them into strings as lists are
        # unhashable, which leads to problems in the assignment of linestyles
        if isinstance(variables[key][0], list):
            variables[key] = list(map(lambda x: str(x), variables[key]))

        # remove other values than the desired ones from the dataframe (when not
        # all values are present in the "variables" dictionary e.g. for clarity)
        df = df[df[key].isin(variables[key])]

    # clean up nodes to be only valid numbers
    if nodes == []:
        nodes = [k for k in range(p.K)]
    else:
        nodes = list(filter(lambda x: x >= 0 and x < p.K, nodes))

    # convert the stringified lists back to regular lists
    for metric in metrics:
        for column in df.columns:
            if metric in column and df[column].dtype == "O":
                df[column] = df[column].apply(ast.literal_eval)

    df["timestamps"] = df["timestamps"].apply(ast.literal_eval)

    return df, variables, metrics, nodes


def groupRows(
    df: pd.DataFrame,
    variables: dict[str, list[object]],
    averageMethods: dict[str, str],
    coupling: list[list[str]],
) -> pd.DataFrame:
    """
    Group the rows that correspond to the same set of parameters, but have
    different monte-carlo parameters (room-dimensions, sound sources, ...)

    Parameters
    ----------
        df: pd.DataFrame
            The dataframe

        variables: dict[str, list[object]]
            The variables of interest for the visualization

        averageMethods: dict[str, str]
            The way in which averaging should happen

        coupling: list[list[str]]
            Variables that should be changed simultaneously, requires the
            corresponding values to have equal lengths.
    """
    generatorValues = []
    for key, value in variables.items():
        generatorValues.append((key, value))
    generator = parameterGenerator(
        sweepParameters(), generatorValues, coupling=coupling
    )

    newDf = pd.DataFrame(columns=df.columns)
    variableNames = variables.keys()

    for combination in generator():
        tmpDf = df.copy()
        newRow = pd.DataFrame(columns=df.columns, index=[newDf.shape[0]], dtype=object)

        # retrieve all relevant columns to average over
        for var in variableNames:
            tmpDf: pd.DataFrame = tmpDf[tmpDf[var] == combination.__getattribute__(var)]
            newRow[var] = combination.__getattribute__(var)

        # set the timestamp, assume they are equal for all monte-carlo variables
        newRow.at[newRow.index[0], "timestamps"] = tmpDf["timestamps"][tmpDf.index[0]]

        # average over all rows, per metric
        metricCols = list(
            filter(lambda x: x not in [*variableNames, "timestamps"], df.columns)
        )

        for column in metricCols:
            data = np.array(list(tmpDf[column]))
            metric = list(filter(lambda x: x in column, averageMethods.keys()))[0]
            newRow.at[newRow.index[0], column] = averageData(
                data, method=averageMethods[metric]
            )

        if newDf.shape[0] == 0:
            newDf = newRow
        else:
            newDf = pd.concat((newDf, newRow))

    return newDf


def gatherData(
    row: pd.DataFrame, metric: str, nodes: list[int], averageMethod: str = "normal"
) -> dict[str, float | list[float]]:
    """
    Given a row and a metric, extract all relevant data from that row:
    timestamp, data from the networkwide filters and local and/or centralized
    values to be able to plot later on.

    Parameters
    ----------
    row: pd.DataFrame
        The row of the csv to extract data from

    metric: str
        The metric to extract

    nodes: list[int]
        The nodes involved in the computation

    averageMethod: str, "normal" or "dB"
        How to average across nodes: `normal` just does an arithmetic mean, "dB"
        first transforms the values back to regular, then does the averaging and
        lastly goes back to dB. It is assumed all metrics are "powers"; dB
        computations happen with `10 log10()`.

    Returns
    -------
    A dictionary with the relevant keys: `timestamps`, `nw` and optionally `loc`
    and `centr` if those latter two were present in the dataframe.
    """
    results = dict()
    results["timestamps"] = row["timestamps"]

    # extract networkwide information
    nwData = []
    for k in nodes:
        nwData.append(row[f"{metric}_nw_{k}"])

    nwData = np.array(nwData)
    results["nw"] = averageData(nwData, averageMethod)

    localHeader = f"{metric}_loc_{nodes[0]}"
    centrHeader = f"{metric}_centr_{nodes[0]}"
    if localHeader in row.index:
        localData = []
        for k in nodes:
            localData.append(row[f"{metric}_loc_{nodes[k]}"])
        results["loc"] = averageData(localData, averageMethod)
    else:
        results["loc"] = None

    if centrHeader in row.index:
        centrData = []
        for k in nodes:
            centrData.append(row[f"{metric}_centr_{nodes[k]}"])
        results["centr"] = averageData(centrData, averageMethod)
    else:
        results["centr"] = None

    return results


def averageData(data: np.ndarray, method: str = "normal") -> np.ndarray:
    match method:
        case "normal":
            data = np.mean(data, axis=0)
        case "dB":  # get back to the nominal value, not the power!
            dataNormal = np.power(10, np.array(data) / 20)
            dataAvg = np.mean(dataNormal, axis=0)
            data = 20 * np.log10(dataAvg)
        case _:
            raise ValueError(f"Unknown averagemethod encountered! ({method})")
    return data


if __name__ == "__main__":
    mpl.use("TkAgg")  # avoid issues when plotting
    plt.ion()
    sys.exit(main())
