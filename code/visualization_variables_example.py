"""
Input for `visualize_sweeps.py`. Same remark as `sweep_variables_example.py`,
make your own to avoid having to push this file all the time!

"User input"
Variables to select on: keys are the variables to select in the dataframe,
values are lists of values for that variable to select, if the list is
empty, all potential values are selected.

Only variables that have 0 or no entries in their list get assigned a
distinguishing feature (color, marker, linestyle). Assignment happens in
that order where it is used that dictionaries have a fixed return sequence
in python: the largest non-singleton gets assigned the color, the second
gets assigned the marker and the third one gets assigned a linestyle. If
there are more than three non-singletons, execution stops, similar for the
case where there are more values than "features" available.
"""

# the folder containing the relevant data
folderName: str = "Path/to/sweep/output/folder"

# dictionary containing which values should be selected
variables: dict[str, list[object]] = dict()
variables["lFFT"] = [1024]
variables["deltaUpdate"] = [10, 50, 100]
variables["lmbd"] = [0.9, 0.99, 0.999]
variables["GEVD"] = [False]
variables["Gamma"] = [0.0]
variables["mu"] = [1.0]
variables["sequential"] = [True]

# which variables to couple for the visualization, requires both of them to have
# equal lengths
coupling: list[list[str]] = []

# which metrics to plot
metrics: list[str] = ["LS_cost", "SINR", "MSE_w", "MSE_d", "STOI"]

# which nodes to use for the visualization
nodes: list[int] = []
