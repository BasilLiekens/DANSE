from dataclasses import dataclass, field
import pandas as pd


@dataclass
class simulationContainer:
    """
    Class that is used to postprocess the results from a simulation and stores
    them appropriately.

    Parameters
    ----------
    K: int
        The number of nodes in the network

    Mk: int
        The number of channels per node (assumed to be equal for each one)

    R: int
        The number of channels that's being communicated between nodes

    fs: int
        The sampling frequency of the simulations

    variables: list[str]
        The variables that were being swept over

    metrics: dict[str, list[str]]
        The metrics that were tracked in "launch": the keys are the types used
        "nw", "loc", "centr" the values are lists containing which metrics.

    path_to_file: str
        The path to where to save the .csv file.
    """

    #
    K: int = 3  # the number of nodes in the network
    Mk: int = 4  # the number of microphones per node
    R: int = 1  # the number of signals being communicated between nodes
    #
    fs: float = int(16e3)  # the sampling frequency of the simulations
    #
    variables: list[str] = field(default_factory=list("lmbd"))  # variables swept over
    metrics: dict[str, list[str]] = field(default_factory=list("SINR"))
    #
    path_to_file: str = "data.csv"
    #

    def __post_init__(self):
        # Create the dataframe with the appropriate columns: the variables of the
        # sweep alongside the timestamps of each of the metrics (allows to
        # compare sweeps that update at different rates + easily visualize the
        # progression). Lastly, all the desired metrics are also in there, each
        # node is represented.
        self.columns = self.variables + ["timestamps"]
        for type in self.metrics:
            metrics = self.metrics[type]
            metricTypes = [f"{metric}_{type}" for metric in metrics]
            for k in range(self.K):
                nodeK = [f"{metricType}_{k}" for metricType in metricTypes]
                self.columns += nodeK

        # use `object` to be able to store lists in the dataframe
        self.df: pd.DataFrame = pd.DataFrame(columns=self.columns, dtype=object)

    def processSimulation(
        self,
        vars: dict[str, object],
        timestamps: list[int],
        computedMetrics: dict[str, float | list[float]],
    ):
        """
        Process the simulation by computing the metrics of interest and adding a
        record to the dataframe to reflect this simulation

        Parameters
        ----------
        vars: dict[str, object]
            A dictionary with keys the variables that are being swept over,
            should only contain exactly the ones passed in when instantiating
            this class.

        timestamps: list[int]
            The timestamps at which the updates occurred. Enables to compare
            updates that happen at different frequencies.
        """
        # bookkeeping
        if set(computedMetrics.keys()) != set(self.columns[len(self.variables) + 1 :]):
            raise ValueError(
                "computedMetrics should have the same set of keys as passed in originally!"
            )
        if set(vars.keys()) != set(self.variables):
            raise ValueError(
                "The passed in variables dict should have the same entries as this object!"
            )

        # create new row and fill in the values
        newRow = pd.DataFrame(columns=self.df.columns, index=[0], dtype=object)
        newRow.at[0, "timestamps"] = timestamps
        for var, value in vars.items():
            newRow.at[0, var] = value

        for metric, value in computedMetrics.items():
            newRow.at[0, metric] = value

        # get a monotonic index by replacing index of the row
        newRow.rename(index={0: self.df.shape[0]}, inplace=True)

        # append new row to the dataframe (if it's not empty, else just replace)
        if self.df.shape[0] == 0:
            self.df = newRow
        else:
            self.df = pd.concat((self.df, newRow))

        # write back after each iteration for safety
        self.store(self.path_to_file)

    def store(self, path: str):
        self.df.to_csv(path, index=False)
