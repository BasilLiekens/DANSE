from __future__ import annotations  # resolve circular dependency
from dataclasses import dataclass
import DANSE_base
import numpy as np
import yaml


@dataclass
class logger:
    """
    base logger class with basic functionality.
    """

    def append(self, name: str, value):
        """
        Append "value" to the "name" attribute of logger. If not existent, a
        warning is raised.

        Classes inheriting from this class are only allowed to have list
        attributes if they want to make use of the "append" functionality.
        """
        try:
            data = getattr(self, name)
            data.append(value)
        # will catch both the error of the field not existing as well as the error of appending to a non-list attribute
        except:
            print(f"Something went wrong during appending, skipping this step.")


@dataclass
class nodeLogger(logger):
    """
    Class for storing the Ws and Gs of a node in the network. This also provides
    the option to store audio and noise contributions. However, it is discouraged
    to do so in the case of batchmode simulations to avoid taking up too much
    memory. Audio and noise data is assumed to be fed in the frequency domain.

    No bookkeeping is done to limit the amount of overhead for the logging. In
    principle this could/should be done (check the dimensions of the data)
    """

    #
    K: int = 10  # nb nodes in the network
    Mk: int = 2  # nb mics per node
    R: int = 1  # nb channels to communicate
    #
    lFFT: int = 1024  # nb of fft bins
    #

    def __post_init__(self):
        self._Wkk: list[np.ndarray] = []
        self._Gk: list[np.ndarray] = []
        self._audio: list[np.ndarray] = []  # contribution of desired signal to output
        self._noise: list[np.ndarray] = []  # noise and interferer contribtion to out
        self._d: list[np.ndarray] = []  # the reference signal during an update


@dataclass
class FCLogger(logger):
    """
    Logger that plays the role of the central data sink in a fully connected
    network. Takes care of some of the functions the central orchestrator needs
    to make everything more readable: the core class should only be concerned
    with the orchestration.

    This class then takes care of the data collection as well as the required
    utility functions.

    It effectively breaks some of the encapsulation rules with respect to the
    nodeLogger, but uses them read-only + considers them as peer object for ease
    of use + efficiency of implementations (deepcopying at every iteration would
    be expensive to say the least)
    """

    #
    K: int = 10  # number of nodes in the network
    Mk: int = 2  # number of mics per node
    R: int = 1  # number of channels that's being communicated between nodes
    #
    lFFT: int = 1024  # number of frequency bins in the fft
    #

    def __post_init__(self):
        self._nodeLoggers: list[nodeLogger] = [None for _ in range(self.K)]
        self._nwFilters: list[list[np.ndarray]] = [[] for _ in range(self.K)]

    def link(self, nodeNb: int, node: DANSE_base.node.batchNode):
        """
        Effectively says that "node" is the "nodeNb"'th node of the network and
        uses the node only for the logger as this is what is of importance.
        """
        # breaks some encapsulation rules, but makes it significantly easier +
        # is used as read-only.
        self._nodeLoggers[nodeNb] = node._logger

    def computeNetworkWideFilter(self, nodeNb: int) -> np.ndarray:
        """
        Compute the network-wide equivalent of a filter corresponding to a
        specific node. Follows equation (37) in the "DANSE part I: Sequential
        node updating" paper by A. Bertrand where it is used that Gkk = I.

        /!\\ Note: returns a numpy array that is constructed through views on the
        nodeloggers. Hence, overwriting part of the matrix would render the
        data of the loggers incorrect /!\\ .
        """
        Wkk = self._nodeLoggers[nodeNb]._Wkk[-1]
        Gk = self._nodeLoggers[nodeNb]._Gk[-1]

        w_NW = np.zeros(
            (int(np.ceil(self.lFFT / 2 + 1)), self.Mk * self.K, self.R),
            dtype=np.complex128,
        )

        w_NW[:, nodeNb * self.Mk : (nodeNb + 1) * self.Mk, :] = Wkk

        for k in range(nodeNb):
            wSlice = (
                self._nodeLoggers[k]._Wkk[-1] @ Gk[:, k * self.R : (k + 1) * self.R, :]
            )
            w_NW[:, k * self.Mk : (k + 1) * self.Mk, :] = wSlice

        for k in range(nodeNb + 1, self.K):
            wSlice = (
                self._nodeLoggers[k]._Wkk[-1] @ Gk[:, (k - 1) * self.R : k * self.R, :]
            )
            w_NW[:, k * self.Mk : (k + 1) * self.Mk, :] = wSlice

        self._nwFilters[nodeNb].append(w_NW)  # log the filter immediately
        return w_NW

    def getOutput(
        self, nodeNb: int, mode: str = "full"
    ) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray]
        | tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]
    ):
        """
        Get the total output of a node alongside its noise and audio contributions.

        The user is responsible itself for ensuring no duplicates are present in
        the output (e.g. in the context of batchmode simulations where the
        results are stored at every iteration).

        The output is returned in the frequency domain!

        Parameters
        ----------
        nodeNb: int
            The number of the node for which to return the outputs

        mode: str, optional
            How the outputs should be returned: two options: directly concatenate
            all frames ("full") or get the individual contributions to e.g.
            compute progression over time ("frames"). Defaults to "full".

        Returns
        -------
        A tuple with 3 elements (4 if `mode == "full"`):
            - The full output (audio + noise and interference)

            - The contribution of the audio to the output

            - The contribution of the noise to the output

            - (Optional) The desired signal

        The precise form of these outputs depends on the `mode` argument. If
        mode is `full` all of these outputs will be np.ndarrays (3D: frequency
        domain). Else they will be lists of 3D np.ndarrays: each entry being the
        outputs gathered in one updating cycle.
        """
        audio = self._nodeLoggers[nodeNb]._audio
        noise = self._nodeLoggers[nodeNb]._noise
        d = self._nodeLoggers[nodeNb]._d
        match mode:
            case "full":
                audioOut = np.concatenate(audio, axis=2)
                noiseOut = np.concatenate(noise, axis=2)
                return audioOut + noiseOut, audioOut, noiseOut
            case "frames":
                combined = [audio[i] + noise[i] for i in range(len(audio))]
                return combined, audio, noise, d
            case _:
                raise ValueError("Unrecognized option for `mode`")

    def getNetworkWideFilters(self, nodeNb: int) -> list[np.ndarray]:
        return self._nwFilters[nodeNb]
