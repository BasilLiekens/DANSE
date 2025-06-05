from __future__ import annotations  # resolve circular dependency
from dataclasses import dataclass
from .MWF import MWF_computation, GEVD_MWF_computation
import numpy as np
from typing_extensions import Any, Self
import utils
import warnings


@dataclass
class DANSENode:
    """
    Parent class for the "batchNode" and "onlineNode". Defines the generic
    methods that are needed by these inheriting classes.

    Parameters
    -----------------
        K: int
            The number of nodes in the network

        Mk: int
            The number of channels in per node (assumed to be fixed for all nodes)

        R: int
            Number of channels to communicate between nodes

        lFFT: int
            Number of samples in the FFT

        overlap: float
            percentage overlap between subsequent stft frames

        windowType: str
            the type of window (currently only "sqrt hanning" and "ones" supported)

        GEVD: bool
            whether or not to use a GEVD-based updating rule

        Gamma: float, non-negative
            A regularization constant to be used during the updates of DANSE.
            Currently only relevant for regular DANSE, not for GEVD-DANSE.
            (should be >= 0!)

        mu: float, non-negative
            Weighting factor to control the trade-off speech distortion vs.
            noise reduction.

        seed:
            the seed to use for numpy random initializations

        trackOutput:
            Whether or not to keep track of the outputs and desired signal every
            time that is requested (not really ergonomic, but easiest to
            implement that way. Other option would be to always compute the
            outputs and log and then just return that value when requested)
    """

    #
    K: int = 10
    Mk: int = 2
    R: int = 1
    #
    lFFT: int = 1024
    overlap: float = 0.5
    windowType: str = "sqrt hanning"
    #
    GEVD: bool = False
    Gamma: float = 0
    mu: float = 1
    #
    seed: int = 64
    #
    trackOutput: bool = False
    #

    def __post_init__(self):
        """Update derived attributes"""
        np.random.seed(self.seed)

        if self.windowType == "sqrt hanning":
            self.window: np.ndarray = np.sqrt(np.hanning(self.lFFT))
        elif self.windowType == "ones":
            self.window: np.ndarray = 1 / np.sqrt(1 / self.overlap) * np.ones(self.lFFT)
        else:
            warnings.warn("Window type not recognized, using scaled ones instead.")
            self.window: np.ndarray = 1 / np.sqrt(1 / self.overlap) * np.ones(self.lFFT)

        # selection matrix to determine the desired signals
        self.E: np.ndarray = np.vstack(
            (np.eye(self.R), np.zeros((self.Mk + (self.K - 2) * self.R, self.R)))
        )

        # autocorrelation matrices
        self.Ryy: np.ndarray = np.zeros(
            (
                int(np.ceil(self.lFFT / 2 + 1)),
                self.Mk + (self.K - 1) * self.R,
                self.Mk + (self.K - 1) * self.R,
            ),
            dtype=np.complex128,
        )
        self.Rnn: np.ndarray = np.zeros_like(self.Ryy)

        # create an identity matrix along each fft bin
        idx = np.arange(self.Ryy.shape[1])
        self.Ryy[:, idx, idx] = 1e0
        self.Rnn[:, idx, idx] = 1e-1

        # weight matrices
        self.Wkk: np.ndarray = np.random.randn(
            int(np.ceil(self.lFFT / 2 + 1)), self.Mk, self.R
        ).astype(np.complex128)
        self.Gk: np.ndarray = np.random.randn(
            int(np.ceil(self.lFFT / 2 + 1)), (self.K - 1) * self.R, self.R
        ).astype(np.complex128)

        self.Wkk_H: np.ndarray = np.conj(self.Wkk.transpose(0, 2, 1))
        self.Gk_H: np.ndarray = np.conj(self.Gk.transpose(0, 2, 1))

        # "pure" signals: size ["lFFT / 2 + 1" x R x any]
        self.audio: np.ndarray = np.random.randn(
            int(np.ceil(self.lFFT / 2 + 1)), self.R, int(1e3)
        )
        self.noise: np.ndarray = np.random.randn(
            int(np.ceil(self.lFFT / 2 + 1)), self.R, int(1e3)
        )

        # fused signals: size ["lFFT / 2 + 1" x R x any]
        self.fusedAudio: np.ndarray = np.random.randn(
            int(np.ceil(self.lFFT / 2 + 1)), self.R, int(1e3)
        )
        self.fusedNoise: np.ndarray = np.random.randn(
            int(np.ceil(self.lFFT / 2 + 1)), self.R, int(1e3)
        )

        # logger
        self._logger: utils.logging.nodeLogger = utils.logging.nodeLogger(
            self.K, self.Mk, self.R, self.lFFT
        )
        self._logger.append("_Wkk", self.Wkk)
        self._logger.append("_Gk", self.Gk)

        # connected nodes
        self.connectedNodes: list[Self] = []

    def link(self, otherNode: Self):
        """
        Link two nodes together, meaning they can communicate to each other
        directly for updating without the orchestrator having to collect all
        data and having to pass it around.

        It should be noted this method is mostly defined as a parent class
        method for convenience. linking a "batchNode" and an "onlineNode" will
        lead to weird results.
        """
        # bookkeeping to check if there are no duplicates
        if self in otherNode.connectedNodes or otherNode in self.connectedNodes:
            warnings.warn(
                "This node is already connected to the node to which it is linked now!"
            )

        elif self == otherNode:
            warnings.warn("The nodes that are being attempted to link are the same!")

        else:
            self.connectedNodes.append(otherNode)
            otherNode.connectedNodes.append(self)

    def getFusedSignals(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return a (copy of) the most recently fused signals
        """
        return self.fusedAudio, self.fusedNoise

    def getWeights(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return a (copy of) the weights
        """
        return self.Wkk, self.Gk

    def getOutput(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the output based on the most recently received inputs and fused
        signals as well as the most recently computed filter weights.

        This is also logged for easier handling.
        """
        dtilde, ntilde = self._gatherMeasurements()

        # speed up the computation by exploiting the fact that a part of the
        # computations was done already.
        dIn = self.E.T @ dtilde
        dOut = self.fusedAudio + self.Gk_H @ dtilde[:, self.Mk :, :]
        nOut = self.fusedNoise + self.Gk_H @ ntilde[:, self.Mk :, :]

        if self.trackOutput:
            self._logger.append("_d", dIn)
            self._logger.append("_audio", dOut)
            self._logger.append("_noise", nOut)

        return dOut, nOut

    def receive(self, audio: np.ndarray, noise: np.ndarray):
        """
        Given audio and noise, fuse them with the Wkk and both return and store
        the lastly fused signals (store them as well to be able to communicate).

        Parameters
        ---------------
            audio:  3D numpy array of dimensions
                    ["lFFT" x "nb channels" x "nb frames"] containing the
                    contribution of the desired signal to each of the mics
            noise:  3D numpy array of same dimensions as "audio", but now for
                    the contributions of the noise.

        Returns
        ---------------
            The fused signals. Shape ["lFFT" x "R" x "nb frames"]. This result
            is also stored to be able to let the nodes communicate themselves.
        """
        if audio.ndim != 3 or noise.ndim != 3:
            raise ValueError("Audio and noise should be three dimensional")
        if audio.shape != noise.shape:
            raise ValueError("Audio and noise should have the same shape")
        if audio.shape[0] != self.lFFT / 2 + 1 or audio.shape[1] != self.Mk:
            raise ValueError(
                f"Audio and noise should have first dimension {self.lFFT / 2 + 1}"
                "(single-sided fft) and second {self.Mk}"
            )

        self.audio = audio
        self.noise = noise
        self.fusedAudio = self.Wkk_H @ audio
        self.fusedNoise = self.Wkk_H @ noise

    def updateAutocorrelations(self, vad: np.ndarray = None):
        """
        Given the most recent incoming audio and noise, update Ryy and Rnn. This
        has to be a separate function as all nodes should have received their
        samples for them to be fused with the proper weights and communicated.

        This function should always be called directly after "receive" to ensure
        the autocorrelations stay in sync with the data.

        The vad should also be passed in to allow for construction of the
        autocorrelation matrices as would be done in practice.

        Groundtruth information is only used in the case of "vad" being "None".
        Else just the vad is used to segment the data. This means that Rnn will
        not be updated if vad contains no 0's (to prevent crashes the previous
        value is retained in that case).
        """
        pass

    def _gatherMeasurements(self) -> tuple[np.ndarray, np.ndarray]:
        # obtain fused signals of connected nodes
        zAudio = []
        zNoise = []
        for node in self.connectedNodes:
            fusedAudio, fusedNoise = node.getFusedSignals()
            zAudio.append(fusedAudio)
            zNoise.append(fusedNoise)

        # create the "tilde" signals, concatenate along the first axis as that's
        # the "channel axis"
        dtilde = np.concatenate((self.audio, *zAudio), axis=1)
        ntilde = np.concatenate((self.noise, *zNoise), axis=1)

        return dtilde, ntilde

    def update(self, alpha: float = 1):
        """
        Perform an update of the local node parameters.

        Parameters
        ----------------
            GEVD:   whether or not to do a GEVD-based update
            alpha:  the amount of data to take into account from the previous
                    update (allows to do an rS-DANSE_K update instead of common
                    DANSE)

        Returns
        ----------------
            a copy of the new Wkk and Gk
        """
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha should be in [0, 1]")

        if self.GEVD:
            W_sol = GEVD_MWF_computation(
                self.Ryy, self.Rnn, self.E, self.Gamma, self.mu
            )
        else:
            W_sol = MWF_computation(self.Ryy, self.Rnn, self.E, self.Gamma, self.mu)

        self.Wkk = (1 - alpha) * self.Wkk + alpha * W_sol[:, : self.Mk, :]
        self.Gk = W_sol[:, self.Mk :, :]

        self.Wkk_H = np.conj(self.Wkk.transpose(0, 2, 1))
        self.Gk_H = np.conj(self.Gk.transpose(0, 2, 1))

        self._logger.append("_Wkk", self.Wkk)
        self._logger.append("_Gk", self.Gk)

    def __eq__(self, other: any):
        """
        Only consider two nodes to be equal if they ARE the same object, don't
        just look at their attributes (what python does by default).
        """
        if not self is other:
            return False
        else:
            return True


@dataclass
class batchNode(DANSENode):
    """
    Class that contains nodes for batch updates. I.e., there is no state in the
    covariance matrices: every time update is called the autocorrelation is
    recomputed based on the newly received data. This is in contrast to online
    nodes that update their correlation matrices every time a new datapoint
    comes in.

    Parameters
    ------------------
        K: int
            the number of nodes in the network

        Mk: int
            the number of channels per node (assumed to be fixed)

        R: int
            the number of channels to communicate between nodes

        lFFT: int
            the number of samples in the fft

        overlap: float
            percentage overlap between subsequent stft frames

        windowType: str
            the type of window, currently only sqrt hanning and ones are supported.

        GEVD: bool
            Boolean indicating whether or not to use a GEVD updating rule

        Gamma: float, non-negative
            Regularization constant for the updates of DANSE (just regular DANSE,
            GEVD-DANSE doesn't suffer from the same problem as the regular one).

        seed:       the seed to be set.
    """

    #
    K: int = 2  # nb nodes in network, needed for preallocation (including this node)
    Mk: int = 10  # nb channels per node (assumed to be fixed)
    R: int = 4  # nb channels to communicate
    #
    lFFT: int = 1024  # nb samples in the STFT
    overlap: float = 0.5  # percentage overlap in STFT
    windowType: str = "sqrt hanning"
    #
    GEVD: bool = False
    Gamma: float = 0
    mu: float = 1
    #
    seed: int = 64
    #
    trackOutput: bool = False
    #

    def __post_init__(self):
        """Update derived attributes"""
        super().__post_init__()

    def updateAutocorrelations(self, vad: np.ndarray = None):
        """
        Construct (normalized autocorrelations).

        Batch mode simulations, hence all data is used and equally weighted.
        Previous information is discarded.
        """
        dtilde, ntilde = self._gatherMeasurements()
        ytilde = dtilde + ntilde

        if vad is None:
            self.Rnn = (ntilde @ np.conj(ntilde.transpose(0, 2, 1))) / ntilde.shape[2]
            self.Ryy = (ytilde @ np.conj(ytilde.transpose(0, 2, 1))) / ytilde.shape[2]
        else:
            # only update correlation matrix if there are enough segments
            # available to still have a full rank matrix: stupid edgecases where
            # there are only a very select number of frames available to update
            # either matrix would otherwise lead to "eigh" crashing (due to rank)
            if np.sum(~vad) >= ytilde.shape[1]:
                self.Rnn = (
                    ytilde[:, :, ~vad] @ np.conj(ytilde[:, :, ~vad].transpose(0, 2, 1))
                ) / np.sum(~vad)

            if np.sum(vad) >= ytilde.shape[1]:
                self.Ryy = (
                    ytilde[:, :, vad] @ np.conj(ytilde[:, :, vad].transpose(0, 2, 1))
                ) / np.sum(vad)

    def __eq__(self, other: Any):
        """
        Only consider two nodes to be equal if they ARE the same object, don't
        just look at their attributes (what python does by default).
        """
        return super().__eq__(other)


@dataclass
class onlineNode(DANSENode):
    """
    Class that contains the structure for the online node updating: now there is
    some state in the correlation matrices.

    Parameters
    -----------------
        K: int
            The number of nodes in the network

        Mk: int
            The number of channels in per node (assumed to be fixed for all nodes)

        R: int
            Number of channels to communicate between nodes

        lFFT: int
            Number of samples in the FFT

        overlap: float
            percentage overlap between subsequent stft frames

        windowType: str
            the type of window (currently only "sqrt hanning" and "ones" supported)

        GEVD: bool
            whether or not to use a GEVD based updating rule

        Gamma: float, non-negative
            Regularization constant for the DANSE updates, only relevant for
            regular DANSE for now.

        seed: int
            seed to use for numpy random (initialization)

        deltaUpdate: int
            the number of samples that should be processed before an update of the filters.

        lmbd: float
            the smoothing factor for the updates of the correlation matrices.
    """

    #
    K: int = 10
    Mk: int = 2
    R: int = 1
    #
    lFFT: int = 1024
    overlap: float = 0.5
    windowType: str = "sqrt hanning"
    #
    GEVD: bool = False
    Gamma: float = 0
    mu: float = 1
    #
    seed: int = 64
    #
    trackOutput: bool = True
    #
    deltaUpdate: int = int(1e4)
    lmbd: float = 0.999
    #

    def __post_init__(self):
        super().__post_init__()

    def updateAutocorrelations(self, vad: np.ndarray = None):
        """
        Update the old autocorrelation matrices using the most recent new data.

        If vad is passed in as "None", groundtruth is used for Rnn, else the
        vad is used for assigning which frame (even if that implies either of
        the two is not updated in that segment).
        """
        dtilde, ntilde = self._gatherMeasurements()
        ytilde = dtilde + ntilde

        # segment data
        if vad is None:
            noiseSegments = ntilde
            fullSegments = ytilde
        else:
            noiseSegments = ytilde[:, :, ~vad]
            fullSegments = ytilde[:, :, vad]

        # update autocorrelations
        for i in range(noiseSegments.shape[2]):
            seg = noiseSegments[:, :, [i]]
            self.Rnn = self.lmbd * self.Rnn + (1 - self.lmbd) * (
                seg @ np.conj(seg.transpose(0, 2, 1))
            )

        for i in range(fullSegments.shape[2]):
            seg = fullSegments[:, :, [i]]
            self.Ryy = self.lmbd * self.Ryy + (1 - self.lmbd) * (
                seg @ np.conj(seg.transpose(0, 2, 1))
            )

    def __eq__(self, other: Any):
        return super().__eq__(other)
