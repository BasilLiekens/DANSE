from __future__ import annotations  # resolve circular dependency
from dataclasses import dataclass
from .MWF import MWF_computation, GEVD_MWF_computation
import numpy as np
import os
import warnings


@dataclass
class DANSENode:
    """
    Class that is adapted from the `onlineNode` class in the simulator, but now
    for use on the RPi's in the MARVELO framework. The most significant change
    being the way in which the inputs are handled: rather than collecting
    measurements from other nodes, the inputs are given now to allow a job to
    call the updating method after receiving all needed inputs

    Parameters
    ----------
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

        deltaUpdate: int
            The number of frames to be collected prior to updating

        lmbd: float
            Smoothing parameter for the autocorrelation matrices
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
    sequential: bool = True
    nodeNb: int = 0  # sequential updating => offset to remove need for orchestrator
    #
    alpha0: float = 1  # initial scaling for the alpha parameter
    alphaFormat: str = "harmonic"
    #
    seed: int = 64
    #
    deltaUpdate: int = int(1e4)
    lmbd: float = 0.999
    #
    basePath: str = os.path.join("/home", "RPi", "installations")
    semaphoreName: str = "semaphore"
    weightsName: str = "weights.npz"
    WkkName: str = "Wkk.npz"
    #

    def __post_init__(self):
        """Update derived attributes"""
        np.random.seed(self.seed)

        # some bookkeeping
        if self.alphaFormat not in ["harmonic", "ones"]:
            warnings.warn("Alpha format not recognized, defaulting to ones")
            self.alphaFormat = "ones"

        if self.sequential:
            self.frameIdx: int = (self.K - 1 - self.nodeNb) * self.deltaUpdate + 1
            self.deltaUpdate *= self.K  # sequential => updating frequency decreases
        else:
            self.frameIdx: int = 0
        self.updateIdx: int = 1  # start at one to avoid 1/0 in harmonic case

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

        # weight matrices, initialize randomly to avoid weird artefacts
        self.Wkk: np.ndarray = np.random.randn(
            int(np.ceil(self.lFFT / 2 + 1)), self.Mk, self.R
        ).astype(np.complex128)
        self.Gk: np.ndarray = np.random.randn(
            int(np.ceil(self.lFFT / 2 + 1)), (self.K - 1) * self.R, self.R
        ).astype(np.complex128)

        self.W: np.ndarray = np.concatenate((self.Wkk, self.Gk), axis=1)
        self.W_H: np.ndarray = np.conj(self.W.transpose(0, 2, 1))

        self.Wkk_H: np.ndarray = self.W[:, : self.Mk, :]
        self.Gk_H: np.ndarray = self.W_H[:, self.Mk :, :]

        # bookkeeping to store the weights for later reference
        self.weights: list[np.ndarray] = []
        self.weights.append(self.W)

    def step(self, ytilde: np.ndarray, vad: np.ndarray) -> np.ndarray:
        """
        Update the old autocorrelation matrices using the most recent new data.

        Every frame is used individually, if the time is there to update, do so.

        In the end the filtered signals are returned (in the frequency domain)

        Parameters
        ----------
            ytilde: 3D np.ndarray
                The concatenation of the own measurements and the incoming fused
                signals from other nodes. Shape:
                [`lFFT // 2 + 1` x `Mk + (K-1)R` x `nFrames`]

            vad: 1D, np.ndarray
                A frequency domain per-frame VAD with an equal amount of frames
        """
        # filter for output
        filtered = self.W_H @ ytilde

        # iterate over frames, if needed, update
        for i in range(ytilde.shape[2]):
            self.frameIdx += 1
            frame = ytilde[:, :, [i]]
            corrUpdate = frame @ np.conj(frame.transpose(0, 2, 1))

            if vad[i]:
                self.Ryy = self.lmbd * self.Ryy + (1 - self.lmbd) * corrUpdate
            else:
                self.Rnn = self.lmbd * self.Rnn + (1 - self.lmbd) * corrUpdate

            if self.frameIdx % self.deltaUpdate == 0:
                alpha = self._constructNextAlpha()
                self._update(alpha)
                self.updateIdx += 1

        return filtered

    def _update(self, alpha: float = 1):
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
        if self.GEVD:
            W_sol = GEVD_MWF_computation(
                self.Ryy, self.Rnn, self.E, self.Gamma, self.mu
            )
        else:
            W_sol = MWF_computation(self.Ryy, self.Rnn, self.E, self.Gamma, self.mu)

        self.Wkk = (1 - alpha) * self.Wkk + alpha * W_sol[:, : self.Mk, :]
        self.Gk = W_sol[:, self.Mk :, :]

        self.W = np.concatenate((self.Wkk, self.Gk), axis=1)
        self.W_H = np.conj(self.W.transpose(0, 2, 1))

        self.Wkk_H = self.W_H[:, : self.Mk, :]
        self.Gk_H = self.Gk_H[:, self.Mk :, :]

        # some bookkeeping to track progress etc.
        self.weights.append(self.W)
        self._storeWeights()

    def _constructNextAlpha(self) -> float:
        """
        Construct a generator for the alphas for the updates of the weights.
        If the updating is sequential, this is always 1, else there is some
        underlying rule.
        """
        if self.sequential:
            base = 1
        else:
            match self.alphaFormat:
                case "harmonic":
                    base = 1 / self.updateIdx
                case "ones":
                    base = 1

        return self.alpha0 * base

    def _storeWeights(self):
        """
        Utility function for storing the weights to be able to track the
        progress etc.

        The system call to `touch` is used to indicate the file's contents have
        been changed without having to check either of the `.npz` files directly
        which avoid race conditions where the file has already been modified,
        but the writing has not completed. This would then result in an issue
        while reading the file, in turn leading to crashes.
        """
        np.savez(os.path.join(self.basePath, self.WkkName), Wkk=self.Wkk)
        np.savez(os.path.join(self.basePath, self.weightsName), weights=self.weights)
        os.system(f"touch {os.path.join(self.basePath, self.semaphoreName)}")
