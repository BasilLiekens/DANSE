from fission.core.jobs import PythonJob
from fission.core.pipes import BasePipe

from dataclasses import dataclass, field
import numpy as np
import os
import resampy
from scipy import signal
import soundfile as sf
import utils
import warnings


@dataclass
class MWFJob(PythonJob):
    """
    Class that performs an online-mode MWF, can be local or centralized. The
    math is based on that from the simulator.

    For the recording part, a set of `recordingJob`s can be used that pipe their
    outputs into this class, which can then concatenate them and take appropriate
    action.

    It is assumed that a calibration playback sequence was done (without
    interfering speakers) and that the location of that calibration sequence is
    the following: `/home/RPi/installations/recording.wav`, that then gets
    loaded and based on that the VAD can be precomputed.
    """

    #
    nChannels: int = 4  # number of channels per node, assume each node has the same
    R: int = 1  # the number of outputs
    #
    lFFT: int = 1024
    overlap: float = 0.5
    #
    windowType: str = "sqrt hanning"
    fs: float = int(16e3)
    #
    vadType: str = "silero"
    #
    deltaUpdate: int = 100  # nb frames collected before performing an update
    lmbd: float = 0.99  # exponential smoothing factor
    #
    GEVD: bool = True  # use a GEVD-based updating rule?
    Gamma: float = 0  # the L2-regularization factor used in the simulations
    mu: float = 1  # mu parameter in the SDW-MWF, 1 = regular MWF
    #
    inputs: list[BasePipe] = None  # length determines the number of nodes
    outputs: list[BasePipe] = None
    #
    basePath: str = os.path.join("/home", "RPi", "installations")
    DEFAULT_NODE: str = None
    DEPENDENCIES: list[str] = field(default_factory=lambda: [utils])
    #

    def __post_init__(self):
        super().__init__(inputs=self.inputs, outputs=self.outputs)
        if (
            self.inputs == None
            or len(self.inputs) == 0
            or self.outputs == None
            or len(self.outputs) == 0
        ):
            raise ValueError("Inputs and outputs should be lists of non-zero length!")
        self.nNodes = len(self.inputs)

        # validate some values
        if self.nChannels <= 0:
            raise ValueError(f"Nb channels should be positive, got {self.nChannels}!")
        if self.deltaUpdate <= 0:
            raise ValueError(
                f"DeltaUpdate should be strictly positive, got {self.deltaUpdate}!"
            )
        if self.lmbd <= 0 or self.lmbd >= 1:
            raise ValueError(
                f"lmbd should lie between 0 and 1 (exclusively), got {self.lmbd}!"
            )
        if self.Gamma < 0:
            raise ValueError(f"Gamma should be non-negative, got {self.Gamma}!")
        if self.mu < 0:
            raise ValueError(f"mu should be non-negative, got {self.mu}")

        # define the filter updating function
        if self.GEVD:
            self.filterUpdateFct = utils.DANSE_base.MWF.GEVD_MWF_computation
        else:
            self.filterUpdateFct = utils.DANSE_base.MWF.MWF_computation

        # initialize some objects
        match self.windowType:
            case "sqrt hanning":
                self.window: np.ndarray = np.sqrt(np.hanning(self.lFFT))
            case "ones":
                self.window: np.ndarray = (
                    1 / np.sqrt(1 / self.overlap) * np.ones(self.lFFT)
                )
            case _:
                warnings.warn("Window type not recognized, using scaled ones instead.")
                self.window: np.ndarray = (
                    1 / np.sqrt(1 / self.overlap) * np.ones(self.lFFT)
                )

        self.hop = int((1 - self.overlap) * self.lFFT)
        self.STFTObj: signal.ShortTimeFFT = signal.ShortTimeFFT(
            self.window,
            hop=self.hop,
            fs=self.fs,
            fft_mode="onesided",
            mfft=self.lFFT,
        )
        self.inputBuffer: np.ndarray = np.zeros(
            (self.lFFT, self.nChannels * self.nNodes), dtype=np.float64
        )  # for storing incomplete segments between inputs
        self.outputBuffer: np.ndarray = np.zeros((self.lFFT, self.R), dtype=np.float64)
        self.insertionIdx: int = 0  # the position at which to insert data

        # frequency domain autocorrelation matrices
        self.Ryy: np.ndarray = np.zeros(
            (
                self.lFFT // 2 + 1,
                self.nChannels * self.nNodes,
                self.nChannels * self.nNodes,
            ),
            dtype=np.complex128,
        )
        self.Rnn: np.ndarray = np.zeros_like(self.Ryy)

        diagTerms = np.arange(self.nChannels * self.nNodes)
        self.Ryy[:, diagTerms, diagTerms] = 1
        self.Rnn[:, diagTerms, diagTerms] = 1e-1

        # for the computation of the filter weights
        self.e1: np.ndarray = np.vstack(
            (np.eye(self.R), np.zeros((self.nNodes * self.nChannels - self.R, self.R)))
        )

        self.W: np.ndarray = self.filterUpdateFct(
            self.Ryy, self.Rnn, self.e1, self.Gamma, self.mu
        )
        self.W_H: np.ndarray = np.conj(self.W.transpose(0, 2, 1))

        self.weights: list[np.ndarray] = [self.W]  # logger

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

        # only save the weights here, as __init__ happens on device.
        np.savez(os.path.join(self.basePath, "weights.npz"), weights=self.weights)

        # generate the vad
        recordingData, fs = sf.read(os.path.join(self.basePath, "recording_dry.wav"))
        recordingData = recordingData[:, 0]  # only the first channel is of interest
        if fs != self.fs:
            recordingData = resampy.resample(
                recordingData, sr_orig=fs, sr_new=self.fs, axis=0
            )

        self.vad = utils.vad.transformVAD(
            utils.vad.computeVAD(recordingData, self.fs, type=self.vadType),
            self.lFFT,
            self.overlap,
        )
        self.frameIdx: int = 0  # indicate in which frame you are

        self._node

    def run(self, *args: np.ndarray):
        """
        Update the correlation matrices with the incoming data, and if a certain
        number of frames has been observed, update the weights. Also estimate
        the outputs by means of the filters.

        Parameters
        -----------
            args: list[np.ndarray]
                A list with recordings from each node, should be equally long
                (hence use a fixed blocksize across nodes, and enfore that a
                block of that length is always sent)

        Yields
        -------
            The estimated output as a numpy array with number of columns equal
            to `R`
        """
        if len(args) != self.nNodes:
            raise ValueError(
                f"Unexpected number of inputs, expected {self.nNodes}, got {len(args)}!"
            )

        sigs = np.concatenate(args, axis=1)
        to_output = []

        while self.insertionIdx + sigs.shape[0] >= self.lFFT:
            offset = self.lFFT - self.insertionIdx
            self.inputBuffer[self.insertionIdx :, :] = sigs[:offset, :]
            sigs = sigs[offset:, :]  # snip off the used part

            # frame is filled up => window, transform and update
            windowed = self.inputBuffer * self.STFTObj.win.reshape((-1, 1))
            fd = np.fft.rfft(windowed, n=self.lFFT, axis=0).reshape(
                (-1, sigs.shape[1], 1)  # reshape for 3D multiplication
            )

            # update of the correlation matrices
            corrUpdate = fd @ np.conj(fd.transpose(0, 2, 1))  # outer product
            if self.vad[self.frameIdx]:
                self.Ryy = self.lmbd * self.Ryy + (1 - self.lmbd) * corrUpdate
            else:
                self.Rnn = self.lmbd * self.Rnn + (1 - self.lmbd) * corrUpdate
            self.frameIdx += 1

            # filtering for outputs
            out_fd = self.W_H @ fd
            out_td = np.fft.irfft(out_fd, n=self.lFFT, axis=0)[:, :, 0]  # frame idx
            out_td = out_td * self.STFTObj.dual_win.reshape((-1, self.R))

            # check if filter update is applicable
            if self.frameIdx % self.deltaUpdate == 0:
                self.W = self.filterUpdateFct(
                    self.Ryy, self.Rnn, self.e1, self.Gamma, self.mu
                )
                self.W_H = np.conj(self.W.transpose(0, 2, 1))
                self.weights.append(self.W)
                np.savez(
                    "/home/RPi/installations/weights.npz", weights=self.weights
                )  # store weights

            # update buffers and insertionIdx
            self.inputBuffer = np.roll(self.inputBuffer, shift=-self.hop, axis=0)
            self.insertionIdx = self.lFFT - self.hop

            self.outputBuffer += out_td
            to_output.append(np.copy(self.outputBuffer[: self.hop, :]))
            self.outputBuffer = np.roll(self.outputBuffer, shift=-self.hop, axis=0)
            self.outputBuffer[-self.hop :, :] = 0  # reset buffer

        # fill last part
        if self.insertionIdx + sigs.shape[0] < self.lFFT:
            self.inputBuffer[
                self.insertionIdx : self.insertionIdx + sigs.shape[0], :
            ] = sigs
            self.insertionIdx += sigs.shape[0]

        # return the relevant output
        if len(to_output) != 0:
            return len(self.outputs) * tuple([np.concatenate(to_output, axis=0)])
        else:  # be sure to always output something, None will close the channel
            return len(self.outputs) * tuple([np.zeros((0, self.R))])

    def __eq__(self, other: object):
        return self is other  # two objects are only equal if they are the same object
