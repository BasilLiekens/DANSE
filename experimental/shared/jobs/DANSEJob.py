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
class DANSEJob(PythonJob):
    """
    Class that performs the DANSE algorithm. Math & implementation is based on
    that one from the simulator. Actually this is just a wrapper around the
    `node` class that ensures the delegation happens properly.

    The same principles as in `MWFJob` apply: a first trial recording is
    expected to be able to compute the VAD. Additionally, again a variable
    amount of inputs is allowed.

    It should also be noted here that a frame of delay is introduced: MARVELO
    only allows for one `run` function, and there are two steps to be completed:
    first of all there is a necessity to perform the signal fusion, which is
    then followed by updating the correlation matrices & performing some signal
    estimation. However, to this end, the fused signals should be available as
    well. Therefore, it is easiest to compute the fused signal in one iteration,
    communicate that one, and in the next timestep perform the estimations etc.

    This approach introduces an extra frame of delay, which might be harmful for
    echo etc.
    """

    #
    Mk: int = 4  # nb channels per node
    R: int = 1  # number of outputs + nb channels to communicate
    #
    lFFT: int = 1024
    overlap: float = 0.5
    #
    windowType: str = "sqrt hanning"
    fs: float = int(16e3)
    #
    vadType: str = "silero"
    #
    deltaUpdate: int = 100
    lmbd: float = 0.99
    #
    GEVD: bool = True
    Gamma: float = 0
    mu: float = 1
    #
    sequential: bool = True
    nodeNb: int = 0  # sequential updating => offset to remove need for orchestrator
    #
    alpha0: float = 1  # intial scaling for the alpha parameter
    alphaFormat: str = "harmonic"
    #
    seed: int = 64
    #
    inputs: list[BasePipe] = None
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
            raise ValueError("Inputs and outputs should be lists of non-zero length")
        self.K = len(self.inputs)  # includes this node!

        # validate some values
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
        if self.nodeNb < 0 or len(self.inputs) < self.nodeNb:
            raise ValueError(
                f"Node number should be non-negative and smaller than the total number of nodes, got {self.nodeNb}."
            )
        if self.alpha0 <= 0:
            raise ValueError(f"Scaling for the alphas should be non-negative!")

        # initialize some objects, window & STFTObj => dual window
        match self.windowType:
            case "sqrt hanning":
                window: np.ndarray = np.sqrt(np.hanning(self.lFFT))
            case "ones":
                window: np.ndarray = np.sqrt(1 - self.overlap) * np.ones(self.lFFT)
            case _:
                warnings.warn("Window type not recognized, using scaled ones instead.")
                window: np.ndarray = np.sqrt(1 - self.overlap) * np.ones(self.lFFT)

        self.hop = int((1 - self.overlap) * self.lFFT)
        STFTObj: signal.ShortTimeFFT = signal.ShortTimeFFT(
            window,
            hop=self.hop,
            fs=self.fs,
            fft_mode="onesided",
            mfft=self.lFFT,
        )
        self.dual_win: np.ndarray = STFTObj.dual_win.reshape(
            (-1, 1)
        )  # reshape to allow for broadcasting

        # node that does all the processing
        self.node: utils.DANSE_base.DANSENode = utils.DANSE_base.DANSENode(
            self.K,
            self.Mk,
            self.R,
            self.lFFT,
            self.overlap,
            self.windowType,
            self.GEVD,
            self.Gamma,
            self.mu,
            self.sequential,
            self.nodeNb,
            self.alpha0,
            self.alphaFormat,
            self.seed,
            self.deltaUpdate,
            self.lmbd,
            self.basePath,
        )

        # only an output buffer is required. The input is provided as fd already
        self.outputBuffer: np.ndarray = np.zeros((self.lFFT, self.R), dtype=np.float64)

    def setup(self, *args, **kwargs):
        """
        Some further setup: obtain the vad and the alpha array
        """
        super().setup(*args, **kwargs)
        self.node._storeWeights()

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
        self.frameIdx: int = 0

    def run(self, *args: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Given all frames from your own recording alongside the fused signals
        from other nodes, perform some steps of the DANSE algorithms

        Parameters
        ----------
            args: np.ndarray
                A set of ndarrays, the first one being your own sensor signals,
                the later ones being the fused signals from other nodes. Both
                should be of shape [`lFFT // 2 + 1` x `nChannels` x `nFrames`]
                where `nChannels` = `Mk` for recordings and `R` for fused signals

        Returns
        -------
            An np.ndarray with the time-domain output
        """
        # each input argument comes with an extra dimension => first remove that one
        unwrapped = []
        for arg in args:
            unwrapped.append(arg[0])

        ytilde = np.concatenate(unwrapped, axis=1)
        vad = self.vad[self.frameIdx : self.frameIdx + ytilde.shape[2]]
        self.frameIdx += ytilde.shape[2]

        out_fd = self.node.step(ytilde, vad)

        output = []
        for i in range(out_fd.shape[2]):
            seg_td = np.fft.irfft(out_fd[:, :, i], axis=0) * self.dual_win
            self.outputBuffer += seg_td

            # copy is needed as the data is overwritten later on.
            self.outputBuffer = np.roll(self.outputBuffer, shift=-self.hop, axis=0)
            output.append(np.copy(self.outputBuffer[-self.hop :, :]))
            self.outputBuffer[-self.hop :, :] = 0

        if len(output) > 0:
            toReturn = np.concatenate(output, axis=0)
        else:
            toReturn = np.zeros((0, self.R))

        return tuple([toReturn])

    def __eq__(self, other: object):
        return self is other
