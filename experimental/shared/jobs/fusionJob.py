from fission.core.jobs import PythonJob
from fission.core.pipes import BasePipe

from dataclasses import dataclass
from datetime import datetime
import numpy as np
import os
import warnings


@dataclass
class fusionJob(PythonJob):
    """
    This class handles the incoming recordings and then fuses them and forwards
    them to "real DANSE jobs". This way there is no buffering required of the
    audio which would be the case for an implementation with only one class;
    due to the existence of only 1 `run` method, it would be required to wait
    for the next time `run` was called before the handling could be started as
    only then would the next samples be available.

    Another issue that this solves is the fact that the handling would become
    too entangled + there is a certain risk of deadlocks: more specifically the
    startup is quite hard to get properly working: the `run` method would only
    be called in case all inputs are available, but in the beginning there is no
    way to have the inputs already as those would be coming from other DANSE jobs.
    """

    #
    Mk: int = 4
    R: int = 1
    #
    lFFT: int = 1024
    overlap: float = 0.5
    #
    windowType: str = "sqrt hanning"
    fs: int = int(16e3)
    #
    vadType: str = "silero"
    #
    inputs: list[BasePipe] = None
    outputs: list[BasePipe] = None
    #
    basePath: str = os.path.join("/home", "RPi", "installations")
    weightsFile: str = os.path.join("Wkk.npz")
    semaphoreFile: str = "semaphore"
    #
    DEFAULT_NODE: str = None
    DEPENDENCIES: list[str] = None
    #

    def __post_init__(self):
        super().__init__(inputs=self.inputs, outputs=self.outputs)

        # perform some validation
        if (
            self.inputs == None
            or len(self.inputs) != 1
            or self.outputs == None
            or len(self.outputs) == 0
        ):
            raise ValueError("Inputs should be of length 1, outputs of non-zero length")

        # number of outputs
        self.K = len(self.outputs)

        # initialize some objects
        match self.windowType:
            case "sqrt hanning":
                self.window: np.ndarray = np.sqrt(np.hanning(self.lFFT))
            case "ones":
                self.window: np.ndarray = np.sqrt(self.overlap) * np.ones(self.lFFT)
            case _:
                warnings.warn("Window type not recognized, using scaled ones instead.")
                self.window: np.ndarray = (
                    1 / np.sqrt(1 / self.overlap) * np.ones(self.lFFT)
                )
        self.window = self.window.reshape((-1, 1))  # allow for broadcasting

        self.hop = int((1 - self.overlap) * self.lFFT)

        self.inputBuffer: np.ndarray = np.zeros(
            (self.lFFT, self.Mk), dtype=np.float64
        )  # for storing incomplete segments between inputs

        # set the pointer at its default location
        self.insertionIdx: int = self.lFFT - self.hop

        # paths to files
        self.path_to_semaphore: str = os.path.join(self.basePath, self.semaphoreFile)
        self.path_to_weights: str = os.path.join(self.basePath, self.weightsFile)

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

        if os.path.isfile(self.path_to_semaphore) and os.path.isfile(
            self.path_to_weights
        ):
            self.Wkk: np.ndarray = np.load(self.path_to_weights)["Wkk"]
            self.lastRead: float = os.path.getmtime(self.path_to_weights)
        else:
            self.Wkk: np.ndarray = np.random.randn(
                self.lFFT // 2 + 1, self.Mk, self.R
            ).astype(np.complex128)
            self.lastRead: float = datetime.now().timestamp()

        self.Wkk_H = np.conj(self.Wkk.transpose(0, 2, 1))

    def run(self, recording: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """
        Update the buffer, and if it is full, return a new STFT frame. If the
        buffer is not filled, returns an empty frame. Likewise, if the buffer
        is filled more than once, several STFT frames are returned.

        However, due to the implementation of the recordingJob, this shouldn't
        be a problem if the blocksize is chosen in accordance with the STFT
        parameters. It is preferable to do so to avoid getting in trouble later
        downstream.

        A potential remedy would be to only return one frame every time, but
        that could lead to issues with empty buffers etc. nonetheless. It should
        also be noted here that returning `None` is not an option: it would
        close the connection.

        Returns
        -------
            A set of tuples, one for each output pipe. The tuples contain both
            STFT frames used for the fusing of the signal as well as the fused
            signals themselves (bypassing the need for a frame delay due to the
            fusing and subsequent communication before the other ones are ready)

            This also allows to eliminate the cycle in the network graph by
            means of writing the filter weights which have to be communicated
            back to this job to disk. This is an expensive operation, but at
            least to do some implementation.
        """
        # check if a new weight is available and load from disk
        if (
            os.path.exists(self.path_to_semaphore)
            and os.path.getmtime(self.path_to_semaphore) != self.lastRead
        ):
            self.Wkk = np.load(self.path_to_weights)["Wkk"]
            self.Wkk_H = np.conj(self.Wkk.transpose(0, 2, 1))
            self.lastRead = os.path.getmtime(self.path_to_semaphore)

        frames = []
        fused = []
        while self.insertionIdx + recording.shape[0] >= self.lFFT:
            offset = self.lFFT - self.insertionIdx
            self.inputBuffer[self.insertionIdx :, :] = recording[:offset, :]
            recording = recording[offset:, :]

            # window, transform and fuse
            windowed = self.inputBuffer * self.window
            fd = np.fft.rfft(windowed, n=self.lFFT, axis=0).reshape((-1, self.Mk, 1))
            frames.append(fd)
            fused.append(self.Wkk_H @ fd)

            # update the buffer, no need to zero since it is overwritten
            self.inputBuffer = np.roll(self.inputBuffer, axis=0, shift=-self.hop)
            self.insertionIdx = self.lFFT - self.hop

        # insert the remainder of the input
        self.inputBuffer[
            self.insertionIdx : self.insertionIdx + recording.shape[0], :
        ] = recording
        self.insertionIdx += recording.shape[0]

        # return the frames
        if len(frames) > 0:
            frames = np.concatenate(frames, axis=2)
            fused = np.concatenate(fused, axis=2)
        else:
            frames = np.zeros((self.lFFT // 2 + 1, self.Mk, 0), dtype=np.complex128)
            fused = np.zeros((self.lFFT // 2 + 1, self.R, 0), dtype=np.complex128)

        toReturn = [[frames]]
        for _ in range(self.K - 1):
            toReturn.append([fused])

        return tuple(toReturn)
