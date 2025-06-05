from fission.core.jobs import PythonJob, LocalJob
from fission.core.pipes import BasePipe

from dataclasses import dataclass
import numpy as np
import os
import soundfile as sf


@dataclass
class collectorJob(LocalJob, PythonJob):
    #
    fs: float = int(16e3)
    recLen: float = 10.0  # [s]
    nChannels: int = 4
    #
    basePath: str = os.path.join(
        "/home",
        "basil-liekens",
        "Msc-Thesis-Danse",
        "code",
        "rpi",
        "experimental-validation",
    )
    fileName: str = "recording.wav"
    #
    inputs: list[BasePipe] = None
    #
    DEFAULT_NODE: str = None
    DEPENDENCIES: list[str] = None
    #

    def __post_init__(self):
        super().__init__(inputs=self.inputs)
        self.audioBuffer: np.ndarray = np.zeros(
            (int(self.fs * self.recLen), self.nChannels)
        )
        self.insertionIdx: int = 0

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

    def run(self, *args: np.ndarray):  # collector jobs, no outputs
        input1 = args[0]
        segLength = input1.shape[0]
        self.audioBuffer[self.insertionIdx : self.insertionIdx + segLength, :] = input1
        self.insertionIdx += segLength
        sf.write(
            os.path.join(self.basePath, self.fileName),
            data=self.audioBuffer,  # don't normalize between dry and wet recordings!
            samplerate=self.fs,
        )

    def __eq__(self, other: object):
        return self is other
