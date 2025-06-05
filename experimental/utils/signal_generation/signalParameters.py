from dataclasses import dataclass, field
import numpy as np
import os
from typing_extensions import Self
import yaml


@dataclass
class signalParameters:
    """
    Class that allows to read a given `.yml` file and treat all values in there
    as python objects for conveniece.
    """

    # Basic config for signal generation
    signalLength: float = 10  # length of the recordings [s]
    speakerFs: int = int(48e3)  # sampling frequency of the speakers
    micFs: int | None = int(
        16e3
    )  # sampling frequency of the mics, if None, same as speakers
    #
    audioBase: str = "/path/to/audio/files"  # basepath for the audio
    speechPrefix: str = (
        "/suffix/after/audioBase/to/desired/signals"  # folder where speech is located
    )
    interfererPrefix: str = (
        "/suffix/after/audioBase/to/interfering/signals"  # folder for noise/interference
    )
    #
    desiredSpeakers: list[str] = field(
        default_factory=lambda: ["list", "of", "desired signals"]
    )  # the signals to use for the speaker
    interferingSpeakers: list[str] = field(
        default_factory=lambda: ["list", "of", "interfering signals"]
    )  # interfering sources
    #
    templateLength: int = 1024  # number of template samples on receiving side [samples]
    templateType: str = "MLS"  # type of template to use
    nFreqs: int = 10  # number of frequencies in the template when type = SOS
    blockSize: int = 512
    #

    # output config
    output_device: str = "name of output_device"
    #
    sources: list[str] = field(default_factory=lambda: ["list", "of", "speakers"])
    output_channels: list[int] = field(
        default_factory=lambda: ["mappings", "to", "output", "channels", "sounddevice"]
    )
    #

    # mic array config
    speakerID: str = "partial_name_of_speaker"  # partial ID of the speaker
    desChannels: list[int] = field(default_factory=lambda: 1 + np.arange(4))

    # how to save the received files
    baseDir: str = "/path/to/save/folder"
    fileName: str = "recording_dry"
    fileType: str = ".wav"

    # frequency domain processing processing
    lFFT: int = 1024  # nb samples in fft
    overlap: float = 0.5  # the overlap between subsequent frames
    R: int = 1
    #
    windowType: str = "sqrt hanning"  # type of window for STFT
    vadType: str = "silero"  # what type of vad to use
    #
    deltaUpdate: int = 100  # number of frames before an update can take place
    lmbd: float = 0.99  # exponential smoothing parameter
    #
    GEVD: bool = False  # GEVD-based updating rule?
    Gamma: float = 0  # L2 regularization factor?
    mu: float = 1  # mu parameter for the SDW-MWF, if 1 a regular is used
    #
    sequential: bool = True
    alpha0: float = 1.0
    alphaFormat: str = "harmonic"
    #
    path_to_calibration: str = "/home/RPi/installations/recording_dry.wav"
    baseDir_pi: str = "/home/RPi/installations"
    #
    seed: int = 64
    #

    def load_from_yaml(self, path_to_cfg: str) -> Self:
        """Load parameters from config file"""
        with open(path_to_cfg, "r") as file:
            data = yaml.safe_load(file)
        for key, value in data.items():
            setattr(self, key, value)
        self.__post_init__()
        return self

    def __post_init__(self):
        """Update derived attributes"""
        self.audioSources: list[str] = []

        for source in self.desiredSpeakers:
            self.audioSources.append(os.path.join(self.speechPrefix, source))
        for source in self.interferingSpeakers:
            self.audioSources.append(os.path.join(self.interfererPrefix, source))

        self.nChannels = len(self.desChannels)
