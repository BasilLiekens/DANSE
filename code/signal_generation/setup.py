# Contents of package:
# Classes and functions related to the acoustic scenario.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import os
from dataclasses import dataclass, field

import numpy as np
import yaml


@dataclass
class Parameters:
    rd: list[float] = field(
        default_factory=lambda: [5.0, 5.0, 3.0]
    )  # room dimensions (m) (can be 2-D or 3-D)
    temperature: float = 20.0  # temperature (°C)
    humidity: float = 50.0  # relative humidity (%)
    t60: float = 0.0  # reverberation time (s)
    #
    fs: int = 16000  # sampling frequency [Hz]
    RIR_length: int = 8000  # length of the RIRs [samples]
    #
    SRO: list[int] = field(
        default_factory=lambda: [0]
    )  # list with SRO's per node, either length = 1 or = K
    #
    audio_base: str = os.path.join(
        "path", "to", "audio-files"
    )  # path to recordings that can be used for generating signals
    noise_base: str = os.path.join(
        "path", "to", "noise-files"
    )  # path to recordings that can be used for generating signals
    recording_base: str = os.path.join("path", "to", "RPi", "recordings")
    #
    K: int = 10  # number of nodes
    Mk: int = 2  # number of microphones per node (all nodes have the same Mk)
    R: int = 1  # number of channels to communicate
    node_diameter: float = (
        0.2  # node diameter (m) (where the microphones are placed around the node's center) (all nodes have the same diameter)
    )
    min_inter_sensor_d: float = 0.05  # minimal inter-sensor distance within a node (m)
    #
    Ns: int = 1  # number of desired sources
    Nn: int = 1  # number of noise sources
    min_d_sources: float = 0.5  # minimum distance between sources (m)
    min_d_sources_wall: float = (
        0.5  # minimum distance between each source and walls (m)
    )
    min_d_nodes_wall: float = 0.5  # minimum distance between each node and walls (m)
    #
    lFFT: int = 1024  # number of points in the STFT
    window: str = "sqrt hanning"  # the default window to use in the STFT
    overlap: float = 0.5  # overlap between subsequent frames in the STFT
    #
    alphaFormat: str = "harmonic"  # the alphas to use in the case of synchronous
    alpha0: int = 1  # the scaling of the alphas
    #
    deltaUpdate: int = int(1e2)  # number of samples before an update happens
    lmbd: float = 0.999  # exponential smoothing parameter for online updating
    #
    useVAD: bool = False  # whether or not to use a VAD
    vadType: str = "energy"  # which type of VAD to use
    #
    GEVD: bool = False  # whether or not to use the GEVD-based version of DANSE
    sequential: bool = True  # Use the sequential or synchronous version of DANSE
    updateMode: str = (
        "exponential"  # mode to update the correlation matrices (only applicable to online networks)
    )
    Gamma: float = 0  # regularization constant for the computation of Wiener solution
    mu: float = 1  # mu constant for the speech-distortion weighting of the regular MWF
    #
    signal_length: float = 5.0  # the length of the generated signals (s)
    recorded_signals: bool = (
        False  # whether or not to use recordings for generating the audio signal
    )
    recorded_noise: bool = False  # idem as above, but now for the noise signals
    #
    SIR: float = 0  # Signal-to-Interference ratio (measured w.r.t the first node)
    measurement_noise: bool = False  # whether or not to add measurement noise
    measurement_SNR: float = (
        10.0  # The SNR of the measurement noise with regards to the incoming signal.
    )
    #
    include_silences: bool = (
        False  # include silences in the source signal (to allow proper estimates)
    )
    silence_period: int = 1000  # the number of samples in one period (on and off)
    silence_duty_cycle: float = 0.5  # time that the source signal is not forced to 0.
    #
    seed: int = 42  # random generator seed
    #
    audio_sources: list[str] = field(
        default_factory=lambda: []
    )  # the signal to use for the audio sources, length == `Ns`
    noise_sources: list[str] = field(
        default_factory=lambda: []
    )  # the signal to use for the noise sources, length == `Nn`
    recording_session: str = (
        "Recording name"  # which folder to look for in the recordings
    )
    #

    def __post_init__(self):
        """Update derived attributes."""
        np.random.seed(self.seed)  # set random seed
        self.rd = np.array(self.rd)  # convert to numpy array for easier manipulation
        self.dim = len(self.rd)  # room dimensionality

        # expand SRO to have the correct length
        match len(self.SRO):
            case 1:
                self.SRO = self.SRO * self.K
            case self.K:
                pass  # nothing should be done
            case _:
                raise ValueError(
                    f"Expected SRO to be of length 1 or {self.K}, got {len(self.SRO)}"
                )

        if not np.any(np.array(self.SRO) == 0):
            raise ValueError(f"At least one element of SRO should be 0, got {self.SRO}")

    def load_from_yaml(self, path: str):
        """Load parameters from a YAML file."""
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        for key, value in data.items():
            setattr(self, key, value)
        if "SRO" not in data.keys():
            self.SRO = [0]  # reset the SRO to its former state if loading from yaml
        self.__post_init__()  # update derived attributes
        return self

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        # Print all attributes, each on a separate line
        return "Parameters:\n  >> " + "\n  >> ".join(
            [f"{key}: {value}" for key, value in self.__dict__.items()]
        )
