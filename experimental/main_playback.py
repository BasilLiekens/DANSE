"""
File used to start playback.
"""

import numpy as np
import sounddevice as sd
from utils import signal_generation as siggen


class AudioMeasurementSession:
    """
    Handles audio measurement sessions including device setup, calibration,
    and recording of sweep and audio measurements.
    """

    def __init__(self, config_path: str, calibration: bool = False):
        # Load configuration
        p = siggen.signalParameters().load_from_yaml(config_path)

        # Save the I/O device IDs
        for device in sd.query_devices():
            if p.output_device in device["name"]:
                self.output_device_id = device["index"]

        # Class variable for blocksize
        self.blocksize = p.blockSize  # Get blocksize from yaml file

        # Set input and output devices
        self.speakerFs: float = p.speakerFs
        self.setup_devices(p)

        # generate the signals to be played
        self.signal: np.ndarray = siggen.generateSpeakerSignal(
            p.signalLength,
            p.speakerFs,
            p.audioBase,
            p.audioSources,
            p.templateLength,
            calibration,
            p.micFs,
            p.templateType,
            p.nFreqs,
        )

    def setup_devices(self, p: siggen.signalParameters):
        """
        Verify and set up input/output devices and their channels
        """
        # Output device (speakers) - Using output device from the yaml file
        output_device_info = sd.query_devices(self.output_device_id, kind="output")
        print("Output Device Info:")
        print(f"Name: {output_device_info['name']}")
        print(f"Max Output Channels: {output_device_info['max_output_channels']}")

        # Configuration for channel mapping
        self.device = self.output_device_id
        self.output_channels = p.output_channels

        # Validate channel configurations
        if len(self.output_channels) > output_device_info["max_output_channels"]:
            raise ValueError("Requested output channels exceed device capabilities")

    def perform_audio_playback(self):
        """
        Record measurement for any audio file (speech, noise, music) for a specific configuration
        """
        output_mapping = self.output_channels

        # Load the pre-normalized audio signal and apply source-specific gain
        source_gain_db = -60  # default to -60 if not calibrated
        source_gain_linear = 10 ** (source_gain_db / 20)
        output_signal = (
            self.signal * source_gain_linear
        ).T  # reshape into column-wise data

        # start the playback
        print("Starting playback!")
        sd.play(
            data=output_signal,
            samplerate=self.speakerFs,
            mapping=output_mapping,
            device=self.device,
            blocksize=self.blocksize,
        )
        sd.wait()
        print("Playback done!")


if __name__ == "__main__":
    PATH_TO_CFG = "config/cfg.yml"
    calibration = False

    session = AudioMeasurementSession(PATH_TO_CFG, calibration)
    session.perform_audio_playback()
