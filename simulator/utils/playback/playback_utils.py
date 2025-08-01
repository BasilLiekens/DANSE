import numpy as np
import os
import soundfile as sf


def writeSoundFile(signal: np.ndarray, fs: int, title: str = "audiofile"):
    """
    Write the signal contained in "signal" to a .wav file such that it can be
    listened to for further reference. This is a replacement for the
    "listen_to_array" function as it caused crashes.

    Parameters
    ------------
        signal: A 1D numpy array containing the audio data to write.
        fs:     The sampling rate of the audio/file
        title:  Title to give the file.
    """
    if signal.squeeze().ndim > 1:
        raise ValueError(
            f"Expected signal to only have 1 dimension, got {signal.squeeze().ndim} instead"
        )

    tmp = signal.squeeze()
    # normalize data
    tmp = 1 / np.max(np.abs(tmp)) * tmp
    sf.write(os.path.join(os.getcwd(), "output", "audio", title + ".wav"), tmp, fs)
