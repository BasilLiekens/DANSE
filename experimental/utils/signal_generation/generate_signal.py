import numpy as np
import os
import resampy
import soundfile as sf
from .template import constructTemplate


def generateSpeakerSignal(
    length: float,
    speakerFs: float,
    audioBase: str,
    audioSources: list[str],
    templateLength: int,
    calibration: bool = False,
    micFs: float | None = None,
    templateType: str = "SOS",
    nFreqs: int = 10,
):
    """
    Given a set of parameters, generate the signal that each speaker should play.
    This includes the addition of a template in the beginning of one speaker
    signal where the others should remain silent etc.

    Parameters
    ----------
        length:
            The duration of the signal in seconds.

        speakerFs: float
            The frequency at which the speakers will play their audio.

        audioBase: str
            The folder where the audio files can be found.

        audioSources: list[str]
            A list with all audio files to use, the length is assumed to be
            equal to the number of speakers.

        templateLength: int
            The length of the template, in samples.

        calibration: bool, optional
            Whether or not a calibration sequence is used where only the desired
            signal is played. Defaults to `False`.

        micFs: float, optional
            The sampling frequency of the mics, can be different from the one of
            the speakers. If this is `None`, in which case it is assumed to be
            the same as the one of the speakers.

        templateType: str, optional
            The type of template to generate, cfr. the docs of
            `constructTemplate()`

        nFreqs: str, optional
            Number of frequencies in the template in the case where `SOS` is the
            type. Cfr. `constructTemplate()`
    """
    # bookkeeping
    nSpeakers = len(audioSources)
    if micFs == None:
        micFs = speakerFs

    template = constructTemplate(templateLength, micFs, templateType, nFreqs)
    signalsMicFs = np.zeros((nSpeakers, length * micFs))

    # add individual recordings
    for idx, source in enumerate(audioSources):
        # if calibration, only 1 speaker signal desired!
        if (idx == 0) or (idx != 0 and not calibration):
            data, fs = sf.read(os.path.join(audioBase, source), dtype="float64")
            data = data[: int(length * fs)]  # snip off the relevant part
            data /= np.std(data)  # rescale to unit power
            data = resampy.resample(data, sr_orig=fs, sr_new=micFs)

            if idx == 0:  # insert the template
                data = np.concatenate((template, data))
            else:  # insert a set of zeros of equal length to account for the template
                data = np.concatenate((np.zeros((templateLength)), data))

            signalsMicFs[idx, :] = data[: int(length * micFs)]

    # bring back to the speaker sampling frequency
    signals = resampy.resample(signalsMicFs, sr_orig=micFs, sr_new=speakerFs)
    return signals
