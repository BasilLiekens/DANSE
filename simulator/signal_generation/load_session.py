import numpy as np
import os
import resampy
from .setup import Parameters
import silero_vad as silero  # for cleaning up the signal
import soundfile as sf


def load_session(p: Parameters) -> tuple[np.ndarray, np.ndarray]:
    """
    Equivalent of `generate_micsigs`, but now uses real recordings instead of a
    room. It is expected that there are two types of filenames available in the
    targeted folder: `recording_dry_X.wav` and `recording_wet_X.wav`. The former
    should be a recording that only contains the desired signal (+ measurement
    noise) while the other one contains all sources.

    Despite the fact that the dry signals still contain some measurement noise,
    they are not cleaned: using silero to do this will result in a different vad
    the second time.

    This way of operating is also representative of how the processing would
    happen in practice.

    Parameters
    ----------
        p: Parameters
            The parameters that were loaded from yaml: provides quite a lot of
            info that is not needed as well.

    Returns
    -------
        A tuple with 2 np.ndarrays, both of shape [`nb Mics` x `signal_length`].
        The first contains the (cleaned up) contribution of the desired signal,
        while the latter contains the full recording per node.
    """
    folderName = os.path.join(p.recording_base, p.recording_session)

    dry_signature = os.path.join(folderName, "recording_dry_{}.wav")
    wet_signature = os.path.join(folderName, "recording_wet_{}.wav")

    dry_signals = np.zeros((p.K * p.Mk, int(p.signal_length * p.fs)))
    wet_signals = np.zeros_like(dry_signals)

    # load the signals
    for k in range(p.K):
        dry_data, dry_fs = sf.read(dry_signature.format(k + 1))
        wet_data, wet_fs = sf.read(wet_signature.format(k + 1))

        if dry_data.shape[1] != p.Mk or wet_data.shape[1] != p.Mk:
            raise ValueError(
                f"Expected all recordings to have {p.Mk} channels, got "
                f"{dry_data.shape[1]} for dry and {wet_data.shape[1]} for wet."
            )

        if dry_fs != p.fs:
            dry_data = resampy.resample(dry_data, sr_orig=dry_fs, sr_new=p.fs, axis=0)

        if wet_fs != p.fs:
            wet_data = resampy.resample(wet_data, sr_orig=wet_fs, sr_new=p.fs, axis=0)

        dry_signals[k * p.Mk : (k + 1) * p.Mk, :] = dry_data[
            : int(p.signal_length * p.fs), :
        ].T
        wet_signals[k * p.Mk : (k + 1) * p.Mk, :] = wet_data[
            : int(p.signal_length * p.fs), :
        ].T

    return dry_signals, wet_signals


def getStartSample(sig: np.ndarray, fs: int = int(16e3)) -> int:
    """
    Given a signal `sig`, snip off the first part of a recording (let the
    recording start at the first detected speech segment).

    Parameters
    ----------
        sig: np.ndarray, 1D
            The signal to snip off a part

        fs: int
            The sampling frequence of the signal, should be 16 or 8 kHz to
            function with `Silero`

    Returns
    -------
        The index of the first speech segment as detected by `Silero`
    """
    model = silero.load_silero_vad()
    speech_timestamps = silero.get_speech_timestamps(
        sig, model, sampling_rate=fs, return_seconds=False
    )
    return speech_timestamps[0]["start"]
