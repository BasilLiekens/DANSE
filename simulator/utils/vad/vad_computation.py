import numpy as np
import silero_vad as silero


def computeVAD(sig: np.ndarray, fs: float = None, type: str = "energy") -> np.ndarray:
    """
    Wrapper function for different methods of computing the vad of a signal "sig"

    Parameters
    -------------
        sig: 1D numpy ndarray
            the signal to compute a vad on. Should be a 1D signal, hence the user is
            responsible for selecting the channel based on which the vad should be
            computed. Also, it is up to the user whether or not to use the clean
            audio or the "contaminated" one.

        fs: float, optional
            sampling frequency of the recordings, is needed for silero. If the
            default (`None`) is passed in, the default of silero is used, which
            is 16 kHz.

        type: str, optional
            string indicating what type of algorithm is used to determine the vad.
            Currently only "energy" and "silero" are supported. Defaults to "energy"

    Returns
    ---------------
        A 1D numpy array of the same shape as "sig" with boolean entries
        indicating whether or not the speech is active in that frame.
    """
    if sig.ndim != 1:
        raise ValueError("Signal should be 1D.")

    match type:
        case "energy":
            return _energyBasedVAD(sig)
        case "silero":
            return _sileroVAD(sig, fs)
        case _:
            raise ValueError("VAD type not recognized! Impossible to continue.")


def transformVAD(vad: np.ndarray, lFFT: int, overlap: float):
    """
    Transform a computed time domain vad into one that can be used for frequency
    domain processing. Purely a heuristic version that counts whether or not the
    active speech segments dominate.
    """
    hop = int((1 - overlap) * lFFT)
    # "+1" accounts for the fact that the first slice is started at 0:hop
    nSegments = int(np.ceil(vad.shape[0] / hop)) + 1

    transformed = np.zeros((nSegments), dtype=bool)
    for i in range(nSegments):
        endIdx = (i + 1) * hop
        # use a max to ensure the index starts at 0. "overflowing" to the right
        # does not matter as much
        seg = vad[np.maximum(endIdx - lFFT, 0) : endIdx]
        transformed[i] = np.sum(seg) >= np.sum(~seg)

    return transformed


def _energyBasedVAD(audio: np.ndarray) -> np.ndarray:
    """
    Compute the VAD based on the energy of the signal (not really reliable).

    To work even half decent, this requires the groundtruth audio signal
    """
    vadThreshold = 1e-3 * np.std(audio)
    return np.abs(audio) > vadThreshold


def _sileroVAD(audio: np.ndarray, fs: float = None) -> np.ndarray:
    model = silero.load_silero_vad()
    if fs != None:
        speech_timestamps = silero.get_speech_timestamps(
            audio, model, return_seconds=False, sampling_rate=fs
        )
    else:
        speech_timestamps = silero.get_speech_timestamps(
            audio, model, return_seconds=False
        )

    vad = np.zeros((audio.shape[0]), dtype=np.bool)
    for segment in speech_timestamps:
        vad[segment["start"] : segment["end"]] = True
    return vad
