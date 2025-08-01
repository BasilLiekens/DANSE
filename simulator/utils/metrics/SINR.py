import numpy as np
from utils import vad


def SINR(
    y: np.ndarray | list[np.ndarray],
    x1: np.ndarray | list[np.ndarray],
    x2: np.ndarray | list[np.ndarray],
) -> float | list[float]:
    """
    Compute the signal-to-interference-and-noise ratio for two sources (one
    target source and one interfering source). The script assumes there is
    access to each source's contribution in the processed output.

    Parameters
    ----------
    -y : [N x 1] np.ndarray[float] or list of this
        Actual processed output signal (`N` is the number of samples).
    -x1 : [N x 1] np.ndarray[float] or list of this
        Processed output attributed to desired signal.
    -x2 : [N x 1] np.ndarray[float] or list of this
        Processed output attributed to noise and interference.

    Returns
    -------
    -SINR :
    """

    if isinstance(y, np.ndarray):
        return _singular_SINR(y, x1, x2)

    elif isinstance(y, list):
        return [_singular_SINR(y[i], x1[i], x2[i]) for i in range(len(y))]

    else:
        raise ValueError("Unsupported type found!")


def vadBasedSINR(
    y: np.ndarray | list[np.ndarray],
    x: np.ndarray | list[np.ndarray],
    vadType: str = "silero",
    fs: int = int(16e3),
) -> float | list[float]:
    """
    Compute the SINR, but use the signal `x` to compute a VAD, which can then be
    used to compute the power of the complete signal and only noise, which can
    then be used to compute the SINR. This function is predominantly made for
    the postprocessing of recordings where the groundtruth contribution is not
    available, which makes this way of computing more reliable.

    Parameters
    ----------
        y: 1D np.ndarray or list thereof
            The complete signals to compute the SINR of

        x: 1D np.ndarray or list thereof
            A signal that can be used to compute a VAD of the desired speaker

        vadType: str
            The type of VAD to use.

    Returns
    -------
        A float or list of floats (depending on the input) with the SINR per
        segment
    """
    if isinstance(y, np.ndarray):
        return _singular_vad_based_SINR(y, x, vadType, fs)

    elif isinstance(y, list):
        return [
            _singular_vad_based_SINR(y[i], x[i], vadType, fs) for i in range(len(y))
        ]

    else:
        raise ValueError("Unsupported type found!")


def _singular_SINR(y: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> float:
    # Sanity check (check whether `y = x1 + x2` based on RMSE of residual)
    if np.abs(np.sqrt(np.sum((y - x1 - x2) ** 2)) / np.sqrt(np.sum(y**2))) > 0.01:
        print("/!\\ Something is wrong, `y` should be the sum of `x1` and `x2`.")
        print("SIR can not be computed -- Returning NaN.")
        sir = np.nan
    else:
        sir = 10 * np.log10(np.var(x1) / np.var(x2))

    return float(sir)


def _singular_vad_based_SINR(
    y: np.ndarray, x: np.ndarray, vadType: str, fs: float
) -> float:
    sigVad = vad.computeVAD(x, fs=fs, type=vadType)

    sigNoise = y[sigVad]
    noiseOnly = y[~sigVad]

    Psn = np.var(sigNoise)
    Pn = np.var(noiseOnly)

    return float(10 * np.log10((Psn - Pn) / Pn))
