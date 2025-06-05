import numpy as np
import pystoi


def computeSTOI(
    x: np.ndarray, y: np.ndarray | list[np.ndarray], fs: int
) -> float | list[float]:
    """
    Compute the STOI value given a clean signal `x` and either a single
    processed signal or multiple ones in a list `y`. Depending on the format of
    `y` either a float or a set of floats is returned.

    The input should be a 2D array with dimension [`nChannels` x `nSamples`],
    for now it is only tested with `nChannels` == 1!
    """

    def _computeSingularSTOI(x: np.ndarray, y: np.ndarray, fs: int) -> float:
        xPrime = np.squeeze(x)
        yPrime = np.squeeze(y)
        if xPrime.ndim != 1 or x.shape != y.shape:
            raise ValueError(
                "x and y should have the same shapes and be squeezable into 1D!"
            )
        return float(np.minimum(pystoi.stoi(xPrime, yPrime, fs), 1))

    if isinstance(y, np.ndarray):
        return _computeSingularSTOI(x.T, y.T, fs)

    elif isinstance(y, list):
        return [_computeSingularSTOI(x.T, y[i].T, fs) for i in range(len(y))]

    else:
        raise ValueError(f"Unsupported type found!")
