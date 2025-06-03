import numpy as np


def MSE_w(
    W_centr: np.ndarray, W_nw: np.ndarray | list[np.ndarray]
) -> float | list[float]:
    """
    Given the centralized MWF `W_centr` and (a set of) other filters `W_nw`,
    return the difference between the two.
    """

    def _singular_MSE_w(W_centr: np.ndarray, W_nw: np.ndarray) -> float:
        return float(np.mean(np.abs(W_centr - W_nw) ** 2))

    if isinstance(W_nw, np.ndarray):
        return _singular_MSE_w(W_centr, W_nw)

    elif isinstance(W_nw, list):
        return [_singular_MSE_w(W_centr, W_nw[i]) for i in range(len(W_nw))]

    else:
        raise ValueError("Unsupported type found!")
