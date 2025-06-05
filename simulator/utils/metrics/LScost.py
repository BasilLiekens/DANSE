import numpy as np


def LScost(d: np.ndarray, dTilde: np.ndarray | list[np.ndarray]) -> float | list[float]:
    """
    Compute the least squares (LS) cost

    Parameters
    ----------
    d: np.ndarray, 1D
        The desired signal

    dTilde: np.ndarray or list[np.ndarray], 1D
        The output of a filter with as reference `d`. Should have the same shape.
        Can be either just one np.ndarray or a list of np.ndarrays.

    Returns
    -------
    The least squares cost, either in float format if the passed in object was
    just a single ndarray, or a list of floats.
    """

    def _singular_LS_cost(d: np.ndarray, dTilde: np.ndarray) -> float:
        if d.shape != dTilde.shape:
            raise ValueError(
                f"Expected two inputs of the same shape, received {d.shape} and {dTilde.shape} instead."
            )
        return float(np.mean((d - dTilde) ** 2))

    if isinstance(dTilde, np.ndarray):
        return _singular_LS_cost(d, dTilde)

    elif isinstance(dTilde, list):
        return [_singular_LS_cost(d, dTilde[i]) for i in range(len(dTilde))]

    else:
        raise ValueError("Unsupported type found!")
