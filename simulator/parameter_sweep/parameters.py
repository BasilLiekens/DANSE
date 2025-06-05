from dataclasses import dataclass
import numpy as np
from signal_generation.setup import Parameters


@dataclass
class sweepParameters:
    """
    Class that holds all parameters that can be swept over during the parameter
    sweeps done in "main_sweep.py".
    """

    #
    lFFT: int = 1024  # number of points in the fft
    window: str = "sqrt hanning"
    overlap: float = 0.5  # overlap between subsequent frames in the fft [%]
    #
    deltaUpdate: int = 100  # number of correlation matrix updates before DANSE update
    lmbd: float = 0.99  # smoothing factor for correlation matrices
    #
    GEVD: bool = True  # Whether or not to use GEVD-based MWF's
    sequential: bool = True  # Whether or not to do sequential node updating
    #

    def adjustParameters(self, parameters: Parameters) -> Parameters:
        """
        Use the fields in this dataset and adjust the ones in "parameters" to
        obtain an updated set of parameters that can be used for a new sweep.

        **Warning: this adjusts the passed in object's values!**

        Parameters
        ----------
        parameters: Parameters
            The parameters to adjust with this object's fields
        """
        for key, value in self.__dict__.items():
            if isinstance(value, list):
                value = np.array(value)  # convert to np array for easier handling
            parameters.__setattr__(key, value)

        return parameters

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        # Print all attributes, each on a separate line
        return "Parameters:\n  >> " + "\n  >> ".join(
            [f"{key}: {value}" for key, value in self.__dict__.items()]
        )
