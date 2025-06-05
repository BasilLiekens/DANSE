import numpy as np
from scipy import signal


def constructTemplate(
    templateLen: int, fs: float, type: str = "SOS", nFreqs: int = 10
) -> np.ndarray:
    """
    Construct the template that can be used to determine the start of the
    actual signal, used for signal alignment between different recordings.

    Parameters:
    -----------
        templateLen: int
            the length of the template in samples

        fs: int
            the sampling frequency of the system [Hz]

        type: str, optional
            the type of template to make, two options: sum-of-sinusoids (SOS),
            or maximum-length sequence (MLS). Defaults to SOS.

        nFreqs: int, optional
            the number of frequencies to use in the sum of sinusoids case,
            defaults to 10.
    """
    match type:
        case "SOS":  # sum of sinusoids
            freqs = np.linspace(0, fs // 2, nFreqs).reshape((-1, 1))
            samples = np.arange(templateLen).reshape((1, -1))
            template = np.sum(np.sin(2 * np.pi * freqs / fs * samples), axis=0)
        case "MLS":  # cfr. https://en.wikipedia.org/wiki/Maximum_length_sequence
            m = int(np.log2(templateLen))
            template, _ = signal.max_len_seq(m, length=templateLen)
            template = template.astype(np.float64)
        case _:
            raise ValueError(f"Unknown option {type}!")

    return template
