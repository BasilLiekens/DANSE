import matplotlib.pyplot as plt
import numpy as np


def micPlotter(
    audio: np.ndarray, noise: np.ndarray, title: str = "Microphone signal"
) -> plt.Figure:
    """
    Given the audio and noise contributions of a signal, plot each of them
    individually as well as the total signal.

    Parameters:
    -----------
        audio:  the desired contribution to the signal
        noise:  the undesired contribution to the signal
    """
    audio = np.squeeze(audio)
    noise = np.squeeze(noise)

    if audio.ndim != 1 or noise.ndim != 1:
        raise ValueError("Both inputs should have only 1 dimension")
    if audio.size != noise.size:
        raise ValueError("Both inputs should have the same length")

    fig, ax = plt.subplots()
    fig.set_size_inches(8.5, 5.5)
    ax.plot(audio + noise, label="Total signal")
    ax.plot(audio, label="Contribution of desired signal")
    ax.plot(noise, label="Noise contribution")
    ax.grid()
    ax.legend()
    ax.set_xlabel("samples")
    ax.set_ylabel("amplitude")
    ax.set_title(title)
    ax.autoscale(tight=True)

    return fig


def STFTPlotter(
    STFT: np.ndarray, title: str = "STFT of microphone signal (single-sided)"
) -> plt.Figure:
    """
    Given 1 STFT (of only 1 signal, hence should be a 2D array), plot the
    spectrogram (in dB).

    Parameters
    ----------
        STFT:   A 2D numpy array containing the STFT with dimensions
                ["lFFT" x "nFrames"].
        title:  The title of the plot.

    Returns
    ----------
        The created spectrogram
    """
    # bookkeeping
    if STFT.ndim != 2:
        raise ValueError(f"STFT should be a 2D array, got {STFT.ndim} instead.")

    # convert to dB
    STFTdB = 20 * np.log10(np.maximum(np.abs(STFT), 1e-10))  # ensure there is a plot
    fig, ax = plt.subplots()
    fig.set_size_inches(8.5, 5.5)
    cmesh = ax.pcolormesh(STFTdB, cmap="plasma", vmin=-70, vmax=30)
    ax.set_xlabel("frames")
    ax.set_ylabel("frequency bin")
    ax.set_title(title)
    cbar = fig.colorbar(cmesh)
    cbar.set_label("magnitude [dB]")
    return fig
