import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal


def MSEwPlotter(
    W_MWF: np.ndarray,
    W_DANSE: list[np.ndarray],
    label: str = "Filter",
    marker: str = "o",
    fig: plt.Figure = None,
) -> plt.Figure:
    """
    Plot the evolution of the difference between DANSE and centralized MWF
    weights.

    Parameters
    -------------
        W_MWF:      The obtained centralized MWF weights, should be of shape
                    ["lFFT" x "nb Channels per node" "nb nodes" x "nb outputs"]

        W_DANSE:    A list of numpy ndarrays that depict the evolution of the
                    weights over time. Each individual array should have the
                    same dimensions as the W_MWF

        label:      The label to give to the (added) plot.

        marker:     The marker for this specific line

        fig:        A matplotlib object should it be desired to augment an
                    existing figure, then the new data is just plotted on top of
                    the old one to be able to recursively build up a figure. If
                    a fresh new figure is desired, just None can be passed (or
                    no argument as the default is None)

    Returns
    -------------
        The (augmented) figure.
    """
    if fig == None:
        fig, ax = plt.subplots()
        fig.set_size_inches(8.5, 5.5)
    else:
        ax = fig.gca()

    errors = [np.mean(np.abs((W_MWF - W_DANSE[x])) ** 2) for x in range(len(W_DANSE))]

    ax.semilogy(
        1 + np.arange(len(errors)),
        errors,
        "-" + marker,
        fillstyle="none",
        linewidth=2,
        markevery=0.1,
        label=label,
    )
    ax.grid(True, which="both")
    ax.autoscale(tight=True)
    ax.set(
        xlabel="Iteration",
        ylabel=r"$\mathbb{E}\{\|\mathbf{W}^{centr} - \mathbf{W}^{NW}\|_2^2\}$",
        title=r"Evolution of $MSE_W$ over iterations",
    )
    ax.legend()
    fig.tight_layout()
    return fig


def LScostPlotter(
    W_MWF_central: dict[str, np.ndarray],
    W_DANSE: dict[str, list[np.ndarray]],
    desired: np.ndarray,
    input: np.ndarray,
    STFTObj: signal.ShortTimeFFT,
    colors: dict[str, str],
    markers: dict[str, str],
    W_MWF_online: dict[str, list[np.ndarray]] | None = None,
    W_MWF_local: dict[str, np.ndarray] | None = None,
    fig: plt.Figure = None,
):
    """
    Plot the evolution of the difference between DANSE and centralized MWF
    outputs.

    Parameters
    -------------
        W_MWF_central: dict[str, np.ndarray]
            The obtained centralized MWF weights, should be of shape
            ["lFFT" x "nb Channels per node" "nb nodes" x "nb outputs"]. The
            keys are the labels that will be used.

        W_MWF_local: dict[str, np.ndarray]
            Obtained local MWF weights.
            Shape ["lFFT" x "nb Channels per node" x "nb Outputs"], it is assumed
            these filters act on the first channels of the signal.

        W_DANSE: dict[str, list[np.ndarray]]
            A list of numpy ndarrays that depict the evolution of the weights
            over time. Each individual array should have the same dimensions as
            the W_MWF. Keys of the dict are the labels that will be used.

        desired: np.ndarray, 2D
            The time domain desired signal (should be of dimensions
            ["nb Channels" x "duration"]).

        input: np.ndarray, 2D
            Input to the DANSE algorithm: should have dimensions:
            ["nb channels per nodes" "nb nodes" x "duration"]

        STFTObj: signal.ShortTimeFFT
            The STFT object to be used for (I)STFT's

        colors: dict[str, str]
            The colors to be used for specific algorithms

        markers: dict[str, str]
            The markers to be used for specific algorithms

        W_MWF_online: dict[str, list[np.ndarray]]
            Same story as `W_DANSE`, but now for the centralized solutions that
            are built up in an online fashion.

        fig: plt.Figure, optional
            A matplotlib object should it be desired to augment an existing
            figure, then the new data is just plotted on top of the old one to
            be able to recursively build up a figure. If a fresh new figure is
            desired, just None can be passed (or no argument as the default is
            None)

    Returns
    -------------
        The (augmented) figure.
    """
    if fig == None:
        fig, ax = plt.subplots()
        fig.set_size_inches(8.5, 5.5)
    else:
        ax = fig.gca()

    centralizedKeys = list(W_MWF_central.keys())
    DANSEKeys = list(W_DANSE.keys())

    # transform to STFT domain, put frequency axis in front.
    inputFreq = STFTObj.stft(input, axis=1).transpose(1, 0, 2)

    # loop over the inputs in the centralized weights
    for key in centralizedKeys:
        centralized_W = W_MWF_central[key]
        # determine centralized output
        centralized_out = np.conj(centralized_W.transpose(0, 2, 1)) @ inputFreq
        centralized_out = np.real_if_close(
            STFTObj.istft(centralized_out, f_axis=0, t_axis=2)
        ).T
        centralized_out = centralized_out[:, : desired.shape[1]]
        LS_centralized = np.mean((desired - centralized_out) ** 2)
        ax.hlines(
            LS_centralized,
            1,
            len(W_DANSE[DANSEKeys[0]]),
            colors="k",
            linestyles="dashed",
        )
        ax.annotate(key, (1, LS_centralized), fontsize=plt.rcParams["font.size"] - 2)

    # loop over inputs in the DANSE weights
    for key in DANSEKeys:
        # determine outputs for DANSE
        Ws = W_DANSE[key]
        LS_danse = _computeLSCosts(Ws, inputFreq, STFTObj, desired)

        ax.semilogy(
            1 + np.arange(len(Ws)),
            LS_danse,
            "-" + markers[key],
            color=colors[key],
            linewidth=2,
            markevery=0.1,
            fillstyle="none",
            label=key,
        )

    # if needed, add the online-mode centralized filters
    if W_MWF_online != None:
        for key in W_MWF_online.keys():
            Ws = W_MWF_online[key]
            LS_danse = _computeLSCosts(Ws, inputFreq, STFTObj, desired)

            ax.semilogy(
                1 + np.arange(len(Ws)),
                LS_danse,
                "--",
                color="tab:gray",
            )
            ax.annotate(
                key,
                (len(Ws), LS_danse[-1]),
                size=plt.rcParams["font.size"] - 2,
                color="tab:gray",
                horizontalalignment="right",
            )

    # if needed, add the batchmode local filters
    if W_MWF_local != None:
        for key in W_MWF_local.keys():
            W_local = W_MWF_local[key]
            local_out = (
                np.conj(W_local.transpose(0, 2, 1))
                @ inputFreq[:, : W_local.shape[1], :]
            )
            local_out = np.real_if_close(
                STFTObj.istft(local_out, f_axis=0, t_axis=2)
            ).T[:, : desired.shape[1]]
            LS_local = np.mean((local_out - desired) ** 2)

            ax.axhline(LS_local, linestyle="dashed", color="k", linewidth=2)
            ax.annotate(
                key, (1, LS_local), size=plt.rcParams["font.size"] - 2, color="k"
            )

    ax.grid(True, which="both")
    ax.legend()
    ax.set(
        xlabel="Iteration",
        ylabel=r"$\mathbb{E}\{\|\mathbf{d}[k] - \mathbf{\hat{d}}[k]\|_2^2\}$",
        title="Evolution of LS cost over iterations",
    )
    ax.autoscale(tight=True, axis="x")
    fig.tight_layout()
    return fig


def _computeLSCosts(
    Ws: list[np.ndarray],
    sig: np.ndarray,
    STFTObj: signal.ShortTimeFFT,
    desired: np.ndarray,
) -> list[float]:
    """
    Utility function that allows for the computation of the LS costs given an
    input signal (in the frequency domain), a set of weights (in the freaquency
    domain) and the desired signal (in the time domain). The STFTObj is just
    used for the ISTFT. The metric is not computed in the frequency domain
    directly because of the WOLA processing.
    """
    lFFT, R, nFrames = sig.shape
    # stack for faster ISTFTs
    sigStacked = np.zeros((lFFT, R * len(Ws), nFrames), dtype=np.complex128)
    for i in range(len(Ws)):
        Wkk_H = np.conj(Ws[i].transpose(0, 2, 1))
        sigStacked[:, i * R : (i + 1) * R, :] = Wkk_H @ sig

    outStacked = STFTObj.istft(sigStacked, f_axis=0, t_axis=2).T[:, : desired.shape[1]]

    LS_costs = [
        np.mean((desired - outStacked[i * R : (i + 1) * R, :]) ** 2)
        for i in range(len(Ws))
    ]
    return LS_costs
