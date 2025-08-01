# Purpose of script:
# Postprocess a set of recordings done on RPi's. Just use the same structures as
# always, but now bypass the construction of the signal as only a dry and wet
# recording is available.
#
# Context:
# Msc thesis, real-life part
#
# (c) Basil Liekens

import DANSE_base
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import signal_generation as siggen
import sys
import utils
import warnings


def main():
    ## config
    # general
    nodeToTrack = 0
    nodesToTrack = [nodeToTrack]  # only track 1 node in these experiments!
    sequential = [True, False]
    gevd = [False, True]

    # algorithms to use more in-depth; storing results & more advanced plots.
    desiredAlgos = [r"$DANSE_1$", r"$GEVD-DANSE_1$"]

    ## setup
    p = siggen.Parameters().load_from_yaml(PATH_TO_CFG)
    dry, wet = siggen.load_session(p)

    # dry & wet are still different recordings. Hence, the measurement noise
    # among others is different making purely relying on these not sensible.
    audioSubstitute = np.zeros_like(dry)

    match p.window:
        case "sqrt hanning":
            window = np.sqrt(np.hanning(p.lFFT))
        case "ones":
            window = np.sqrt(1 - p.overlap) * np.ones(p.lFFT)
        case _:
            warnings.warn("Window type not recognized, using scaled ones instead.")
            window: np.ndarray = np.sqrt(1 - p.overlap) * np.ones(p.lFFT)

    STFTObj = signal.ShortTimeFFT(
        window, int((1 - p.overlap) * p.lFFT), p.fs, fft_mode="onesided"
    )

    e1 = np.vstack(
        (
            np.zeros((p.Mk * nodeToTrack, p.R)),
            np.eye(p.R),
            np.zeros(((p.K - nodeToTrack) * p.Mk - p.R, p.R)),
        )
    )
    vad = (
        utils.vad.computeVAD(
            dry[nodeToTrack * p.Mk, :],
            p.fs,
            p.vadType,
        )
        if p.useVAD
        else None
    )

    ## start simulations
    ## local MWF
    W_MWF_loc, _, _ = DANSE_base.MWF_fd(
        audioSubstitute[nodeToTrack * p.Mk : (nodeToTrack + 1) * p.Mk, :],
        wet[nodeToTrack * p.Mk : (nodeToTrack + 1) * p.Mk, :],
        e1[nodeToTrack * p.Mk : (nodeToTrack + 1) * p.Mk, :],
        STFTObj,
        GEVD=False,
        Gamma=p.Gamma,
        mu=p.mu,
        vad=vad,
    )
    W_MWF_gevd_loc, _, _ = DANSE_base.MWF_fd(
        audioSubstitute[nodeToTrack * p.Mk : (nodeToTrack + 1) * p.Mk, :],
        wet[nodeToTrack * p.Mk : (nodeToTrack + 1) * p.Mk, :],
        e1[nodeToTrack * p.Mk : (nodeToTrack + 1) * p.Mk, :],
        STFTObj,
        GEVD=True,
        Gamma=p.Gamma,
        mu=p.mu,
        vad=vad,
    )

    ## centralized MWF
    W_MWF_fd, _, _ = DANSE_base.MWF_fd(
        audioSubstitute, wet, e1, STFTObj, GEVD=False, Gamma=p.Gamma, mu=p.mu, vad=vad
    )
    W_MWF_gevd, _, _ = DANSE_base.MWF_fd(
        audioSubstitute, wet, e1, STFTObj, GEVD=True, Gamma=p.Gamma, mu=p.mu, vad=vad
    )

    wetFd = STFTObj.stft(wet, axis=1).transpose(1, 0, 2)
    fd_out = np.conj(W_MWF_fd.transpose(0, 2, 1)) @ wetFd
    gevd_fd_out = np.conj(W_MWF_gevd.transpose(0, 2, 1)) @ wetFd

    central_out = STFTObj.istft(fd_out, f_axis=0, t_axis=2).T[:, : wet.shape[1]]
    central_gevd_out = STFTObj.istft(gevd_fd_out, f_axis=0, t_axis=2).T[
        :, : wet.shape[1]
    ]

    ## Online mode centralized MWF
    Ws_online, _, _ = DANSE_base.MWF_fd_online(
        audioSubstitute,
        wet,
        e1,
        STFTObj,
        p.deltaUpdate,
        p.lmbd,
        p.updateMode,
        GEVD=False,
        Gamma=p.Gamma,
        mu=p.mu,
        vad=vad,
    )
    Ws_GEVD_online, _, _ = DANSE_base.MWF_fd_online(
        audioSubstitute,
        wet,
        e1,
        STFTObj,
        p.deltaUpdate,
        p.lmbd,
        p.updateMode,
        GEVD=True,
        Gamma=p.Gamma,
        mu=p.mu,
        vad=vad,
    )

    ## perform DANSE
    # storage for results
    Ws_DANSE: dict[str, list[np.ndarray]] = dict()
    audioOut: dict[str, list[np.ndarray]] = dict()
    noiseOut: dict[str, list[np.ndarray]] = dict()

    for seqMode in sequential:
        for gevdMode in gevd:
            algo = f"${'' if seqMode else 'rS-'}{'GEVD-' if gevdMode else ''}DANSE_1$"

            network = DANSE_base.online_network(
                p.K,
                p.Mk,
                p.R,
                p.lFFT,
                p.overlap,
                p.window,
                p.alphaFormat,
                p.alpha0,
                gevdMode,
                seqMode,
                p.Gamma,
                p.mu,
                p.useVAD,
                p.vadType,
                p.fs,
                p.seed,
                nodesToTrack=nodesToTrack,
                updateMode=p.updateMode,
                deltaUpdate=p.deltaUpdate,
                lmbd=p.lmbd,
            )

            # Both wet & dry are used as the dry recording is used for the VAD.
            # It should be noted however that for the metrics the only thing
            # that can be relied on is the joint output.
            Ws, audio_DANSE, noise_DANSE = network.performDANSE(dry, wet - dry)

            # Remove the encapsulating list, only 1 output.
            Ws_DANSE[algo] = Ws[0]
            audioOut[algo] = audio_DANSE[0]
            noiseOut[algo] = noise_DANSE[0]

    ## compute metrics
    # first snip off the part with the remainder of the synchronization pulse
    # for more honest metrics
    firstIdx = siggen.getStartSample(dry[nodeToTrack * p.Mk, :], p.fs)
    vadReferenceSINR = dry[0, firstIdx:]

    SINRinit = utils.metrics.vadBasedSINR(
        wet[0, firstIdx:], vadReferenceSINR, p.vadType, p.fs
    )
    SINRafter_fd = utils.metrics.vadBasedSINR(
        central_out[0, firstIdx:], vadReferenceSINR, p.vadType, p.fs
    )
    SINRafter_GEVD = utils.metrics.vadBasedSINR(
        central_gevd_out[0, firstIdx:], vadReferenceSINR, p.vadType, p.fs
    )
    SINRafter_DANSE: dict[str, float] = dict()

    vadReferenceSTOI = dry[nodeToTrack * p.Mk : nodeToTrack * p.Mk + p.R, firstIdx:]
    STOIinit = utils.metrics.computeSTOI(
        vadReferenceSTOI,
        wet[nodeToTrack * p.Mk : nodeToTrack * p.Mk + p.R, firstIdx:],
        p.fs,
    )
    STOIafter_fd = utils.metrics.computeSTOI(
        vadReferenceSTOI, central_out[:, firstIdx:], p.fs
    )
    STOIafter_gevd = utils.metrics.computeSTOI(
        vadReferenceSTOI, central_gevd_out[:, firstIdx:], p.fs
    )
    STOIafter_DANSE: dict[str, float] = dict()

    sortedAlgos = sorted(list(Ws_DANSE.keys()), key=lambda x: len(x))
    for algo in sortedAlgos:
        d = audioOut[algo]
        n = noiseOut[algo]
        SINRafter_DANSE[algo] = utils.metrics.vadBasedSINR(
            (d + n)[0, firstIdx:], vadReferenceSINR, p.vadType, p.fs
        )
        STOIafter_DANSE[algo] = utils.metrics.computeSTOI(
            vadReferenceSTOI, (d + n)[[0], firstIdx:], p.fs
        )

    print(
        f"Initial SINR:\t\t\t{SINRinit} dB\nAfter Centralized MWF:\t\t"
        f"{SINRafter_fd} dB\nAfter Centralized GEVD MWF:\t{SINRafter_GEVD} dB"
    )
    for algo in sortedAlgos:
        nSpaces = len(sortedAlgos[-1]) - len(algo)  # list is sorted on length!
        print(f"SINR after {algo[1:-1]}:{nSpaces*' '}\t{SINRafter_DANSE[algo]} dB")

    print(
        f"\nInitial STOI:\t\t\t{STOIinit}\nAfter Centralized MWF:\t\t{STOIafter_fd}"
        f"\nAfter Centralized GEVD MWF:\t{STOIafter_gevd}"
    )
    for algo in sortedAlgos:
        nSpaces = len(sortedAlgos[-1]) - len(algo)  # list is sorted on length!
        print(f"STOI after {algo[1:-1]}:{nSpaces*' '}\t{STOIafter_DANSE[algo]}")

    ## store the centralized and regular outputs
    utils.playback.writeSoundFile(central_out[0, :], p.fs, "after_centralized")
    utils.playback.writeSoundFile(
        central_gevd_out[0, :], p.fs, "after_centralized_gevd"
    )

    for algo in desiredAlgos:
        for algo in sortedAlgos:
            utils.playback.writeSoundFile(
                (audioOut[algo] + noiseOut[algo])[0, :], p.fs, f"after_{algo[1:-1]}"
            )

    ## plot some results
    colors, markers = utils.plotting.getParameters()

    desired = dry[[nodeToTrack * p.Mk], :]

    Ws_centralized = {
        "Centralized batch": W_MWF_fd,
        "Centralized GEVD batch": W_MWF_gevd,
    }
    WS_local = {"Local batch": W_MWF_loc, "Local GEVD batch": W_MWF_gevd_loc}
    Ws_centr_online = {
        "Centralized online": Ws_online,
        "Centralized GEVD online": Ws_GEVD_online,
    }

    fig = None
    for algo, weights in Ws_DANSE.items():
        fig = utils.plotting.MSEwPlotter(
            W_MWF_gevd if "GEVD-" in algo else W_MWF_fd,
            weights,
            label=algo,
            color=colors[algo],
            marker=markers[algo],
            fig=fig,
        )

        ax = fig.gca()
        ax.set(ylim=[1e-4, 1e2])

    fig = utils.plotting.LScostPlotter(
        Ws_centralized,
        Ws_DANSE,
        desired,
        wet,
        STFTObj,
        colors,
        markers,
        Ws_centr_online,
        WS_local,
    )
    ax = fig.gca()
    ax.set(ylim=[1e-5, 5e-2])

    _ = utils.plotting.STFTPlotter(
        STFTObj.stft(dry[nodeToTrack * p.Mk, :]), "Clean signal"
    )
    _ = utils.plotting.STFTPlotter(
        STFTObj.stft(wet[nodeToTrack * p.Mk, :]), "Received signal"
    )
    _ = utils.plotting.STFTPlotter(
        STFTObj.stft(central_out)[0, :], "After centralized filtering"
    )
    _ = utils.plotting.STFTPlotter(
        STFTObj.stft(central_gevd_out)[0, :],
        "After centralized GEVD filtering",
    )

    for algo in desiredAlgos:
        if algo in sortedAlgos:
            utils.plotting.STFTPlotter(
                STFTObj.stft((audioOut[algo] + noiseOut[algo])[0, :]), f"After {algo}"
            )

    plt.show(block=True)


if __name__ == "__main__":
    mpl.use("TkAgg")  # avoid issues when plotting
    plt.ion()

    PATH_TO_CFG = "config/cfg.yml"

    sys.exit(main())
