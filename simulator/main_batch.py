# Purpose of script:
# Main script for starting batchmode simulations of DANSE algorithms
#
# Context:
# This script has been created by Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
# alongside the "asc.py" and "setup.py" files as a starting point for simulations
# for the thesis "Distributed signal estimation in a real-world wireless sensor
# network". Later it was adapted by Basil Liekens to enhance the functionality.
#
# (c) Basil Liekens & Paul Didier

import DANSE_base
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import signal_generation as siggen
import sys
import utils
import warnings


def main():
    ## config
    # general
    nIter = int(5e1)
    nodeToTrack = 0
    nodesToTrack = [nodeToTrack]  # only one node is taken into account!
    sequential = [True, False]
    gevd = [False, True]

    # algorithms to use more in-depth: storing results & more advanced plots.
    desiredAlgos = [r"$DANSE_1$", r"$GEVD-DANSE_1$"]

    ## setup
    p = siggen.Parameters().load_from_yaml(PATH_TO_CFG)

    audio, noise, room = siggen.create_micsigs(p)
    fullAudio = np.sum(audio, axis=2)
    fullNoise = np.sum(noise, axis=2)

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
            np.sum(audio, axis=2)[nodeToTrack * p.Mk, :],
            p.fs,
            p.vadType,
        )
        if p.useVAD
        else None
    )

    ## start simulations
    # compute centralized MWF
    W_MWF_fd, signal_fd, interference_fd = DANSE_base.MWF.MWF_fd(
        fullAudio, fullNoise, e1, STFTObj, GEVD=False, Gamma=p.Gamma, mu=p.mu, vad=vad
    )
    W_MWF_gevd, audio_gevd, noise_gevd = DANSE_base.MWF.MWF_fd(
        fullAudio, fullNoise, e1, STFTObj, GEVD=True, Gamma=p.Gamma, mu=p.mu, vad=vad
    )

    ## perform DANSE iterations
    # storage for results
    Ws_DANSE: dict[str, list[np.ndarray]] = dict()
    audioOut: dict[str, list[np.ndarray]] = dict()
    noiseOut: dict[str, list[np.ndarray]] = dict()

    for seqMode in sequential:
        for gevdMode in gevd:
            algo = f"${'' if seqMode else 'rS-'}{'GEVD-' if gevdMode else ''}DANSE_1$"

            network = DANSE_base.batch_network(
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
                nodesToTrack,
            )

            Ws, audio_DANSE, noise_DANSE = network.performDANSE(
                fullAudio, fullNoise, nIter
            )

            # Remove the encapsulating list, only 1 output.
            Ws_DANSE[algo] = Ws[0]
            audioOut[algo] = audio_DANSE[0]
            noiseOut[algo] = noise_DANSE[0]

    ## compute metrics
    SINRinit = utils.metrics.SINR(
        (fullAudio + fullNoise)[0, :], fullAudio[0, :], fullNoise[0, :]
    )
    SINRafter_fd = utils.metrics.SINR(
        (signal_fd + interference_fd)[0, :], signal_fd[0, :], interference_fd[0, :]
    )
    SINRafter_GEVD = utils.metrics.SINR(audio_gevd + noise_gevd, audio_gevd, noise_gevd)
    SINRafter_DANSE: dict[str, float] = dict()

    STOIinit = utils.metrics.computeSTOI(
        fullAudio[[0], :], (fullAudio + fullNoise)[[0], :], p.fs
    )
    STOIafter_fd = utils.metrics.computeSTOI(
        fullAudio[[0], :], (signal_fd + interference_fd)[[0], :], p.fs
    )
    STOIafter_gevd = utils.metrics.computeSTOI(
        fullAudio[[0], :], (audio_gevd + noise_gevd)[[0], :], p.fs
    )
    STOIafter_DANSE: dict[str, float] = dict()

    sortedAlgos = sorted(list(Ws_DANSE.keys()), key=lambda x: len(x))
    for algo in sortedAlgos:
        d = audioOut[algo]
        n = noiseOut[algo]
        SINRafter_DANSE[algo] = utils.metrics.SINR((d + n)[0, :], d[0, :], n[0, :])
        STOIafter_DANSE[algo] = utils.metrics.computeSTOI(
            fullAudio[[0], :], (d + n)[[0], :], p.fs
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

    ## store results
    utils.playback.writeSoundFile((fullAudio + fullNoise)[0, :], p.fs, "received")

    utils.playback.writeSoundFile(
        (signal_fd + interference_fd)[0, :], p.fs, "after_centralized"
    )
    utils.playback.writeSoundFile(
        (audio_gevd + noise_gevd)[0, :], p.fs, "after_centralized_gevd"
    )

    for algo in desiredAlgos:
        if algo in sortedAlgos:
            utils.playback.writeSoundFile(
                (audioOut[algo] + noiseOut[algo])[0, :], p.fs, f"after_{algo[1:-1]}"
            )

    ## plot some results
    colors, markers = utils.plotting.getParameters()
    desired = np.sum(audio, axis=2)[[0], :]
    full = np.sum(audio, axis=2) + np.sum(noise, axis=2)
    Ws_centralized = {"Centralized": W_MWF_fd, "Centralized GEVD": W_MWF_gevd}

    fig1 = None
    for algo, weights in Ws_DANSE.items():
        fig1 = utils.plotting.MSEwPlotter(
            W_MWF_gevd if "GEVD-" in algo else W_MWF_fd,
            weights,
            label=algo,
            marker=markers[algo],
            color=colors[algo],
            fig=fig1,
        )

    _ = utils.plotting.LScostPlotter(
        Ws_centralized, Ws_DANSE, desired, full, STFTObj, colors, markers
    )

    _ = utils.plotting.STFTPlotter(
        STFTObj.stft((fullAudio + fullNoise)[nodeToTrack * p.Mk, :]), "Received signal"
    )
    _ = utils.plotting.STFTPlotter(
        STFTObj.stft((signal_fd + interference_fd)[0, :]),
        "After centralized filtering",
    )

    for algo in desiredAlgos:
        if algo in sortedAlgos:
            _ = utils.plotting.STFTPlotter(
                STFTObj.stft((audioOut[algo] + noiseOut[algo])[0, :]), f"After {algo}"
            )

    plotter = utils.plotting.spatialResponse(room, p, STFTObj, granularity=0.1)
    _ = plotter.plotFilterResponse(W_MWF_fd, filterType="Centralized")
    _ = plotter.plotFilterResponse(W_MWF_gevd, filterType="Centralized GEVD")

    for algo in desiredAlgos:
        if algo in sortedAlgos:
            _ = plotter.plotFilterResponse(weights[-1], filterType=algo)

    plt.show(block=True)

    return 0


if __name__ == "__main__":
    mpl.use("TkAgg")  # use TkAgg backend to avoid issues when plotting
    plt.ion()  # interactive mode on

    PATH_TO_CFG = "config/cfg.yml"  # path to configuration file (YAML)

    sys.exit(main())
