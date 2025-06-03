# Purpose of script:
# Main script for starting online mode simulations of DANSE algorithms
#
# Context:
# Simulation part of the thesis "Distributed signal estimation in a real-world
# wireless sensor network". This script is mostly based on the script
# "main_batch.py", but the main difference is now that the processing happens in
# an online fashion.
#
# (c) Basil Liekens

import DANSE_base
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pyinstrument import Profiler
import scipy.signal as signal
import signal_generation as siggen
import sys
import utils
import warnings


def main():
    nodeToTrack = 0

    ## setup
    p = siggen.Parameters().load_from_yaml(PATH_TO_CFG)

    audio, noise, room = siggen.create_micsigs(p)
    fullAudio = np.sum(audio, axis=2)
    fullNoise = np.sum(noise, axis=2)

    match p.window:
        case "sqrt hanning":
            window = np.sqrt(np.hanning(p.lFFT))
        case "ones":
            window = 1 / np.sqrt(1 / (1 - p.overlap)) * np.ones(p.lFFT)
        case _:
            warnings.warn("Window type not recognized, using scaled ones instead.")
            window: np.ndarray = 1 / np.sqrt(1 / p.overlap) * np.ones(p.lFFT)

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
    profiler = Profiler()
    profiler.start()

    ## centralized MWF
    W_MWF_fd, signal_fd, interference_fd = DANSE_base.MWF.MWF_fd(
        fullAudio, fullNoise, e1, STFTObj, GEVD=False, Gamma=p.Gamma, mu=p.mu, vad=vad
    )
    W_MWF_gevd, signal_gevd, noise_gevd = DANSE_base.MWF.MWF_fd(
        fullAudio, fullNoise, e1, STFTObj, GEVD=True, Gamma=p.Gamma, mu=p.mu, vad=vad
    )

    ## Online mode centralized MWF
    Ws_online, _, _ = DANSE_base.MWF_fd_online(
        fullAudio,
        fullNoise,
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
        fullAudio,
        fullNoise,
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
    network = DANSE_base.online_network(
        p.K,
        p.Mk,
        p.R,
        p.lFFT,
        p.overlap,
        p.window,
        p.alphaFormat,
        p.alpha0,
        p.GEVD,
        p.sequential,
        p.Gamma,
        p.mu,
        p.useVAD,
        p.vadType,
        p.fs,
        p.seed,
        nodesToTrack=[nodeToTrack],
        updateMode=p.updateMode,
        deltaUpdate=p.deltaUpdate,
        lmbd=p.lmbd,
    )
    Ws, audioOut, noiseOut = network.performDANSE(fullAudio, fullNoise)

    ## stop profiler
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))

    ## compute metrics
    SINRinit = utils.metrics.SINR(
        (fullAudio + fullNoise)[0, :], fullAudio[0, :], fullNoise[0, :]
    )
    SINRafter_fd = utils.metrics.SINR(
        signal_fd + interference_fd, signal_fd, interference_fd
    )
    SINRafter_GEVD = utils.metrics.SINR(
        signal_gevd + noise_gevd, signal_gevd, noise_gevd
    )
    SINRafter_DANSE = utils.metrics.SINR(
        audioOut[0][0, :] + noiseOut[0][0, :], audioOut[0][0, :], noiseOut[0][0, :]
    )
    STOIinit = utils.metrics.computeSTOI(
        fullAudio[[0], :], (fullAudio + fullNoise)[[0], :], p.fs
    )
    STOIafter_fd = utils.metrics.computeSTOI(
        fullAudio[[0], :], (signal_fd + interference_fd), p.fs
    )
    STOIafter_gevd = utils.metrics.computeSTOI(
        fullAudio[[0], :], (signal_gevd + noise_gevd), p.fs
    )
    STOIafter_DANSE = utils.metrics.computeSTOI(
        fullAudio[[0], :], audioOut[0][[0], :] + noiseOut[0][[0], :], p.fs
    )

    print(
        f"Initial SINR:\t\t\t{SINRinit} dB\nAfter Centralized MWF:\t\t"
        f"{SINRafter_fd} dB\nAfter Centralized GEVD MWF:\t{SINRafter_GEVD} dB"
        f"\nAfter DANSE:\t\t\t{SINRafter_DANSE} dB"
    )
    print(
        f"Initial STOI:\t\t\t{STOIinit}\nAfter Centralized MWF:\t\t{STOIafter_fd}"
        f"\nAfter Centralized GEVD MWF:\t{STOIafter_gevd}\nAfter DANSE:\t\t\t{STOIafter_DANSE}"
    )

    ## store the centralized and regular outputs
    utils.playback.writeSoundFile(
        (signal_fd + interference_fd)[0, :], p.fs, "after_centralized"
    )
    utils.playback.writeSoundFile(
        (audioOut[0] + noiseOut[0])[0, :], p.fs, "after_DANSE"
    )

    ## plot results
    _ = utils.plotting.plotWeights(W_MWF_fd, filterName="regular centralized")
    _ = utils.plotting.plotWeights(W_MWF_gevd, filterName="GEVD centralized")
    _ = utils.plotting.MSEwPlotter(W_MWF_fd, Ws[0], label=r"$DANSE_1$", marker="o")

    desired = fullAudio[[nodeToTrack * p.Mk], :]
    full = fullAudio + fullNoise

    Ws_centralized = {"Centralized": W_MWF_fd, "Centralized GEVD": W_MWF_gevd}
    Ws_centr_online = {
        "Centralized online": Ws_online,
        "Centralized GEVD online": Ws_GEVD_online,
    }
    Ws_DANSE = {
        f"${'' if p.sequential else 'rS-'}{'GEVD-' if p.GEVD else ''}DANSE_1$": Ws[0]
    }
    markers = {
        r"$DANSE_1$": "o",
        r"$rS-DANSE_1$": "x",
        r"$GEVD-DANSE_1$": "+",
        r"$rS-GEVD-DANSE_1$": "v",
    }
    colors = {
        r"$DANSE_1$": "tab:blue",
        r"$rS-DANSE_1$": "tab:orange",
        r"$GEVD-DANSE_1$": "tab:green",
        r"$rS-GEVD-DANSE_1$": "tab:red",
    }

    _ = utils.plotting.LScostPlotter(
        Ws_centralized,
        Ws_DANSE,
        desired,
        full,
        STFTObj,
        colors,
        markers,
        Ws_centr_online,
    )
    _ = utils.plotting.STFTPlotter(
        STFTObj.stft((fullAudio + fullNoise)[nodeToTrack * p.Mk, :]), "Received signal"
    )
    _ = utils.plotting.STFTPlotter(
        STFTObj.stft((signal_fd + interference_fd)[0, :]),
        "After centralized filtering",
    )
    _ = utils.plotting.STFTPlotter(
        STFTObj.stft((audioOut[0] + noiseOut[0])[0, :]), "After DANSE"
    )
    _ = utils.plotting.micPlotter(
        fullAudio[nodeToTrack * p.Mk, :],
        fullNoise[nodeToTrack * p.Mk, :],
        "Received signal",
    )
    _ = utils.plotting.micPlotter(
        signal_fd[0, :], interference_fd[0, :], "After centralized filtering"
    )
    _ = utils.plotting.micPlotter(
        signal_gevd[0, :], noise_gevd[0, :], "After centralized GEVD filtering"
    )
    _ = utils.plotting.micPlotter(
        audioOut[0][0, :], noiseOut[0][0, :], "Last DANSE output"
    )

    roomPlotter = utils.plotting.spatialResponse(room, p, STFTObj)
    _ = roomPlotter.plotFilterResponse(W_MWF_fd, "Centralized")
    _ = roomPlotter.plotFilterResponse(W_MWF_gevd, "Centralized GEVD")
    _ = roomPlotter.plotFilterResponse(Ws[0][-1], "Last DANSE")
    plt.show(block=True)


if __name__ == "__main__":
    mpl.use("TkAgg")  # use TkAgg backend to avoid issues when plotting
    plt.ion()  # interactive mode on

    PATH_TO_CFG = "config/cfg.yml"  # path to configuration file (YAML)

    sys.exit(main())
