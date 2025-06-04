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
from pyinstrument import Profiler
from scipy import signal
import signal_generation as siggen
import sys
import utils
import warnings


def main():
    nodeToTrack = 0

    ## setup
    p = siggen.Parameters().load_from_yaml(PATH_TO_CFG)
    dry, wet = siggen.load_session(p)
    audioSubstitute = np.zeros_like(dry)

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
            dry[nodeToTrack * p.Mk, :],
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
        dry, wet - dry, e1, STFTObj, GEVD=False, Gamma=p.Gamma, mu=p.mu, vad=vad
    )
    W_MWF_gevd, signal_gevd, interference_gevd = DANSE_base.MWF.MWF_fd(
        dry, wet - dry, e1, STFTObj, GEVD=True, Gamma=p.Gamma, mu=p.mu, vad=vad
    )

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
    Ws, audioOut, noiseOut = network.performDANSE(dry, wet - dry)

    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))

    ## compute metrics
    SINRinit = utils.metrics.vadBasedSINR(wet[0, :], dry[0, :], p.vadType, p.fs)
    SINRafter_fd = utils.metrics.vadBasedSINR(
        (signal_fd + interference_fd)[0, :], dry[0, :], p.vadType, p.fs
    )
    SINRafter_GEVD = utils.metrics.vadBasedSINR(
        (signal_gevd + interference_gevd)[0, :], dry[0, :], p.vadType, p.fs
    )
    SINRafter_DANSE = utils.metrics.vadBasedSINR(
        audioOut[0][0, :] + noiseOut[0][0, :],
        dry[0, :],
        p.vadType,
        p.fs,
    )
    STOIinit = utils.metrics.computeSTOI(
        dry[[nodeToTrack * p.Mk], :], wet[[nodeToTrack * p.Mk], :], p.fs
    )
    STOIafter_fd = utils.metrics.computeSTOI(
        dry[[nodeToTrack * p.Mk], :], (signal_fd + interference_fd), p.fs
    )
    STOIafter_gevd = utils.metrics.computeSTOI(
        dry[[nodeToTrack * p.Mk], :], (signal_gevd + interference_gevd), p.fs
    )
    STOIafter_DANSE = utils.metrics.computeSTOI(
        dry[[nodeToTrack * p.Mk], :], audioOut[0][[0], :] + noiseOut[0][[0], :], p.fs
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
    utils.playback.writeSoundFile(
        (signal_gevd + interference_gevd)[0, :], p.fs, "after_centralized_gevd"
    )

    ## plot some results
    desired = dry[[nodeToTrack * p.Mk], :]

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
        wet,
        STFTObj,
        colors,
        markers,
        Ws_centr_online,
    )
    _ = utils.plotting.STFTPlotter(
        STFTObj.stft(dry[nodeToTrack * p.Mk, :]), "Clean signal"
    )
    _ = utils.plotting.STFTPlotter(
        STFTObj.stft(wet[nodeToTrack * p.Mk, :]), "Received signal"
    )
    _ = utils.plotting.STFTPlotter(
        STFTObj.stft((signal_fd + interference_fd)[0, :]), "After centralized filtering"
    )
    _ = utils.plotting.STFTPlotter(
        STFTObj.stft(signal_gevd + interference_gevd)[0, :],
        "After centralized GEVD filtering",
    )
    _ = utils.plotting.STFTPlotter(
        STFTObj.stft((audioOut[0] + noiseOut[0])[0, :]), "After DANSE"
    )

    plt.show(block=True)


if __name__ == "__main__":
    mpl.use("TkAgg")  # avoid issues when plotting
    plt.ion()

    PATH_TO_CFG = "simulator/config/cfg.yml"

    sys.exit(main())
