# Purpose of script:
# Main script for starting batchmode simulations of DANSE algorithms
#
# Context:
# This script has been created by Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
# alongside the "asc.py" and "setup.py" files as a starting point for simulations
# for the thesis "Distributed signal estimation in a real-world wireless sensor
# network".
#
# (c) Basil Liekens & Paul Didier

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

    e1 = np.vstack((np.eye(p.R), np.zeros((p.K * p.Mk - p.R, p.R))))
    vad = (
        utils.vad.computeVAD(
            np.sum(audio, axis=2)[0, :],
            fs=p.fs,
            type=p.vadType,
        )
        if p.useVAD
        else None
    )

    ## start simulations
    profiler = Profiler()
    profiler.start()

    ## compute centralized MWF
    W_MWF_fd, signal_fd, interference_fd = DANSE_base.MWF.MWF_fd(
        fullAudio, fullNoise, e1, STFTObj, GEVD=False, Gamma=p.Gamma, mu=p.mu, vad=vad
    )
    W_MWF_gevd, audio_gevd, noise_gevd = DANSE_base.MWF.MWF_fd(
        fullAudio, fullNoise, e1, STFTObj, GEVD=True, Gamma=p.Gamma, mu=p.mu, vad=vad
    )

    ## perform DANSE iterations
    nIter = int(2.5e2)
    network = DANSE_base.batch_network(
        p.K,
        p.Mk,
        p.R,
        p.lFFT,
        p.overlap,
        p.window,
        p.alphaFormat,
        p.alpha0,
        False,  # GEVD
        True,  # sequential
        p.Gamma,
        p.mu,
        p.useVAD,
        p.vadType,
        p.fs,
        p.seed,
    )
    Ws, audioOut, noiseOut = network.performDANSE(fullAudio, fullNoise, nIter)

    network = DANSE_base.batch_network(
        p.K,
        p.Mk,
        p.R,
        p.lFFT,
        p.overlap,
        p.window,
        p.alphaFormat,
        p.alpha0,
        True,  # GEVD
        True,  # sequential
        p.Gamma,
        p.mu,
        p.useVAD,
        p.vadType,
        p.fs,
        p.seed,
    )
    Ws_gevd, audioOut_gevd, noiseOut_gevd = network.performDANSE(
        fullAudio, fullNoise, nIter
    )

    network = DANSE_base.batch_network(
        p.K,
        p.Mk,
        p.R,
        p.lFFT,
        p.overlap,
        p.window,
        p.alphaFormat,
        p.alpha0,
        False,  # GEVD
        False,  # sequential
        p.Gamma,
        p.mu,
        p.useVAD,
        p.vadType,
        p.fs,
        p.seed,
    )
    Ws_sync, _, _ = network.performDANSE(fullAudio, fullNoise, nIter)

    network = DANSE_base.batch_network(
        p.K,
        p.Mk,
        p.R,
        p.lFFT,
        p.overlap,
        p.window,
        p.alphaFormat,
        p.alpha0,
        True,  # GEVD
        False,  # sequential
        p.Gamma,
        p.mu,
        p.useVAD,
        p.vadType,
        p.fs,
        p.seed,
    )
    Ws_gevd_sync, _, _ = network.performDANSE(fullAudio, fullNoise, nIter)

    ## stop profiler after the algorithms themselves
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))

    ## compute metrics
    SINRinit = utils.metrics.SINR((audio + noise)[0, :], audio[0, :], noise[0, :])

    SINRafter_fd = utils.metrics.SINR(
        signal_fd + interference_fd, signal_fd, interference_fd
    )
    SINRafter_GEVD = utils.metrics.SINR(audio_gevd + noise_gevd, audio_gevd, noise_gevd)
    SINRafter_DANSE = utils.metrics.SINR(
        audioOut[0][0, :] + noiseOut[0][0, :], audioOut[0][0, :], noiseOut[0][0, :]
    )
    SINRafter_GEVD_DANSE = utils.metrics.SINR(
        audioOut_gevd[0][0, :] + noiseOut_gevd[0][0, :],
        audioOut_gevd[0][0, :],
        noiseOut_gevd[0][0, :],
    )

    print(
        f"SINR before beamforming: {SINRinit} dB, SINR after beamforming: "
        f"{SINRafter_fd} dB for centralized filter in frequency domain, "
        f"{SINRafter_GEVD} dB for GEVD based centralized filter."
    )
    print(
        f"SINR after DANSE beamforming (in frequency domain): {SINRafter_DANSE}, "
        f"{SINRafter_GEVD_DANSE} dB for GEVD-based DANSE."
    )

    ## write back results
    utils.playback.writeSoundFile(
        np.sum(audio, axis=2)[0, :] + np.sum(noise, axis=2)[0, :], p.fs, "received_data"
    )
    utils.playback.writeSoundFile(
        signal_fd + interference_fd, p.fs, "after_freq_beamforming_centralized"
    )
    utils.playback.writeSoundFile(
        audio_gevd + noise_gevd, p.fs, "after_GEVD_beamforming_centralized"
    )
    utils.playback.writeSoundFile(
        (audioOut[0] + noiseOut[0])[0, :], p.fs, "after_DANSE"
    )
    utils.playback.writeSoundFile(
        (audioOut_gevd[0] + noiseOut_gevd[0])[0, :], p.fs, "after_GEVD_DANSE"
    )

    ## plot some results
    fig1 = utils.plotting.MSEwPlotter(W_MWF_fd, Ws[0], label=r"$DANSE_1$", marker="o")
    fig1 = utils.plotting.MSEwPlotter(
        W_MWF_fd, Ws_sync[0], label=r"$rS-DANSE_1$", marker="x", fig=fig1
    )
    fig1 = utils.plotting.MSEwPlotter(
        W_MWF_gevd, Ws_gevd[0], label=r"$GEVD-DANSE_1$", marker="+", fig=fig1
    )
    fig1 = utils.plotting.MSEwPlotter(
        W_MWF_gevd, Ws_gevd_sync[0], label=r"$rS-GEVD-DANSE_1$", marker="v", fig=fig1
    )

    desired = np.sum(audio, axis=2)[[0], :]
    full = np.sum(audio, axis=2) + np.sum(noise, axis=2)
    Ws_centralized = {"Centralized": W_MWF_fd, "Centralized GEVD": W_MWF_gevd}
    Ws_DANSE = {
        r"$DANSE_1$": Ws[0],
        r"$rS-DANSE_1$": Ws_sync[0],
        r"$GEVD-DANSE_1$": Ws_gevd[0],
        r"$rS-GEVD-DANSE_1$": Ws_gevd_sync[0],
    }
    colors = {
        r"$DANSE_1$": "tab:blue",
        r"$rS-DANSE_1$": "tab:orange",
        r"$GEVD-DANSE_1$": "tab:green",
        r"$rS-GEVD-DANSE_1$": "tab:red",
    }
    markers = {
        r"$DANSE_1$": "o",
        r"$rS-DANSE_1$": "x",
        r"$GEVD-DANSE_1$": "+",
        r"$rS-GEVD-DANSE_1$": "v",
    }

    _ = utils.plotting.LScostPlotter(
        Ws_centralized, Ws_DANSE, desired, full, STFTObj, colors, markers
    )

    plotter = utils.plotting.spatialResponse(room, p, STFTObj, granularity=0.1)
    _ = plotter.plotFilterResponse(W_MWF_fd, filterType="Centralized")
    _ = plotter.plotFilterResponse(W_MWF_gevd, filterType="Centralized GEVD")
    _ = plotter.plotFilterResponse(Ws[0][-1], filterType="DANSE")
    _ = plotter.plotFilterResponse(Ws_gevd[0][-1], filterType="GEVD-DANSE")

    room.plot()

    fig3 = utils.plotting.micPlotter(
        np.sum(audio, axis=2)[0, :], np.sum(noise, axis=2)[0, :], title="Input signals"
    )
    fig4 = utils.plotting.micPlotter(
        audioOut[0][0, :], noiseOut[0][0, :], title="After DANSE"
    )
    fig5 = utils.plotting.micPlotter(
        audioOut_gevd[0][0, :], noiseOut_gevd[0][0, :], title="After GEVD-DANSE"
    )

    # get all figures to the same y-axes
    vmin = np.min(
        (
            np.min(np.sum(audio, axis=2)[0, :] + np.sum(noise, axis=2)[0, :]),
            np.min(audioOut[0][0, :] + noiseOut[0][0, :]),
            np.min(audioOut_gevd[0][0, :] + noiseOut_gevd[0][0, :]),
        )
    )
    vmax = np.max(
        (
            np.max(np.sum(audio, axis=2)[0, :] + np.sum(noise, axis=2)[0, :]),
            np.max(audioOut[0][0, :] + noiseOut[0][0, :]),
            np.max(audioOut_gevd[0][0, :] + noiseOut_gevd[0][0, :]),
        )
    )

    ax3 = fig3.gca()
    ax3.set(ylim=[vmin, vmax])
    ax4 = fig4.gca()
    ax4.set(ylim=[vmin, vmax])
    ax5 = fig5.gca()
    ax5.set(ylim=[vmin, vmax])

    plt.show(block=True)

    return 0


if __name__ == "__main__":
    mpl.use("TkAgg")  # use TkAgg backend to avoid issues when plotting
    plt.ion()  # interactive mode on

    PATH_TO_CFG = "config/cfg.yml"  # path to configuration file (YAML)

    sys.exit(main())
