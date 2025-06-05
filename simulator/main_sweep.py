# Purpose of script:
# Perform a sweep over a set of parameters to check their influence on the
# performance of the algorithm.
#
# Context:
# Simulation part of the thesis "Distributed signal estimation in a real-world
# wireless sensor network".
#
# (c) Basil Liekens

from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os  # some nice path utils
import parameter_sweep
import signal_generation as siggen
import shutil  # for copying just the yml file :sigh:
import sweep_variables as swVars
import sys
from tqdm import tqdm


def main():
    # generate location to store. Do so here already to allow the user to play
    # around with cfg.yml in other contexts.
    now = datetime.now()
    saveFolder = os.path.join(
        "simulator",
        "output",
        "sweeps",
        f"{now.year:02d}{now.month:02d}{now.day:02d}_{now.hour:02d}{now.minute:02d}{now.second:02d}",
    )
    if not os.path.isdir(saveFolder):
        os.mkdir(saveFolder)
    shutil.copy(PATH_TO_CFG, saveFolder)

    p = siggen.Parameters().load_from_yaml(PATH_TO_CFG)
    np.random.seed(p.seed)

    # create the parameters for the sweep and create the generator
    sweepParams = swVars.sweepParams
    coupling = swVars.coupling
    state = np.random.get_state()

    # set up the container (in this order due to the fact that the constructor
    # of the generator pops values from `sweepParams`)
    variables = [sweepParams[i][0] for i in range(len(sweepParams))]
    metrics = {
        "nw": ["LS_cost", "MSE_d", "MSE_w", "SINR", "STOI"],
        "loc": ["LS_cost", "SINR", "STOI"],
        "centr": ["LS_cost", "SINR", "STOI"],
    }
    container = parameter_sweep.simulationContainer(
        p.K, p.Mk, p.R, p.fs, variables, metrics, os.path.join(saveFolder, "data.csv")
    )

    # set up generator
    generator = parameter_sweep.parameterGenerator(
        parameter_sweep.sweepParameters(), sweepParams, coupling
    )

    for combination in tqdm(
        generator(),
        desc="Running over parameter combinations",
        total=len(generator),
    ):
        combination.adjustParameters(p)
        vars = dict()
        for var in variables:
            vars[var] = combination.__getattribute__(var)

        np.random.set_state(state)  # ensure the random initialization happens the same

        # regenerate the mic signals (a bit wasteful, but enables to work with
        # different settings). Ideally, a set of signals is generated beforehand
        # and then selected which one to use each iteration
        audio, noise, _ = siggen.create_micsigs(p)
        fullAudio = np.sum(audio, axis=2)
        fullNoise = np.sum(noise, axis=2)

        network = parameter_sweep.simulation(p)  # allow to be overridden for GC
        computedMetrics, timestamps = network.launch(fullAudio, fullNoise, metrics)
        container.processSimulation(vars, timestamps, computedMetrics)

    # store results
    container.store(os.path.join(saveFolder, "data.csv"))


if __name__ == "__main__":
    PATH_TO_CFG = "config/cfg.yml"
    mpl.use("TkAgg")  # avoid issues when plotting
    plt.ion()
    sys.exit(main())
