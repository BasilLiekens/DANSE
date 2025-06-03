"""
Example file for `sweep_variables.py`. Make your own file to avoid having to push
this one all the time!

This file contains the parameters needed to do the sweeps. I.e., it contains the
`sweepParams` and `coupling` lists that will be used for the generator.

`coupling` requires the lists that are coupled to be of equal length!

An example of how this file can be constructed is as follows:
>>> sweepParams = []
>>> coupling = []
>>>
>>> sweepParams.append(("lFFT", [512, 1024]))
>>> sweepParams.append(("deltaUpdate", [10, 20, 50, 100]))
>>> sweepParams.append(("lmbd", [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]))
>>> sweepParams.append(("GEVD", [True, False]))
>>> sweepParams.append(("sequential", [True, False]))
>>> coupling.append(["deltaUpdate", "lmbd"])
"""

sweepParams: list[tuple[str, list[object]]] = []
coupling: list[list[str]] = []

sweepParams.append(("lFFT", [1024]))
sweepParams.append(("deltaUpdate", [100]))
sweepParams.append(("lmbd", [0.99]))
sweepParams.append(("GEVD", [False, False, False, False, False, False, True]))
sweepParams.append(("sequential", [True, False]))
sweepParams.append(("Gamma", [0.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 0.0]))
sweepParams.append(("mu", [1.0, 5.0, 10.0, 50.0, 100.0]))

# turn into proper Monte-Carlo by changing rooms & speakers
sweepParams.append(("seed", [1024, 64, 512]))
sweepParams.append(("audio_sources", [["source_1"], ["source_1"], ["source_3"]]))
sweepParams.append(
    (
        "noise_sources",
        [["source_1"], ["source_2"], ["source_3"]],
    )
)

coupling.append(["seed", "audio_sources"])
coupling.append(["seed", "noise_sources"])
coupling.append(["GEVD", "Gamma"])
