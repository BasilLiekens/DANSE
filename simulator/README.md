# Simulator

## Folder structure
There are 4 main tasks that can be executed; batch-mode simulations with [main_batch.py](code/main_batch.py), online-mode simulations with [main_online.py](code/main_online.py).

Furthermore, parameter sweeps with [main_sweep.py](code/main_sweep.py) and [visualize_sweeps.py](code/visualize_sweeps.py) can be done. `main_sweep.py` produces a csv file that can be processed with `visualize_sweeps.py` to visualize the results. These files require additional input from the files [sweep_variables.py](code/sweep_variables.py) and [visualization_variables.py](code/visualization_variables.py), respectively. These files are used to determine what parameters are swept.

Lastly, [postprocess_measurements.py](code/postprocess_measurements.py) allows to postprocess a set of recordings by bypassing the signal generation step and instead use the recorded signals.

The config files: `cfg.yml`, `sweep_variables.py` and `visualization_variables.py` are not present as of yet to avoid having to push these all the time. Instead, skeletons are available as `{filename}_example.{extension}` to help you get started.

Output files will be written to `simulator/output/audio` and `simulator/output/sweeps` so ensure those are present (also in .gitignore) and the relative path is correct.

## Known issues
1. There is a memory leak in the parameter sweeps which leads to the sweeps consuming over 30 GB of RAM when sweeping 250+ parameter combinations at once.