# DANSE
This repository contains code for performing wideband simulations of the Distributed Adaptive Node-specific Signal Estimation (DANSE) algorithm[^1],[^2],[^3]. This code handles both the algorithm itself and has a set of visualization tools to help in interpreting the results.

## Setup
Create a virtual environment: `python -m venv .venv` and install the packages listed in [requirements.txt](code/requirements.txt): `pip install -r requirements.txt`.

The user's own audio files should still be brought in to be able to simulate.

Afterwards, set the correct paths in the config files and run one of the three entry points.

## Repository structure
There are 4 main tasks that can be executed; batch-mode simulations with [main_batch.py](code/main_batch.py), online-mode simulations with [main_online.py](code/main_online.py).

Furthermore, parameter sweeps with [main_sweep.py](code/main_sweep.py) and [visualize_sweeps.py](code/visualize_sweeps.py) can be done. `main_sweep.py` produces a csv file that can be processed with `visualize_sweeps.py` to visualize the results. These files require additional input from the files [sweep_variables.py](code/sweep_variables.py) and [visualization_variables.py](code/visualization_variables.py), respectively. These files are used to determine what parameters are swept.

Lastly, [postprocess_measurements.py](code/postprocess_measurements.py) allows to postprocess a set of recordings by bypassing the signal generation step and instead use the recorded signals.

## Known issues
1. There is a memory leak in the parameter sweeps which leads to the sweeps consuming over 30 GB of RAM when sweeping 250+ parameter combinations at once.

## References

[^1]: A. Bertrand and M. Moonen, “Distributed adaptive node-specific signal estimation in fully connected sensor networks – Part I: Sequential node updating,”IEEE Transactions on Signal Processing, vol. 58, no. 10, pp. 5277–5291, 2010.

[^2]: A. Bertrand and M. Moonen, "Distributed Adaptive Node-Specific Signal Estimation in Fully Connected Sensor Networks—Part II: Simultaneous and Asynchronous Node Updating," in IEEE Transactions on Signal Processing, vol. 58, no. 10, pp. 5292-5306, Oct. 2010.

[^3]: A. Hassani, A. Bertrand and M. Moonen, "GEVD-Based Low-Rank Approximation for Distributed Adaptive Node-Specific Signal Estimation in Wireless Sensor Networks," in IEEE Transactions on Signal Processing, vol. 64, no. 10, pp. 2557-2572, May 2016.

