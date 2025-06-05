# DANSE
This repository contains two parts: the first one is a wideband simulator for the Distributed Adaptive Node-specific Signal Estimation (DANSE) algorithm[^1],[^2],[^3]. This code handles both the algorithm itself and has a set of visualization tools to help in interpreting the results. This is located in the [simulator](simulator) folder.

A contribution of this repository is that it also contains code that allows to do validation of the DANSE algorithm in a real-world WASN by making use of the [MARVELO](https://github.com/CN-UPB/MARVELO) framework to orchestrate the network. This code is located in the [experimental](experimental) folder.

## Setup
Since MARVELO is can not be used for python versions smaller than 3.6 or larger than 3.10, it is required to use one that suits these requirements if the `experimental` part is desired. It should also be noted that no audio files are provided here, but open source speech corpora such as the [VCTK corpus](https://datashare.ed.ac.uk/handle/10283/3443) are available.

1. Create a virtual environment: `python -m venv .venv`
2. If the `experimental` part is of interest, install MARVELO according to the instructions available [here](https://marvelo.readthedocs.io/en/latest/md_files/getting_started.html#installation). Note, it is not needed to install ansible through the ppa. Instead it can also be installed through pip: `pip install ansible` (can automatically be done in a later stage).
3. In both cases, install the requirements listed in [requirements.txt](requirements.txt): `pip install -r requirements.txt`. This will overwrite some of the packages installed by MARVELO if that was installed first, but this should not be a problem.

After doing this, if the `experimental` part is of interest install the extra dependencies listed in [MARVELO_requirements.txt](MARVELO_requirements.txt) and set up the nodes according to the [MARVELO documentation](https://marvelo.readthedocs.io/en/latest/md_files/getting_started.html#installation).

Afterwards, follow the specific instructions for the part you want to work with.

## References

[^1]: A. Bertrand and M. Moonen, “Distributed adaptive node-specific signal estimation in fully connected sensor networks – Part I: Sequential node updating,”IEEE Transactions on Signal Processing, vol. 58, no. 10, pp. 5277–5291, 2010.

[^2]: A. Bertrand and M. Moonen, "Distributed Adaptive Node-Specific Signal Estimation in Fully Connected Sensor Networks—Part II: Simultaneous and Asynchronous Node Updating," in IEEE Transactions on Signal Processing, vol. 58, no. 10, pp. 5292-5306, Oct. 2010.

[^3]: A. Hassani, A. Bertrand and M. Moonen, "GEVD-Based Low-Rank Approximation for Distributed Adaptive Node-Specific Signal Estimation in Wireless Sensor Networks," in IEEE Transactions on Signal Processing, vol. 64, no. 10, pp. 2557-2572, May 2016.

