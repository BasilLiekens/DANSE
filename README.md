# DANSE
This repository contains two toolboxes for the Distributed Adaptive Node-specific Signal Estimation (DANSE) algorithm[^1] [^2] [^3]:
1. Under `simulator` a wideband simulator for this algorithm can be found. This code handles allows to perform both batch- and online-mode experiments as well as parameter sweeps. Furthermore, it also allows to postprocess real-world recordings as explained below. More info can be found in the [corresponding readme](./simulator/README.md).
2. The `experimental` folder contains some code that allows to perform experiments in a real-world Wireless Acoustic Sensor Network (WASN). This code makes use of the [MARVELO](https://github.com/CN-UPB/MARVELO) framework (incorporated as a git submodule) to orchestrate all nodes in the WASN. The results of these experiments can be postprocessed using [a script in the simulator](./simulator/postprocess_measurements.py). Again, more info can be found in the [corresponding readme](./experimental/README.md).

- [Setup](#setup)
- [Issues](#issues)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Setup
`MARVELO` enforces python to be $\geq$ 3.6. Development was done using python version 3.10.15. Hence, all python 3.10 can be expected to work. For other versions of python, no guarantees can be given about the functionality of this code, but it can be expected most of them will work fine as well.

It is also noteworthy that this repository is designed for audio processing, but no audio files have been included in this repository. Hence, the user is responsible for adding audio files. During development the [VCTK corpus](https://datashare.ed.ac.uk/handle/10283/3443) was used, but refer to the [corresponding readme](./simulator/README.md) for more details.

Installation instructions (for 1 virtual environment for both toolboxes in the root of the repository):
1. Create a virtual environment: `python -m venv .venv` 
2. Activate the virtual environment (platform and terminal specific).
3. If the `experimental` part is of interest, install `MARVELO` and its dependencies by installing the [corresponding requirements file](./MARVELO_requirements.txt): `pip install -r MARVELO_requirements.txt`
> [!NOTE]
> It is not needed to install `ansible` through the `ppa`. Instead it is automatically installed through pip (`pip install ansible`) when installing the requirements file.
4. In both cases, install the requirements listed in [requirements.txt](requirements.txt): `pip install -r requirements.txt`. This will overwrite some of the packages installed for MARVELO if that was installed first, but this should not pose any problems.
> [!IMPORTANT]
> The ordering of installation is important; `MARVELO` only enforces lower bounds of packages whereas the [requirements.txt](./requirements.txt) file fixes certain package versions.
5. If the `experimental` part is of interest, set up the client and nodes according to the [documentation](https://marvelo.readthedocs.io/en/latest/md_files/getting_started.html#installation). It should be noted that the `fission` package and `ansible` are already installed on the client. Extra information can be found in the [readme](./experimental/README.md).

Afterwards, follow the specific instructions for the part you want to work with.

## Issues
The simulator has a few known issues/points that should be taken into account, but this is also listed in the [corresponding readme](./simulator/README.md).

For other issues, please open a GitHub issue.

## Acknowledgements
This repository contains the most important pieces of code for the Master's thesis "An Analysis of Distributed Adaptive Node-specific Signal Estimation in Simulated and Real-world Wireless Acoustic Sensor Networks" at KU Leuven ESAT/STADIUS.

## Contact
Basil Liekens - basil.liekens@kuleuven.be

## References

[^1]: A. Bertrand and M. Moonen, “Distributed adaptive node-specific signal estimation in fully connected sensor networks – Part I: Sequential node updating,”IEEE Transactions on Signal Processing, vol. 58, no. 10, pp. 5277–5291, 2010.

[^2]: A. Bertrand and M. Moonen, "Distributed Adaptive Node-Specific Signal Estimation in Fully Connected Sensor Networks—Part II: Simultaneous and Asynchronous Node Updating," in IEEE Transactions on Signal Processing, vol. 58, no. 10, pp. 5292-5306, Oct. 2010.

[^3]: A. Hassani, A. Bertrand and M. Moonen, "GEVD-Based Low-Rank Approximation for Distributed Adaptive Node-Specific Signal Estimation in Wireless Sensor Networks," in IEEE Transactions on Signal Processing, vol. 64, no. 10, pp. 2557-2572, May 2016.

