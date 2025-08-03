# Experimental
This folder includes the code that is needed to perform experiments where the DANSE algorithm is applied in a real-world WASN using the [MARVELO](https://github.com/CN-UPB/MARVELO) framework. 

The MARVELO framework itself is included as a git submodule and the required python packages for this can be installed, as described in the [top-level readme](./../README.md), by executing `pip install -r MARVELO_requirements.txt`. For implementation details concerning MARVELO itself, please refer to the [official documentation](https://marvelo.readthedocs.io/en/latest/).

- [Folder structure](#folder-structure)
- [Installing MARVELO](#installing-marvelo)
- [Running experiments](#running-experiments)

## Folder structure
This folder contains the implementation of a few tasks for a WASN: plain recording, a multichannel Wiener filter (MWF) (potentially using the recordings from all nodes in the network) and the DANSE algorithm.

This folder contains a few important files and folders: 
- [manage.py](./manage.py) for starting the MARVELO jobs: simply execute `python manage.py <network_name> run`, where `<network_name>` is `recordings`, `MWF` or `DANSE`. Each of these networks has a dedicated folder that contains a `config.py` folder that implements the actual network. The implementation of all jobs is kept generically under the `shared` folder such that all jobs can access them. (E.g., both the MWF and DANSE networks make use of the jobs originally created for recording.) 
- [main_playback.py](./main_playback.py). This file is used to start playback on speakers connected to the client pc. Synchronization is handled by inserting a "template" that is detected through an autocorrelation method. (Just relying on the start signal of the client to all nodes of the network could cause too large of a mismatch. This approach also has some downsides as it aligns all Room Impulse Responses (RIRs) and might not be 100% exact, but at least ensures the information in frames mostly overlaps: since the processing happens in the frequency domain, being off a few samples is not too harmful.)
- A config file, of which an example is given in [config/cfg_example.yml](config/cfg_example.yml) determines the parameters of the jobs: the "signal processing parameters", number of channels per node (assumed to be constant for all nodes) as well as the signal being played back. (Also allows to configure the template insertion.) When running networks with 3 nodes, this file is the only one that should be touched.

> [!IMPORTANT]
> All networks were designed for 3 nodes, if that would change it would be required to change the `config.py` in the corresponding network folder as well. Additionally, it is assumed all nodes have the same number of channels just like in the simulator. While it would be possible to change the number of sensors per node, each node internally assumes all nodes have the same number of channels as itself to allocate matrices.

## Installing MARVELO
When setting up the nodes in the networks according to the [documentation](https://marvelo.readthedocs.io/en/latest/), there are a few things to note:
- Related to the client:
    1. The `BATMAN` client is not required on the client pc.
- Related to the nodes:
    1. In MARVELO, the `dispynode.py` script is required. To make this work properly, it is important to first run the Ansible playbook. This will ensure a jinja template gets copied over which ensures the `dispynode.py` script is available at boot. (Can be verified by running `ps cax | grep dispynode.py` in the terminal.)

       It should be noted however that older versions of MARVELO install the pip packages in the OS python distributions which is no longer supported on more recent versions of Debian and its derivatives. To circumvent this issue, it is recommended to let the playbook make a directory (e.g. `~/installations`), create a virtual environment there and do everything in there.
    2. Prior to running the first experiment then, activate the virtual environment, kill the running `dispynode.py` process and execute `dispynode.py --clean` in the terminal. If everything was done correctly, the prompt for input that this script should automatically be filled with "terminated", allowing to execute a new command. If something went wrong, executing `dispynode.py --clean` will block the terminal. Subsequent experiments will always require running `dispynode.py --clean` with the correct virtual environment activated, but manually killing the `dispynode.py` process is no longer required.
    3. For communication, the [pickle protocol](https://docs.python.org/3/library/pickle.html) is used. This has as a consequence that everything used to construct the objects sent over from the client to the nodes should be available in the same place. In practice, this will come down to three things: `fission`, `shared` and `utils` to the same location as the `dispynode.py` script (`.venv/bin/.`). 
    
       The `fission` folder could theoretically also be installed the same way as it was done on the client, but the other 2 will have to be copied to the bin of the virtual environment. This can for example be done by executing `scp -r </path/to/folder> <username>@<hostname>.local:~/installations/.venv/bin/.` where it is assumed that the folder in which the virtual environment is called `installations` and is located in the root of the home directory of the user. 

> [!NOTE]
> Using `scp` to copy into the folder that was created with an Ansible playbook is not possible. To circumvent this issue, it is easiest to manually recreate the `.venv` using `ssh` after the playbook has been run. Afterwards, `scp` can be used to copy the required directories to the correct location on the nodes.
    
Three additional, more general comments:
- MARVELO runs in a different folder than the location of the code. Therefore, if you want to read files during execution or write files for logging, it is recommended to use absolute paths rather than relative ones and to also do this in the config file.
- In general it is recommended to use the actual IP address; using `<hostname>.local` often also works, but this might cause some strange artefacts related to MARVELO sometimes internally using the IP address.
- When there is a cycle in the job graph in a MARVELO network, the network will be deadlocked and hence unable to start: each job is waiting for an input from another one to be able to start. This happens for example in the `DANSE` network where the `fusionJob` is reliant on the weights computed in the `DANSEJob`. However, the `DANSEJob` is reliant on the fused signals from other nodes as well, leading to the following cycle: 
   ```
   fusionJob (node 1) -> DANSEJob (node 2) -> fusionJob (node 2) -> DANSEJob (node 1) -> fusionJob (node 1) 
   ```
   Since the weights are updated so infrequently, this cycle can be broken by letting the `DANSEJob` write its weights to disk and let the `fusionJob` read it in from there. Some additional synchronization methods are then used to ensure the weights are only read if they are new and completed. (A race condition could exist where `DANSEJob` is still writing to disk and `fusionJob` already tries to read.)

## Running experiments
To run experiments, the following steps have to be taken:
1. Reactivate the `dispynode.py` script (is required after every execution of a network). 
2. Start the network using [manage.py](./manage.py).
3. Start audio playback using [main_playback.py](./main_playback.py). The insertion of the template will ensure all algorithms are started approximately at the same time. (This means the order is important: first the nodes should be initialized, only then can they detect the template which is a requirement for them to be able to start.)

> [!IMPORTANT]
> The VAD that is used in these experiments, [Silero](https://github.com/snakers4/silero-vad), is a neural network that is trained to detect any speech activity. This means that interfering speakers will also get picked up. (Even if they were to be babble noise, as suggested in the [simulator](./../README.md).) Therefore, the `MWF` and `DANSE` networks have been designed to be used in a two-stage approach where the VAD is computed based on the desired signal recorded in the first stage: 
1. In the first stage the `recording` network makes a recording of only the desired signals: `python manage.py recording run`. 
    - Set the `calibration` flag in [main_playback.py](./main_playback.py) to `True`.
    - Ensure the `path_to_calibration` matches the combination of `baseDir` and `fileName` in the [config file](./config/cfg_example.yml): this ensures the correct file is loaded in in the second stage for the computation of the VAD.
2. Run the network of interest: `python manage.py <network_name> run`. 
    - Set the `calibration` flag in [main_playback.py](./main_playback.py) to `False`.
    - Setting the paths correctly in the previous stage will ensure the VAD will be computed based on a clean version of the desired signal.
    - It is recommended to change the `fileName` attribute in the [config file](./config/cfg_example.yml) to have access to both the clean desired signal as well as the recording with both the desired and interfering signals. 

> [!NOTE]
> If both the clean, desired signal as well as the complete recording are available per node in the network in one folder, the [postprocessing script](./../simulator/postprocess_measurements.py) can be ran to process the recordings with different algorithms etc. (Rather than having to re-run the same experiment with different parameters over and over again.) This postprocessing script assumes dry and wet recordings are stored as `recording_dry_{node_number}.wav` and `recording_wet_{node_number}.wav`, respectively.
