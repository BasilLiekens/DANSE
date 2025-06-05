# Experimental

## Preliminaries
This code internally makes use of the MARVELO code. For more explanation on the implementation details concerning that, please refer to the [official documentation](https://marvelo.readthedocs.io/en/latest/). 

There are, however, a few things to note on this. 

1. The `BATMAN` client is not required on the client pc.
2. In order to be able to allow MARVELO to detect your devices, first do the setup with the playbook which will copy the template. Afterwards, ensure `dispynode.py` is running after boot (`ps cax | grep dispy`). Then manually kill it, activate the virtual environment that stores that job, and restart it by just calling `dispynode.py --clean` in the terminal. This should print some stuff to the terminal, but automatically fill in `terminated` and allow you to enter a new command. If this is the case MARVELO will be able to detect the nodes. After every run this call should be repeated (but killing the `dispynode.py` process is no longer necessary)
3. Since the data is sent using `pickle`, relative imports are important. Therefore, push everything the jobs need into the `bin` of the `.venv` of the devices that will run the job. This means that the folder that contains your jobs as well as the `utils` folders need to be pushed there. This can be done by `scp -r /path/to/folder/on/client {username}@{hostname}.local:~/path/to/venv/.venv/bin/.`.
4. A last comment is that MARVELO runs in a different folder than the location of the code. Therefore, if you want to read files during execution or write files for logging, it is recommended to use absolute paths (alo in the config!)

## Folder structure
This folder contains a few important files: [manage.py](manage.py) for starting the MARVELO processes. A [config file](config/cfg_example.yml) and a set of jobs that can be executed with `manage.py`. The config file determines the parameters of the jobs and is the only part that should be changed for changing the parameters. 

All networks were designed for 3 nodes, if that would change it would be required to change the `config.py` in the corresponding job folder as well. 

When running experiments, there is another important thing to note. The VAD that is used in these experiments, [Silero](https://github.com/snakers4/silero-vad), is a neural network trained to detect any speech activity, which does not allow it to be used in environments with interfering speakers. Therefore a two-stage approach is used. First the desired signal should be played only with the `recording` job(`python manage.py recording run`) and setting the `fileName` attribute in the config file to match the file `path_to_calibration` points to and setting `calibration = True` in [main_playback.py](main_playback.py). In a second stage both then the servers use this file to determine their VAD when both desired and interfering signals are played. To prevent overwriting the dry recordings on the client pc, change the `fileName` attribute for this stage. This will allow to postprocess the recordings later on. (The [postprocessing script](../simulator/postprocess_measurements.py) assumes dry and wet recordings are stored as `recording_dry_{node_number}.wav` and `recording_wet_{node_number}.wav` respectively.)