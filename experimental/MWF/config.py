# ----------------------------------------------------
# FISSION CONFIG FILE
# ----------------------------------------------------

from fission.core.nodes import BaseNode
from fission.core.pipes import PicklePipe

from shared.jobs import recordingJob, MWFJob, collectorJob

import os
from utils.signal_generation import signalParameters

PATH_TO_CONFIG = os.path.join(os.getcwd(), "config", "cfg.yml")
p = signalParameters().load_from_yaml(PATH_TO_CONFIG)

# Enter the user on the remote machines
USER = "RPi"

# The directory all the action is done on the remote machines
REMOTE_ROOT = f"/home/{USER}/fission/"

# Logging
LOG_LEVEL = "DEBUG"
LOG_FILE = "FISSION.log"

# Enter the clients ip within the network, must be visible for nodes
CLIENT_IP = "192.168.1.101"

# Debug window
# Redirects stdout to console
DEBUG_WINDOW = True

# A list of jobs to be executed within the network.
# This can be an instance of BaseJob or any of its subclasses.
JOBS = [
    recordingJob(
        recLen=p.signalLength,
        speakerID=p.speakerID,
        desChannels=p.desChannels,
        DEFAULT_NODE="192.168.1.102",
        outputs=[PicklePipe(1), PicklePipe(2), PicklePipe(3)],
    ),
    recordingJob(
        recLen=p.signalLength,
        speakerID=p.speakerID,
        desChannels=p.desChannels,
        DEFAULT_NODE="192.168.1.103",
        outputs=[PicklePipe(4)],
    ),
    MWFJob(
        nChannels=p.nChannels,
        R=p.R,
        lFFT=p.lFFT,
        overlap=p.overlap,
        windowType=p.windowType,
        fs=p.fs,
        vadType=p.vadType,
        deltaUpdate=p.deltaUpdate,
        lmbd=p.lmbd,
        GEVD=p.GEVD,
        Gamma=p.Gamma,
        mu=p.mu,
        path_to_calibration=p.path_to_calibration,
        DEFAULT_NODE="192.168.1.102",
        inputs=[PicklePipe(2), PicklePipe(4)],
        outputs=[PicklePipe(5)],
    ),
    MWFJob(
        nChannels=p.nChannels,
        R=p.R,
        lFFT=p.lFFT,
        overlap=p.overlap,
        windowType=p.windowType,
        fs=p.micFs,
        vadType=p.vadType,
        deltaUpdate=p.deltaUpdate,
        lmbd=p.lmbd,
        GEVD=p.GEVD,
        Gamma=p.Gamma,
        mu=p.mu,
        path_to_calibration=p.path_to_calibration,
        DEFAULT_NODE="192.168.1.102",
        inputs=[PicklePipe(3)],
        outputs=[PicklePipe(6)],
    ),
    collectorJob(
        recLen=p.signalLength,
        fs=p.micFs,
        inputs=[PicklePipe(1)],
        nChannels=4,
        fileName="recording2.wav",
    ),
    collectorJob(
        recLen=p.signalLength,
        fs=p.micFs,
        inputs=[PicklePipe(5)],
        nChannels=1,
        fileName="centralMWFed.wav",
    ),
    collectorJob(
        recLen=p.signalLength,
        fs=p.micFs,
        inputs=[PicklePipe(6)],
        nChannels=1,
        fileName="localMWFed.wav",
    ),
]

# A list of nodes to be included in the network
# This can be an instance of BaseNode or Multinode
# or a path to csv file defining nodes (see documentation)
NODES = [BaseNode("192.168.1.102"), BaseNode("192.168.1.103")]

# Whether or not files (dependencies) for every job should be copied
# to every node. If not files will be copied before the job starts.
# Turning it on results in higher setup time when starting the network
# but reduces the delay in case of failover. It also demands more disk
# space on the nodes.
PRE_COPY = False

# Optional XML file defining the whole network or a part of it
# This is not recommended and only exists for specific usecases
XML_FILE = None

# Defines how often you servers send a heartbeat to the client.
# When 5 heartbeats are missed a node is presumed to be dead.
# This meas `PULSE_INTERVAL` * 5 is the time to detect a failed node.
# Must be between 0.1 and 1000
PULSE_INTERVAL = 0.1

# Defines how many bytes the head has.
# First 3 bits are reserved for FISSION
HEAD_SIZE = 1
