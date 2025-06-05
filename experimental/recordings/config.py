# ----------------------------------------------------
# FISSION CONFIG FILE
# ----------------------------------------------------

from fission.core.nodes import BaseNode
from fission.core.pipes import PicklePipe

import os
from datetime import datetime
import shutil

from shared.jobs import recordingJob, collectorJob

from utils.signal_generation import signalParameters

PATH_TO_CFG = "/home/basil-liekens/Msc-Thesis-Danse/code/rpi/experimental-validation/config/cfg.yml"
p = signalParameters().load_from_yaml(PATH_TO_CFG)

now = datetime.now()
p.baseDir = os.path.join(
    p.baseDir,
    f"{now.year:02d}{now.month:02d}{now.day:02d}_setup_3",
)

if not os.path.isdir(p.baseDir):
    os.makedirs(p.baseDir)
shutil.copy(PATH_TO_CFG, os.path.join(p.baseDir, "cfg.yml"))

piName = p.fileName + p.fileType

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
        outputs=[PicklePipe(1)],
        DEFAULT_NODE="192.168.1.100",
        fileName=piName,
        recLen=p.signalLength,
    ),
    recordingJob(
        outputs=[PicklePipe(2)],
        DEFAULT_NODE="192.168.1.102",
        fileName=piName,
        recLen=p.signalLength,
    ),
    recordingJob(
        outputs=[PicklePipe(3)],
        DEFAULT_NODE="192.168.1.103",
        fileName=piName,
        recLen=p.signalLength,
    ),
    collectorJob(
        inputs=[PicklePipe(1)],
        DEFAULT_NODE="192.168.1.101",
        nChannels=p.nChannels,
        basePath=p.baseDir,
        fileName=p.fileName + "_1" + p.fileType,
        recLen=p.signalLength,
    ),
    collectorJob(
        inputs=[PicklePipe(2)],
        DEFAULT_NODE="192.168.1.101",
        nChannels=p.nChannels,
        basePath=p.baseDir,
        fileName=p.fileName + "_2" + p.fileType,
        recLen=p.signalLength,
    ),
    collectorJob(
        inputs=[PicklePipe(3)],
        DEFAULT_NODE="192.168.1.101",
        nChannels=p.nChannels,
        basePath=p.baseDir,
        fileName=p.fileName + "_3" + p.fileType,
        recLen=p.signalLength,
    ),
]

# A list of nodes to be included in the network
# This can be an instance of BaseNode or Multinode
# or a path to csv file defining nodes (see documentation)
NODES = [
    BaseNode("192.168.1.100"),
    BaseNode("192.168.1.102"),
    BaseNode("192.168.1.103"),
]

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
PULSE_INTERVAL = 0.5

# Defines how many bytes the head has.
# First 3 bits are reserved for FISSION
HEAD_SIZE = 1
