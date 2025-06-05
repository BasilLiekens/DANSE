# ----------------------------------------------------
# FISSION CONFIG FILE
# ----------------------------------------------------

from fission.core.nodes import BaseNode
from fission.core.pipes import PicklePipe

import os
from datetime import datetime
import shutil

from shared.jobs import recordingJob, collectorJob, fusionJob, DANSEJob
import utils

PATH_TO_CFG = "config/cfg.yml"
p = utils.signal_generation.signalParameters().load_from_yaml(PATH_TO_CFG)

now = datetime.now()
p.baseDir = os.path.join(
    p.baseDir,
    f"{now.year:02d}{now.month:02d}{now.day:02d}_setup_3",
)

if not os.path.isdir(p.baseDir):
    os.makedirs(p.baseDir)
shutil.copy(PATH_TO_CFG, os.path.join(p.baseDir, "cfg.yml"))

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

# A list of nodes to be included in the network
# This can be an instance of BaseNode or Multinode
# or a path to csv file defining nodes (see documentation)
NODES = [
    BaseNode("192.168.1.100"),
    BaseNode("192.168.1.102"),
    BaseNode("192.168.1.103"),
]

# A list of jobs to be executed within the network.
# This can be an instance of BaseJob or any of its subclasses.
# Beware! The outputs of "fusionJob" are heterogeneous! The first is all of the
# frequency domain data whereas the latter ones are the fused data.
JOBS = [
    recordingJob(
        p.signalLength,
        p.micFs,
        p.fileName + p.fileType,
        speakerID=p.speakerID,
        DEFAULT_NODE="192.168.1.100",
        outputs=[PicklePipe(1), PicklePipe(2)],
    ),
    recordingJob(
        p.signalLength,
        p.micFs,
        p.fileName + p.fileType,
        speakerID=p.speakerID,
        DEFAULT_NODE="192.168.1.102",
        outputs=[PicklePipe(3), PicklePipe(4)],
    ),
    recordingJob(
        p.signalLength,
        p.micFs,
        p.fileName + p.fileType,
        speakerID=p.speakerID,
        DEFAULT_NODE="192.168.1.103",
        outputs=[PicklePipe(5), PicklePipe(6)],
    ),
    fusionJob(
        Mk=p.nChannels,
        R=p.R,
        lFFT=p.lFFT,
        overlap=p.overlap,
        windowType=p.windowType,
        fs=p.micFs,
        vadType=p.vadType,
        inputs=[PicklePipe(2)],
        outputs=[PicklePipe(7), PicklePipe(8), PicklePipe(9)],
        DEFAULT_NODE="192.168.1.100",
    ),
    fusionJob(
        Mk=p.nChannels,
        R=p.R,
        lFFT=p.lFFT,
        overlap=p.overlap,
        windowType=p.windowType,
        fs=p.micFs,
        vadType=p.vadType,
        inputs=[PicklePipe(4)],
        outputs=[PicklePipe(10), PicklePipe(11), PicklePipe(12)],
        DEFAULT_NODE="192.168.1.102",
    ),
    fusionJob(
        Mk=p.nChannels,
        R=p.R,
        lFFT=p.lFFT,
        overlap=p.overlap,
        windowType=p.windowType,
        fs=p.micFs,
        vadType=p.vadType,
        inputs=[PicklePipe(6)],
        outputs=[PicklePipe(13), PicklePipe(14), PicklePipe(15)],
        DEFAULT_NODE="192.168.1.103",
    ),
    DANSEJob(
        Mk=p.nChannels,
        R=p.R,
        lFFT=p.lFFT,
        windowType=p.windowType,
        fs=p.micFs,
        vadType=p.vadType,
        deltaUpdate=p.deltaUpdate,
        lmbd=p.lmbd,
        GEVD=p.GEVD,
        Gamma=p.Gamma,
        mu=p.mu,
        sequential=p.sequential,
        nodeNb=0,
        alpha0=p.alpha0,
        alphaFormat=p.alphaFormat,
        seed=p.seed,
        inputs=[PicklePipe(7), PicklePipe(11), PicklePipe(14)],
        outputs=[PicklePipe(16)],
        DEFAULT_NODE="192.168.1.100",
        DEPENDENCIES=[utils],
        basePath=p.baseDir_pi,
    ),
    DANSEJob(
        Mk=p.nChannels,
        R=p.R,
        lFFT=p.lFFT,
        windowType=p.windowType,
        fs=p.micFs,
        vadType=p.vadType,
        deltaUpdate=p.deltaUpdate,
        lmbd=p.lmbd,
        GEVD=p.GEVD,
        Gamma=p.Gamma,
        mu=p.mu,
        sequential=p.sequential,
        nodeNb=1,
        alpha0=p.alpha0,
        alphaFormat=p.alphaFormat,
        seed=p.seed,
        inputs=[PicklePipe(10), PicklePipe(8), PicklePipe(15)],
        outputs=[PicklePipe(17)],
        DEFAULT_NODE="192.168.1.102",
        DEPENDENCIES=[utils],
        basePath=p.baseDir_pi,
    ),
    DANSEJob(
        Mk=p.nChannels,
        R=p.R,
        lFFT=p.lFFT,
        windowType=p.windowType,
        fs=p.micFs,
        vadType=p.vadType,
        deltaUpdate=p.deltaUpdate,
        lmbd=p.lmbd,
        GEVD=p.GEVD,
        Gamma=p.Gamma,
        mu=p.mu,
        sequential=p.sequential,
        nodeNb=2,
        alpha0=p.alpha0,
        alphaFormat=p.alphaFormat,
        seed=p.seed,
        inputs=[PicklePipe(13), PicklePipe(9), PicklePipe(12)],
        outputs=[PicklePipe(18)],
        DEFAULT_NODE="192.168.1.103",
        DEPENDENCIES=[utils],
        basePath=p.baseDir_pi,
    ),
    collectorJob(
        p.micFs,
        p.signalLength,
        p.nChannels,
        p.baseDir,
        p.fileName + "_1" + p.fileType,
        inputs=[PicklePipe(1)],
    ),
    collectorJob(
        p.micFs,
        p.signalLength,
        p.R,
        p.baseDir,
        "DANSE_output_1.wav",
        inputs=[PicklePipe(16)],
    ),
    collectorJob(
        p.micFs,
        p.signalLength,
        p.nChannels,
        p.baseDir,
        p.fileName + "_2" + p.fileType,
        inputs=[PicklePipe(3)],
    ),
    collectorJob(
        p.micFs,
        p.signalLength,
        p.R,
        p.baseDir,
        "DANSE_output_2.wav",
        inputs=[PicklePipe(17)],
    ),
    collectorJob(
        p.micFs,
        p.signalLength,
        p.nChannels,
        p.baseDir,
        p.fileName + "_3" + p.fileType,
        inputs=[PicklePipe(5)],
    ),
    collectorJob(
        p.micFs,
        p.signalLength,
        p.R,
        p.baseDir,
        "DANSE_output_3.wav",
        inputs=[PicklePipe(18)],
    ),
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
PULSE_INTERVAL = 0.5  # increase for more stability.

# Defines how many bytes the head has.
# First 3 bits are reserved for FISSION
HEAD_SIZE = 1
