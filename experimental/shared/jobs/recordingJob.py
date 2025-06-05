from fission.core.jobs import PythonJob
from fission.core.pipes import BasePipe

from dataclasses import dataclass, field
import numpy as np
import os
import sounddevice as sd
import soundfile as sf
import time
from typing import Generator
import utils.signal_generation


@dataclass
class recordingJob(PythonJob):
    """
    This class is based on the class with the same name in
    `code/rpi/MARVELO-test/shared/jobs.py`, but is now adapted to also have the
    ability to adapt to the ability to match the template (cfr.
    `code/rpi/template-test` for more information)

    This is a so-called `sourcejob`, it has only outputs, no inputs, hence it
    also doesn't have an input attribute.
    """

    #
    recLen: float = 10.0  # [s]
    fs: int = int(16e3)  # capped at 16 kHz for the respeaker mic array!
    fileName: str = "recording.wav"
    basePath: str = os.path.join("/home", "RPi", "installations")
    #
    speakerID: str = "respeaker"  # partial name of the mic, used to identify
    blockSize: int = 512  # nb samples per new chunk
    #
    templateLength: int = 1024  # length of the template
    templateType: str = "MLS"  # type of template to use
    threshold: float = 1  # the threshold to use for indicating the start of a match
    templateMatched: bool = False  # whether or not the template was found
    templateStartIdx: int = -1  # the index of the startpoint (found until now)
    templateMaxCorr: float = -1  # the maximal correlation that crossed the threshold
    #
    desChannels: np.ndarray = field(default_factory=lambda: 1 + np.arange(4))
    #
    outputs: list[BasePipe] = None
    #
    DEFAULT_NODE: str = None
    #

    def __post_init__(self):
        super().__init__(outputs=self.outputs)

    def setup(self, *args, **kwargs):
        """
        Update derived attributes, but done in the "setup" function now since
        the discovery + allocation should happen on the device itself!

        E.g., it is impossible to have the device discovery by sounddevice
        happen on the client. `__init__` happens on the client, `setup` happens
        on the device itself.
        """
        super().setup(*args, **kwargs)
        self.deviceIdx, self.nChannels = self._getMicIndex()

        if np.any(self.desChannels >= self.nChannels) or np.any(self.desChannels < 0):
            raise ValueError(
                f"Mic array has {self.nChannels}, but desired channels "
                f"({self.desChannels}) tries to select non-existing ones!"
            )

        # for storing the recording, overprovision a little to tolerate some
        # overflow around the edges (the total size could be not a multiple of
        # the blocksize)
        self.audiobuffer: np.ndarray = np.zeros(
            (int(self.recLen * self.fs) + 2 * self.blockSize, len(self.desChannels)),
            dtype=np.float32,
        )

        # pointers that allow to work with a sliding window approach to the
        # reading and writing of the audiodata.
        self.insertionIdx: int = 0  # where to insert new data
        self.readIdx: int = 0  # last read sample, determine when to read

        self.sleepTime: float = self.blockSize / self.fs / 10  # efficient spin-wait

        # for the template matching, only use 1 channel for that!
        if self.templateLength < self.blockSize:
            raise ValueError(
                f"More than one block should be accessible for the determination"
                f" of the index! Got template length {self.templateLength} and "
                f"blocksize {self.blockSize}."
            )
        self.template: np.ndarray = utils.signal_generation.constructTemplate(
            self.templateLength, self.fs, type=self.templateType
        )
        self.templateBuffer: np.ndarray = np.zeros(
            (int(3 / 2 * self.templateLength), len(self.desChannels))
        )

        # instantiate the stream
        self.stream: sd.InputStream = sd.InputStream(
            samplerate=self.fs,
            blocksize=self.blockSize,
            device=self.deviceIdx,
            channels=self.nChannels,
            callback=lambda indata, frames, time, status: self._callback(
                indata, frames, time, status
            ),
        )
        self.stream.start()

    def _callback(
        self, indata: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ):
        """
        callback for the audio stream, sounddevice expects this exact signature
        without the `self` argument => to use as a proper callback, use a lambda
        function:
        >>> callback = lambda x, y, z, w: self._callback(self, x, y, w, z)

        This callback is responsible for multiple things: first it should do the
        template matching as described in `code/rpi/tests/template-test`, if
        this is done, the actual recording and sending can happen.
        """
        if status:
            print(status)

        # task dispatching
        if not self.templateMatched:
            self._matchTemplate(np.copy(indata[:, self.desChannels]))
        else:  # template was found, let's start processing data
            self.audiobuffer[  # copy mandatory: a view is returned and overwritten
                self.insertionIdx : self.insertionIdx + self.blockSize, :
            ] = np.copy(indata[:, self.desChannels])
            self.insertionIdx += self.blockSize  # indicate the arrival of a new chunk

    def _getMicIndex(self) -> tuple[int, int]:
        """
        Get the sounddevice index of the desired mic array alongside the number
        of supported channels.
        """
        devices = sd.query_devices()
        inIdcs = [
            idx
            for idx in range(len(devices))
            if self.speakerID in devices[idx]["name"].lower()
        ]
        if len(inIdcs) != 1:
            raise RuntimeError(
                f"{len(inIdcs)} mics found with {self.speakerID} in name, ensure exactly one is connected!"
            )
        return inIdcs[0], devices[inIdcs[0]]["max_input_channels"]

    def _matchTemplate(self, indata: np.ndarray):
        """
        Use the newly incoming data to try and find the exact starting position
        of the template. Code is mostly based on `onlinetest.py` in `code/rpi/
        template-test`.

        Parameters
        ----------
            indata: np.ndarray, 2D
                All data that came in (the channels of interest), the first one
                will be used for template matching.
        """
        self.templateBuffer = np.roll(
            self.templateBuffer, shift=-self.blockSize, axis=0
        )
        self.templateBuffer[-self.blockSize :, :] = indata
        oldStartIdx = self.templateStartIdx

        correlation = np.correlate(self.templateBuffer[:, 0], self.template)
        maxIdx = np.argmax(np.abs(correlation))  # strong negative => good relation

        if np.abs(correlation[maxIdx]) < self.templateMaxCorr and oldStartIdx != -1:
            # indicate that a match was found and start filling the buffer
            self.templateMatched = True

            # put the interesting part in the buffer
            offset = oldStartIdx + self.templateLength
            self.insertionIdx = self.templateBuffer.shape[0] - offset
            self.audiobuffer[: self.insertionIdx, :] = self.templateBuffer[offset:, :]

        elif (
            np.abs(correlation[maxIdx]) > self.templateMaxCorr
            and np.abs(correlation[maxIdx]) > self.threshold
        ):
            self.templateMaxCorr = np.abs(correlation[maxIdx])
            self.templateStartIdx = maxIdx

    def run(self) -> Generator[np.ndarray, None, None]:  # no arguments: is a sourcejob
        """
        The function that is responsible for the actual recording.

        Start the stream, then spin-waits until a new slice is available and if
        that is the case, yields that slice.
        """
        # spin-wait until the actual recording can start
        while not self.templateMatched:
            time.sleep(self.sleepTime)

        startTime = time.perf_counter()
        now = startTime

        # only work for a certain period of time. Check if a full block can be
        # yielded (always yield the same size to prevent issues with different
        # nodes sending different numbers of samples which makes synchronization
        # far harder). Otherwise the first packet will not be full size +
        # from time to time there will be some issues with two packets being
        # sent simultaneously.
        while now - startTime <= self.recLen:
            if self.insertionIdx > self.readIdx + self.blockSize:
                yield len(self.outputs) * tuple(
                    [self.audiobuffer[self.readIdx : self.readIdx + self.blockSize, :]]
                )
                self.readIdx += self.blockSize
            else:  # spin-wait
                time.sleep(self.sleepTime)

            now = time.perf_counter()

        self.stream.stop()

        # cleanup + writeback, don't normalize between wet and dry recordings!
        self.audiobuffer = self.audiobuffer[: int(self.recLen * self.fs), :]
        sf.write(
            os.path.join(self.basePath, self.fileName),
            data=self.audiobuffer,
            samplerate=self.fs,
        )
        self.finish()

    def __eq__(self, other: object):
        return self is other
