from __future__ import annotations  # circular dependencies.
from dataclasses import dataclass, field
from .node import DANSENode, batchNode, onlineNode
import numpy as np
import scipy.signal as signal
from tqdm import tqdm
import utils
import warnings


@dataclass
class DANSE_network:
    #
    K: int = 2  # nb nodes in the network
    Mk: int = 10  # nb channels per node
    R: int = 4  # nb channels to communicate
    #
    lFFT: int = 1024
    overlap: float = 0.5
    windowType: str = "sqrt hanning"
    #
    alphaFormat: str = "harmonic"
    alpha0: float = 1
    #
    GEVD: bool = False
    sequential: bool = True
    Gamma: float = 0
    mu: float = 1
    #
    useVAD: bool = False
    vadType: str = "silero"
    #
    fs: float = 8e3
    seed: int = 64
    #
    nodesToTrack: list[int] = field(default_factory=lambda: [0])
    #

    def __post_init__(self):
        """Update derived attributes"""
        np.random.seed(self.seed)

        match self.windowType:
            case "sqrt hanning":
                self.window: np.ndarray = np.sqrt(np.hanning(self.lFFT))
            case "ones":
                self.window: np.ndarray = (
                    1 / np.sqrt(1 / self.overlap) * np.ones(self.lFFT)
                )
            case _:
                warnings.warn("Window type not recognized, using scaled ones instead.")
                self.window: np.ndarray = (
                    1 / np.sqrt(1 / self.overlap) * np.ones(self.lFFT)
                )

        if not self.alphaFormat in ["harmonic", "ones"]:
            warnings.warn(
                "Format to be used for the alphas not recognized, defaulting to 'harmonic'"
            )
            self.alphaFormat = "harmonic"

        for nodeNb in self.nodesToTrack:
            if nodeNb < 0 or nodeNb >= self.K:
                raise ValueError(
                    "Node to track should be a number corresponding to an actual node!"
                )

        self.STFTObj: signal.ShortTimeFFT = signal.ShortTimeFFT(
            self.window,
            int(self.lFFT * (1 - self.overlap)),
            self.fs,
            fft_mode="onesided",
        )

        self.logger: utils.logging.FCLogger = utils.logging.FCLogger(
            self.K, self.Mk, self.R, self.lFFT
        )
        self.nodes: list[DANSENode] = [None for _ in range(self.K)]

    def performDANSE(
        self,
        audio: np.ndarray,
        noise: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        Perform "nIter" iterations of the DANSE algorithm of choice (sequential
        of synchronous, classic or GEVD-based).

        Parameters
        -------------
            audio:      2D array of dimensions ["Mk x K" x "nSamples"]
                        containing the desired contribution to the mics

            noise:      2D array of same shape as "audio", but now with the
                        unwanted contributions.

        Returns
        --------------
            Three lists containing numpy ndarrays: the first one contains the
            network-wide filter equivalents spread over all iterations of the
            desired node. The second one is the contribution of the desired
            signal to the outputs in each node that was tracked and the last one
            is the same but then for the contributions of the noise.
        """
        pass

    def _performDANSESequential(
        self,
        audio: list[np.ndarray],
        noise: list[np.ndarray],
        vad: list[list[np.ndarray]],
        trackOutputs: bool = False,
    ):
        """
        Perform "nIter" updates of the sequential DANSE algorithm.

        After the update of the weights the data is fed into that node again as
        feeding in data automatically refuses it again.
        """
        u = 0  # updating node index
        for i in tqdm(range(len(audio)), desc="Performing iterations", leave=False):
            # update data for all nodes. Make batchmode simulations considerably
            # slower, but only way to ensure correctness for online mode simulations
            for k in range(self.K):
                self.nodes[k].receive(
                    audio[i][:, k * self.Mk : (k + 1) * self.Mk, :],
                    noise[i][:, k * self.Mk : (k + 1) * self.Mk, :],
                )

            for k in range(self.K):
                self.nodes[k].updateAutocorrelations(vad[k][i])

            # update the updating node (sequential hence alpha = 1: default)
            self.nodes[u].update()

            # get the necessary networkwide filters in the logger. If desired to
            # track the outputs of nodes, do so as well (infeasible for batchmode)
            # Also, track the desired signal
            for k in self.nodesToTrack:
                self.logger.computeNetworkWideFilter(k)
                if trackOutputs:
                    self.nodes[k].getOutput()

            # update iteration index
            u = (u + 1) % self.K

    def _performDANSESynchronous(
        self,
        audio: list[np.ndarray],
        noise: list[np.ndarray],
        alphas: np.ndarray,
        vad: list[list[np.ndarray]],
        trackOutputs: bool = False,
    ):
        """
        Perform "nIter" iterations of the simultaneous DANSE algorithm.
        """
        for i in tqdm(range(len(audio)), desc="Performing iterations", leave=False):
            # select an alpha
            alpha = alphas[i]

            # feed data to all nodes
            for k in range(len(self.nodes)):
                audioSlice = audio[i][:, k * self.Mk : (k + 1) * self.Mk, :]
                noiseSlice = noise[i][:, k * self.Mk : (k + 1) * self.Mk, :]

                self.nodes[k].receive(audioSlice, noiseSlice)

            # update all nodes
            for k in range(self.K):
                self.nodes[k].updateAutocorrelations(vad[k][i])
                self.nodes[k].update(alpha)

            # compute the network-wide filter equivalent of nodes to track.
            # If desired to track outputs (non batchmode simulations), do so as
            # well.
            for k in self.nodesToTrack:
                self.logger.computeNetworkWideFilter(k)
                if trackOutputs:
                    self.nodes[k].getOutput()
                    self.nodes[k]._logger.append(
                        "_d", audio[i][:, k * self.Mk : k * self.Mk + self.R, :]
                    )

    def _constructAlphaArray(self, nIter: int) -> np.ndarray:
        match self.alphaFormat:
            case "ones":
                return self.alpha0 * np.ones((nIter))
            case "harmonic":
                return self.alpha0 / (1 + np.arange(nIter))
            case _:
                warnings.warn("Unknown alpha parameter, something went wrong!")
                return self.alpha0 * np.ones((nIter))

    def _computeNetworkWideFilter(self, idx: int) -> np.ndarray:
        """
        Compute the network-wide equivalent of a filter corresponding to a
        specific node. Follows equation (37) in the "DANSE part I: Sequential
        node updating" paper by A. Bertrand where it is used that Gkk = I.
        """
        return self.logger.computeNetworkWideFilter(idx)


@dataclass
class batch_network(DANSE_network):
    #
    K: int = 2  # nb nodes in the network
    Mk: int = 10  # nb channels per node
    R: int = 4  # nb channels to communicate
    #
    lFFT: int = 1024
    overlap: float = 0.5
    windowType: str = "sqrt hanning"
    #
    GEVD: bool = False
    sequential: bool = True
    Gamma: float = 0
    mu: float = 1
    #
    useVAD: bool = False
    vadType: str = "silero"
    #
    fs: float = 8e3
    seed: int = 64
    #
    nodesToTrack: list[int] = field(default_factory=lambda: [0])
    #

    def __post_init__(self):
        super().__post_init__()

        # construct nodes
        self.nodes: list[batchNode] = [None for _ in range(self.K)]
        for k in range(self.K):
            self.nodes[k] = batchNode(
                self.K,
                self.Mk,
                self.R,
                self.lFFT,
                self.overlap,
                self.windowType,
                self.GEVD,
                self.Gamma,
                self.mu,
                self.seed,
            )
            for i in range(k):
                self.nodes[i].link(self.nodes[k])  # link to all previous nodes

        for k in range(len(self.nodes)):
            self.logger.link(k, self.nodes[k])

    def performDANSE(
        self,
        audio: np.ndarray,
        noise: np.ndarray,
        nIter: int,
    ) -> tuple[list[list[np.ndarray]], list[np.ndarray], list[np.ndarray]]:
        # bookkeeping
        if audio.ndim != 2 or noise.ndim != 2:
            raise ValueError("Audio and noise should be 2D arrays")
        if audio.shape != noise.shape:
            raise ValueError("Audio and noise should have the same shape")
        if audio.shape[0] != self.K * self.Mk:
            raise ValueError("The number of channels should be equal to 'Mk' x 'K'")
        if nIter <= 0:
            raise ValueError(
                "The number of iterations should be strictly larger than 0"
            )

        # transform data into the frequency domain, frequency axis in front
        audioFreq = self.STFTObj.stft(audio, axis=1).transpose(1, 0, 2)
        noiseFreq = self.STFTObj.stft(noise, axis=1).transpose(1, 0, 2)

        # segment the data (= duplicate in the case of batchmode simulations)
        # only feasible because these are views in numpy! (else the memory
        # consumption would be huge)
        audioData = [audioFreq for _ in range(nIter)]
        noiseData = [noiseFreq for _ in range(nIter)]

        if self.useVAD:
            vad = [None for _ in range(self.K)]
            for k in range(self.K):
                # Obtain frequency domain vad and segment that too
                vad_fd = utils.vad.transformVAD(
                    utils.vad.computeVAD(
                        audio[k * self.Mk, :], fs=self.fs, type=self.vadType
                    ),
                    self.lFFT,
                    self.overlap,
                )
                vad[k] = [vad_fd for _ in range(nIter)]
        else:
            vad = [[None for _ in range(nIter)] for _ in range(self.K)]

        # pass data to the appropriate handler
        if self.sequential:
            self._performDANSESequential(audioData, noiseData, vad, trackOutputs=False)
        else:
            alphas = self._constructAlphaArray(nIter)
            self._performDANSESynchronous(
                audioData, noiseData, alphas, vad, trackOutputs=False
            )

        # compute the output (infeasible to store all iters in memory because of
        # memory issues). Only the final output interesting for batchmode.
        audioOut = []
        noiseOut = []
        nwFilters = []

        for k in self.nodesToTrack:
            filters = self.logger.getNetworkWideFilters(k)
            nwFilters.append(filters)

            filter_H = np.conj(filters[-1].transpose(0, 2, 1))
            audioTmp = filter_H @ audioFreq
            noiseTmp = filter_H @ noiseFreq

            audioOut.append(
                np.real_if_close(self.STFTObj.istft(audioTmp, f_axis=0, t_axis=2)).T[
                    :, : audio.shape[1]
                ]
            )
            noiseOut.append(
                np.real_if_close(self.STFTObj.istft(noiseTmp, f_axis=0, t_axis=2)).T[
                    :, : audio.shape[1]
                ]
            )

        return nwFilters, audioOut, noiseOut


@dataclass
class online_network(DANSE_network):
    #
    K: int = 2  # nb nodes in the network
    Mk: int = 10  # nb channels per node
    R: int = 4  # nb channels to communicate
    #
    lFFT: int = 1024
    overlap: float = 0.5
    windowType: str = "sqrt hanning"
    #
    alphaFormat: str = "harmonic"
    alpha0: float = 1
    #
    GEVD: bool = False
    sequential: bool = True
    Gamma: float = 0
    mu: float = 1
    #
    useVAD: bool = False
    vadType: str = "silero"
    #
    fs: float = 8e3
    seed: int = 64
    #
    nodesToTrack: list[int] = field(default_factory=lambda: [0])
    #
    updateMode: str = "exponential"
    deltaUpdate: int = int(1e4)
    lmbd: float = 0.999
    #

    def __post_init__(self):
        super().__post_init__()

        # construct nodes
        self.nodes: list[DANSENode] = [None for _ in range(self.K)]
        for k in range(self.K):
            match self.updateMode:
                case "exponential":
                    self.nodes[k] = onlineNode(
                        self.K,
                        self.Mk,
                        self.R,
                        self.lFFT,
                        self.overlap,
                        self.windowType,
                        self.GEVD,
                        self.Gamma,
                        self.mu,
                        self.seed,
                        trackOutput=True,
                        deltaUpdate=self.deltaUpdate,
                        lmbd=self.lmbd,
                    )
                case "windowed":
                    self.nodes[k] = batchNode(
                        self.K,
                        self.Mk,
                        self.R,
                        self.lFFT,
                        self.overlap,
                        self.windowType,
                        self.GEVD,
                        self.Gamma,
                        self.mu,
                        self.seed,
                        trackOutput=True,
                    )
                case _:
                    raise ValueError("Updating mode not recognized!")
            for i in range(k):
                self.nodes[i].link(self.nodes[k])

        # construct logger
        for k in range(self.K):
            self.logger.link(k, self.nodes[k])

        # compute network wide filters prior to anything happening. Can only be
        # done after all linking has been done.
        for k in range(self.K):
            self.logger.computeNetworkWideFilter(k)

    def performDANSE(
        self,
        audio: np.ndarray,
        noise: np.ndarray,
    ) -> tuple[list[list[np.ndarray]], list[np.ndarray], list[np.ndarray]]:
        # bookkeeping
        if audio.ndim != 2 or noise.ndim != 2:
            raise ValueError("Audio and noise should be 2D arrays")
        if audio.shape != noise.shape:
            raise ValueError("Audio and noise should have the same shape")
        if audio.shape[0] != self.K * self.Mk:
            raise ValueError("The number of channels should be equal to 'Mk' x 'K'")

        # transform data into frequency domain
        audioFreq = self.STFTObj.stft(audio, axis=1).transpose(1, 0, 2)
        noiseFreq = self.STFTObj.stft(noise, axis=1).transpose(1, 0, 2)

        # segment the data
        nSegments = int(np.ceil(audioFreq.shape[2] / self.deltaUpdate))

        audioData = [
            audioFreq[:, :, i * self.deltaUpdate : (i + 1) * self.deltaUpdate]
            for i in range(nSegments - 1)
        ]
        noiseData = [
            noiseFreq[:, :, i * self.deltaUpdate : (i + 1) * self.deltaUpdate]
            for i in range(nSegments - 1)
        ]

        audioData.append(audioFreq[:, :, (nSegments - 1) * self.deltaUpdate :])
        noiseData.append(noiseFreq[:, :, (nSegments - 1) * self.deltaUpdate :])

        # Obtain frequency domain vad and segment that too
        if self.useVAD:
            vad = [None for _ in range(self.K)]  # list with vad lists per node
            for k in range(self.K):  # use a per-node vad for increased accuracy
                vad_fd = utils.vad.transformVAD(
                    utils.vad.computeVAD(
                        audio[k * self.Mk, :], fs=self.fs, type=self.vadType
                    ),
                    self.lFFT,
                    self.overlap,
                )
                vad[k] = [
                    vad_fd[i * self.deltaUpdate : (i + 1) * self.deltaUpdate]
                    for i in range(nSegments)
                ]
        else:
            vad = [[None for _ in range(nSegments)] for _ in range(self.K)]

        # pass data to appropriate handler
        if self.sequential:
            self._performDANSESequential(audioData, noiseData, vad, trackOutputs=True)
        else:
            alphas = self._constructAlphaArray(nSegments)
            self._performDANSESynchronous(
                audioData, noiseData, alphas, vad, trackOutputs=True
            )

        nwFilters = []
        audioOut = []
        noiseOut = []
        for k in self.nodesToTrack:
            filters = self.logger.getNetworkWideFilters(k)
            nwFilters.append(filters)

            _, audioTmp, noiseTmp = self.logger.getOutput(k)

            audioOut.append(
                np.real_if_close(self.STFTObj.istft(audioTmp, f_axis=0, t_axis=2)).T[
                    :, : audio.shape[1]
                ]
            )
            noiseOut.append(
                np.real_if_close(self.STFTObj.istft(noiseTmp, f_axis=0, t_axis=2)).T[
                    :, : audio.shape[1]
                ]
            )

        return nwFilters, audioOut, noiseOut

    def getSegmentedOutput(
        self, node: int
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]]:
        """
        Utility function for postprocessing the output of the network.

        Each frame of data that was used to perform one update is stored
        separately in the logger => extract that, transform back to the time
        domain and return the time domain segments alongside the timestamps to
        which they correspond to allow for further postprocessing.

        The only reason this function is presented here is to avoid having to
        construct extra stft objects in scipy.

        Parameters
        ----------
        node: int
            The node for which to collect the outputs


        Returns
        -------
            - A list of numpy arrays containing the time domain contributions of
            the desired signal to the output.

            - A list of numpy arrays containing the time domain contributions of
            interferers and other noise source to the output.

            - A list of numpy arrays containing the time domain desired signal
            during that segment

            - A list of integers indicating the endtime of each segment.
        """
        _, audioF, noiseF, dF = self.logger.getOutput(node, mode="frames")
        audioT = [None for _ in range(len(audioF))]
        noiseT = [None for _ in range(len(audioF))]
        dT = [None for _ in range(len(audioF))]
        timestamps = [None for _ in range(len(audioF))]

        for i in range(len(audioF)):
            # prevent issues with the istft if only one slice.
            if audioF[i].shape[2] > 1:
                audioT[i] = self.STFTObj.istft(audioF[i], f_axis=0, t_axis=2).T
                noiseT[i] = self.STFTObj.istft(noiseF[i], f_axis=0, t_axis=2).T
                dT[i] = self.STFTObj.istft(dF[i], f_axis=0, t_axis=2).T
            else:
                # take ifft, then remove the extra dimension due to the one frame
                audioT[i] = np.fft.irfft(audioF[i], n=self.lFFT, axis=0)[:, :, 0].T
                noiseT[i] = np.fft.irfft(noiseF[i], n=self.lFFT, axis=0)[:, :, 0].T
                dT[i] = np.fft.irfft(dF[i], n=self.lFFT, axis=0)[:, :, 0].T

            # every frame introduces `hop` new samples.
            if audioF[i].shape[2] > 1:  # segment no added value: ISTFT impossible
                timestamps[i] = audioF[i].shape[2] * self.STFTObj.hop + (
                    0 if i == 0 else timestamps[i - 1]
                )
            else:
                timestamps[i] = timestamps[i - 1] if i > 0 else 0

        return audioT, noiseT, dT, timestamps
