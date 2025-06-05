from dataclasses import dataclass, field
import DANSE_base as DANSE
import numpy as np
from signal_generation.setup import Parameters
from utils import vad, metrics


@dataclass
class simulation:
    """
    Wrapper class for convenience of launching single experiments.

    Constructs the network internally and then allows to start start the
    simulations. This class will also be used to do the postprocessing such that
    the main script should only collect all data and store it.
    """

    #
    p: Parameters = field(default_factory=Parameters())
    #

    def __post_init__(self):
        self.network: DANSE.online_network = DANSE.online_network(
            self.p.K,
            self.p.Mk,
            self.p.R,
            self.p.lFFT,
            self.p.overlap,
            self.p.window,
            self.p.alphaFormat,
            self.p.alpha0,
            self.p.GEVD,
            self.p.sequential,
            self.p.Gamma,
            self.p.mu,
            self.p.useVAD,
            self.p.vadType,
            self.p.fs,
            self.p.seed,
            nodesToTrack=list(range(self.p.K)),  # always track all nodes!
            updateMode=self.p.updateMode,
            deltaUpdate=self.p.deltaUpdate,
            lmbd=self.p.lmbd,
        )

    def launch(
        self, audio: np.ndarray, noise: np.ndarray, metrics: dict[str, list[str]]
    ) -> tuple[dict[str, float | list[float]], list[int]]:
        """
        Launch a simulation with inputs `audio` and `noise`. Return the segments
        corresponding to updates alongside the timestamps to allow for updating
        the datacontainer.

        The centralized, batchmode, MWF as well as the local counterpart are
        also computed to allow for incorporating MSE_w in the metrics later on.

        Arguments
        ---------
        audio: [`Mk` * `K`  x `nSamples`] np.ndarray
            The audio used as the input to the signals (not split into channel
            contributions!)

        noise: [`Mk` * `K` x `nSamples`] np.ndarray
            Same as audio but now for the noise and interfering contributions

        metrics: dict[str, list[str]]
            A list containing a combination of `type` (which set of filters to
            use for the computation) and a list of `metrics` to compute for that
            set of filters.

        Returns
        -------
        audio: dict[int, list[np.ndarray]]
            The audio contributions to the output (in segments per node)

        noise: dict[int, list[np.ndarray]]
            The noise contributions to the output (in segments per node)

        d: dict[int, list[np.ndarray]]
            The desired signal for each node (in segments per node)

        NW_filters: dict[int, list[np.ndarray]]
            Per node network-wide equivalent filters

        LS_cost_centr: dict[int, float]
            Per node LS costs of the centralized MWF's

        LS_cost_loc: dict[int, float]
            Per node LS costs of the local MWF's

        central_MWF: dict[int, np.ndarray]
            Per node contralized MWF's
        """
        # perform DANSE
        self.network.performDANSE(audio, noise)

        # prepare for computing the metrics
        audioF = self.network.STFTObj.stft(audio, axis=1).transpose(1, 0, 2)
        noiseF = self.network.STFTObj.stft(noise, axis=1).transpose(1, 0, 2)
        centralized_MWF, local_MWF = self._computeBatchFilters(audio, noise)

        computedMetrics: dict[str, float | list[float]] = dict()

        for k in range(self.p.K):
            W_nw = self.network.logger.getNetworkWideFilters(k)

            W_centr = centralized_MWF[k]
            W_loc = local_MWF[k]
            # compute the metrics
            for type in metrics:
                computedData = self._computeMetrics(
                    type,
                    metrics[type],
                    audioF,
                    noiseF,
                    W_centr,
                    W_loc,
                    W_nw,
                    k,
                    audio.shape[1],
                )

                for metric, data in computedData.items():
                    computedMetrics[f"{metric}_{type}_{k}"] = data

        # compute the timestamps.
        timestamps = np.minimum(  # t=0 also included now!
            [
                i * self.network.STFTObj.hop * self.p.deltaUpdate
                for i in range(len(W_nw))
            ],
            audio.shape[1],
        )
        # transform to a list with a normal int to prevent issues with the dataframe
        timestamps = [int(timestamps[i]) for i in range(len(timestamps))]

        return computedMetrics, timestamps

    def _computeMetrics(
        self,
        type: str,
        metr: list[str],
        audio: np.ndarray,
        noise: np.ndarray,
        W_centr: np.ndarray,
        W_loc: np.ndarray,
        W_nw: list[np.ndarray],
        nodeNb: int,
        sigLen: int,
    ) -> dict[str, float | list[float]]:
        """
        Given a metric `type` and the desired inputs, compute the metric.
        This is a dispatching function that might take in parameters than needed
        as it has to provision for all possible cases.

        !!Metrics are computed in `batchmode`!! The metrics are computed over
        the full signal for every iteration instead of just on the next segment
        (which is where they are used), this is to enable a fair comparison
        between different filters and remove any potential influence of a "bad"
        distribution in a specific segment.

        Parameters
        ----------
        type: str
            The type of metric to compute: "nw" for network-wide, "centr" for
            the centralized solution and "loc" for the localized solution. It
            should be noted that some combinations are not really sensible or
            will downright crash (e.g. computing MSE_w for other options than
            "nw")

        metr: str
            The metric to compute for this set of filters: `LS_cost`, `MSE_w`,
            `SINR` and `STOI` are currently supported. Should the computation of
            `MSE_w` take too much time, it is also possible to rely on the
            autocorrelation matrix of the signal:
            SINR = 10 * log( (W^H @ Rss @ W) / (W^H @ Rnn @ W)).
            This way, the only thing that is required is the covariance matrix
            which would only have to be computed once. However, this is not
            formally investigated: in the time domain it is certainly possible,
            in the frequency domain statistics isn't your biggest friend (double
            filtering)

        audio: np.ndarray, 3D
            The contribution of the desired signal to the input in each channel
            Should be in the frequency domain, shape [`nBins`x`nChannels`x`nFrames`]

        noise: np.ndarray, 3D
            The same as `audio`, but now for the contribution of interferers and
            other forms of noise.

        W_centr: np.ndarray, 3D
            The centralized MWF filter

        W_loc: np.ndarray, 3D
            The localized MWF filter

        W_nw: list[np.ndarray], each 3D
            The list of network-wide filters over time.

        nodeNb: int
            The node of interest, needed to determine the desired signal

        sigLen: int
            The length of the desired signal (needed for the ISTFTs: all input
            is in the frequency domain to prevent having to compute the same
            STFT all the time).

        Returns
        -------
        A dictionary with as keys the metric and as values either a list of
        floats if the `metric == "nw"`, else the value is a single float.
        """
        STFTObj = self.network.STFTObj

        # prepare to compute the metric depending on the type
        match type:
            case "nw":
                W: list[np.ndarray] = W_nw
                e1: np.ndarray = np.zeros((self.p.K * self.p.Mk, self.p.R))
                e1[nodeNb * self.p.Mk : (nodeNb + 1) * self.p.Mk, :] = (
                    self.network.nodes[nodeNb].E[: self.p.Mk, :]
                )
                s: np.ndarray = audio
                n: np.ndarray = noise

            case "centr":
                W: list[np.ndarray] = [W_centr]
                e1: np.ndarray = np.zeros((self.p.K * self.p.Mk, self.p.R))
                e1[nodeNb * self.p.Mk : (nodeNb + 1) * self.p.Mk, :] = (
                    self.network.nodes[nodeNb].E[: self.p.Mk, :]
                )
                s: np.ndarray = audio
                n: np.ndarray = noise

            case "loc":
                W: list[np.ndarray] = [W_loc]
                e1: np.ndarray = self.network.nodes[nodeNb].E[: self.p.Mk, :]
                s: np.ndarray = audio[
                    :, nodeNb * self.p.Mk : (nodeNb + 1) * self.p.Mk, :
                ]
                n: np.ndarray = noise[
                    :, nodeNb * self.p.Mk : (nodeNb + 1) * self.p.Mk, :
                ]

            case _:
                raise ValueError(f"Unrecognized type found: {type}")

        d = e1.T @ STFTObj.istft(s, f_axis=0, t_axis=2).T[:, :sigLen]

        # conditionally compute the filtered signals: expensive, hence not
        # really intersting to compute all the time.
        lFFT, _, R = W[0].shape  # lFFT//2 + 1 for one-sided ffts!
        if "SINR" in metr:
            # stack for faster ISTFTs
            sStacked = np.zeros((lFFT, R * len(W), s.shape[2]), dtype=np.complex128)
            nStacked = np.zeros_like(sStacked)

            for i in range(len(W)):
                W_H = np.conj(W[i].transpose(0, 2, 1))
                sStacked[:, i * R : (i + 1) * R, :] = W_H @ s
                nStacked[:, i * R : (i + 1) * R, :] = W_H @ n

            s_td = STFTObj.istft(sStacked, f_axis=0, t_axis=2).T[:, : d.shape[1]]
            n_td = STFTObj.istft(nStacked, f_axis=0, t_axis=2).T[:, : d.shape[1]]

            # desegment
            sHat = [s_td[i * R : (i + 1) * R, :] for i in range(len(W))]
            nHat = [n_td[i * R : (i + 1) * R, :] for i in range(len(W))]

            dHat: list[np.ndarray] = [sHat[i] + nHat[i] for i in range(len(sHat))]

        if ("LS_cost" in metr or "STOI" in metr) and "SINR" not in metr:
            # stack for faster ISTFTs
            dStacked = np.zeros((lFFT, R * len(W), s.shape[2]), dtype=np.complex128)
            for i in range(len(W)):
                W_H = np.conj(W[i].transpose(0, 2, 1))
                dStacked[:, i * R : (i + 1) * R, :] = W_H @ (s + n)

            dHat_td = STFTObj.istft(dStacked, f_axis=0, t_axis=2).T[:, : d.shape[1]]
            dHat = [dHat_td[i * R : (i + 1) * R, :] for i in range(len(W))]

        # actually a bit of a waste to compute this for both the case of
        # network-wide and centralized, but impact should be limited + solution
        # requires a work-around where the computation of the signals and the
        # metrics happens in two separate stages.
        if "MSE_d" in metr:  # only relevant in the case of network-wide filters
            dHat_centr_freq = np.conj(W_centr.transpose(0, 2, 1)) @ (s + n)
            dHat_centr_td = STFTObj.istft(dHat_centr_freq, f_axis=0, t_axis=2).T[
                :, : d.shape[1]
            ]

        # actually compute the metrics
        toRet = dict()
        for metric in metr:
            match metric:
                case "LS_cost":
                    data: list[float] = metrics.LScost(d, dHat)

                case "MSE_d":
                    if type != "nw":
                        raise ValueError(
                            "Not sensible or impossible to compute MSE_d for non-networkwide solution!"
                        )
                    data: list[float] = metrics.LScost(dHat_centr_td, dHat)

                case "MSE_w":
                    if type != "nw":
                        raise ValueError(
                            "Not sensible or impossible to compute MSE_w for a non-networkwide solution!"
                        )
                    data: list[float] = metrics.MSE_w(W_centr, W)

                case "SINR":
                    dHat = [sHat[i] + nHat[i] for i in range(len(sHat))]
                    data: list[float] = metrics.SINR(dHat, sHat, nHat)

                case "STOI":
                    data: list[float] = metrics.computeSTOI(d, dHat, self.p.fs)

                case _:
                    raise ValueError("Unknown metric found to compute!")

            # add data to the dict
            if type == "nw":
                toRet[metric] = data
            else:  # unpack the list to return a "clean float"
                toRet[metric] = data[0]

        return toRet

    def _computeBatchFilters(
        self, audio: np.ndarray, noise: np.ndarray
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """
        Given `audio` and `noise`, compute the centralized and local solutions
        for each node.

        Parameters
        ----------
        audio: np.ndarray, 2D
            The contributions of the desired signal to each channel, should be
            of shape [`nChannels`x`nSamples`].
        noise: np.ndarray, 2D
            Similar to `audio`, but now for the contribution of the noise and
            any potential interferers.

        Returns
        -------
        Two dictionaries, each with the keys being the node number to which the
        filter belongs. The values are the filters for that node. The first
        dictionary contains the centralized filter, the second one the local one.
        """
        centralized_MWF: dict[int, np.ndarray] = dict()
        local_MWF: dict[int, np.ndarray] = dict()

        for k in range(self.p.K):
            # compute centralized solution (assume the "E" does not make use of
            # the fused signals from other nodes). Just use the first signal
            # to compute the vad on.
            desVAD = (
                vad.computeVAD(
                    audio[k * self.p.Mk, :], fs=self.network.fs, type=self.p.vadType
                )
                if self.p.useVAD
                else None
            )
            local_E = self.network.nodes[k].E[: self.p.Mk, :]
            central_E = np.zeros((self.p.K * self.p.Mk, self.p.R))
            central_E[k * self.p.Mk : (k + 1) * self.p.Mk, :] = local_E

            local_MWF[k], _, _ = DANSE.MWF_fd(
                audio[k * self.p.Mk : (k + 1) * self.p.Mk, :],
                noise[k * self.p.Mk : (k + 1) * self.p.Mk, :],
                local_E,
                self.network.STFTObj,
                GEVD=self.p.GEVD,
                Gamma=self.p.Gamma,
                mu=self.p.mu,
                vad=desVAD,
            )
            centralized_MWF[k], _, _ = DANSE.MWF_fd(
                audio,
                noise,
                central_E,
                self.network.STFTObj,
                GEVD=self.p.GEVD,
                Gamma=self.p.Gamma,
                mu=self.p.mu,
                vad=desVAD,
            )

        return centralized_MWF, local_MWF
