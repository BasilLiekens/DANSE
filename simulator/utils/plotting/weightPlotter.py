import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
from scipy import signal
from signal_generation import setup
import time


def plotWeights(W: np.ndarray, filterName: str = "") -> list[plt.Figure]:
    """
    Given a set of weights, make a visualization of the magnitude of the weights
    This is done by looking at each output separately, and making a kind of
    "spectrogram" of it: each bin has only one contribution per output, hence
    they are stacked in a row.

    Parameters
    ----------
        W: np.ndarray, 3D
            An ndarray containing the weights, should be of shape
            [`lFFT` x `number output channels` x `number inputs`] or
            [`lFFT // 2 + 1` x `number input channels` x `number outputs`]
            depending on whether or not single-sided fft's were used.

        filterName: str, optional
            The name to add to the title of the plots, defaults to "".

    Returns
    -------
        A list of figures, one for each output, with the filter visualizations.
    """
    # bookkeeping
    if W.ndim != 3:
        raise ValueError("Input should be 3D!")

    figs = []
    for i in range(W.shape[2]):
        outputMagnDB = 20 * np.log10(np.abs(W[:, :, i]))

        fig, ax = plt.subplots()
        fig.set_size_inches(8.5, 5.5)
        img = ax.pcolormesh(outputMagnDB, cmap="plasma", vmin=-40, vmax=0)
        cbar = fig.colorbar(img)
        ax.set(
            xlabel="inputs",
            ylabel="frequency bins",
            title=f"Magnitude of the weights of {filterName} filter (output {i+1})",
        )
        cbar.set_label("magnitude [dB]")
        ax.autoscale(tight=True, axis="x")
        fig.tight_layout()
        figs.append(fig)

    return figs


class spatialResponse:
    def __init__(
        self,
        room: pra.ShoeBox,
        p: setup.Parameters,
        STFTObj: signal.ShortTimeFFT,
        granularity: float = 0.1,
    ):
        """
        Given a (set of) filters, and the parameters of the room, generate a
        meshgrid of points in the room and plot the spatial response w.r.t each
        node in the room. One plot is generated for each node in the room.
        The setup just precomputes the RIR to avoid having to recompute them for
        every new plot.

        Parameters
        -----------
            room: pra.Shoebox
                The pyroomacoustics ShoeBox used to generate the RIRs in the
                first place (hence should contain all mics).

            p: setup.Parameters
                The parameters used to generate the original room (the room
                object itself doesn't contain enough information to reconstruct
                it and neither is it possible to make a copy of the thing).

            STFTObj: signal.ShortTimeFFT
                The object used to perform (I)STFTs, used to perform the
                computations in the frequency domain and do WOLA processing to
                compute the magnitude response.

            granularity: float, optional
                float indicating the spacing between mics [m]. Defaults to 0.1.
        """
        self.p = p
        self.room = room
        self.STFTObj = STFTObj

        self.lFFT = p.lFFT
        self.nMics = p.K * p.Mk
        self.nOutputs = p.R

        RIRs, sourcelocs = self._generateRoomRIRs(granularity)
        self.RIRs = RIRs
        self.sourcelocs = sourcelocs

        # put the individual point in front to mimic the natural WOLA processing done
        self.RIRsFreq = self.STFTObj.stft(self.RIRs, axis=0).transpose(2, 0, 1, 3)
        self.extent = [
            np.min(sourcelocs[0]),
            np.max(sourcelocs[0]),
            np.min(sourcelocs[1]),
            np.max(sourcelocs[1]),
        ]

    def plotFilterResponse(
        self, filter: np.ndarray, filterType: str = ""
    ) -> list[plt.Figure]:
        """
        Given a (onesided) frequency domain description of a filter (dimensions
        ["lFFT" / 2 + 1 x "nb mics" x "nb output channels"]), plot the spatial
        response for each of the output channels separately.

        Parameters
        ----------------
            filter:     the filter itself. Should be a 3D numpy array with
                        dimensions ["lFFT" / 2 + 1 x "nb mics" x "nb output channels"].


            filterType: the name to add to the plot.

        Returns
        ----------------
            a list with spatial responses for each filter.
        """
        ## bring the single-sided filter to full length + take hermitian
        W_H = np.conj(filter.transpose(0, 2, 1))

        ## compute the responses at each point + go back to time domain
        filterResponse = W_H @ self.RIRsFreq
        tdResponse = self.STFTObj.istft(filterResponse, f_axis=1, t_axis=3).transpose(
            0, 2, 1
        )
        magnResponse = np.sum(np.square(np.abs(tdResponse)), axis=2)
        magnResponse = 10 * np.log10(magnResponse)  # turn into dB

        figs = [
            self._plotIndividualFilter(
                magnResponse[:, x], vmin=-20, vmax=15, filterType=filterType, number=x
            )
            for x in range(magnResponse.shape[1])
        ]
        return figs

    def _plotIndividualFilter(
        self,
        magnResponse: np.ndarray,
        vmin: float,
        vmax: float,
        filterType: str = "",
        number: int = 0,
    ) -> plt.Figure:
        """
        plot the response for one particular output
        """
        fig, ax = plt.subplots()
        fig.set_size_inches(8.5, 5.5)
        scatter = ax.imshow(
            magnResponse.reshape((self.sourcelocs.shape[1], self.sourcelocs.shape[2])),
            extent=self.extent,
            vmin=vmin,
            vmax=vmax,
            cmap="plasma",
            interpolation="bilinear",
            origin="lower",
        )
        # add sound sources
        for i in range(self.p.Ns):
            pos = self.room.sources[i].position
            ax.scatter(pos[0], pos[1], marker="x", color="w")
            ax.annotate("Sound source", (pos[0], pos[1]), color="w")
        # add noise sources
        for i in range(self.p.Nn):
            pos = self.room.sources[self.p.Ns + i].position
            ax.scatter(pos[0], pos[1], marker="x", color="w")
            ax.annotate("Noise source", (pos[0], pos[1]), color="w")

        ax.set(xlabel="x [m]", ylabel="y [m]")
        ax.autoscale(tight=True)
        ax.set_title(f"Spatial response of the {filterType} filter")

        cb = fig.colorbar(scatter)
        cb.set_label("Magnitude [dB]")
        fig.tight_layout()

        return fig

    def _generateRoomRIRs(self, granularity: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a copy of the room (same dimenions and microphone positions) as the
        one that was used for the simulations, but generate a grid of sources in
        there to be able to generate a spatial response plot of the filter
        Parameters
        ------------
            room:   the pra ShoeBox used to generate the initial scenario, will be
                    used to have the exact microphone positions.

            p:      the parameters that were used to construct the room in the first
                    place.
        Returns
        -----------
            a numpy array of ["length RIR" x "nb Mics" x "nb Sources"] containing
            the RIRs from each source position to all mics. Alongside this the
            meshgrid with all positions of the microphones is returned.
        """
        mics = self.room.mic_array

        if self.p.t60 == 0:
            maxOrd, eAbs = 0, 0.5
        else:
            eAbs, maxOrd = pra.inverse_sabine(self.p.t60, self.p.rd)

        newRoom = pra.ShoeBox(
            p=self.p.rd,
            fs=self.p.fs,
            max_order=maxOrd,
            air_absorption=False,
            materials=pra.Material(eAbs),
            temperature=self.p.temperature,
            humidity=self.p.humidity,
        )

        # sources are spaced 10 cm apart
        sourcelocs1 = np.linspace(
            0, self.p.rd[0], num=int(np.floor(self.p.rd[0] / granularity))
        )
        sourcelocs2 = np.linspace(
            0, self.p.rd[1], num=int(np.floor(self.p.rd[1] / granularity))
        )

        newRoom.add_microphone_array(mics)
        sources = np.meshgrid(sourcelocs1, sourcelocs2)
        sourcelocs = np.array(sources).reshape((2, -1)).T

        # add a source in each position
        for i in range(sourcelocs.shape[0]):
            newRoom.add_source(sourcelocs[i, :])

        t0 = time.time()
        newRoom.compute_rir()
        print(f"RIRs computed in {time.time() - t0:.2f} s.")
        computed_rirs = newRoom.rir

        # only consider the first lFFT samples, should be enough...
        RIRs = np.zeros([self.p.RIR_length, newRoom.n_mics, sourcelocs.shape[0]])
        for i in range(RIRs.shape[1]):
            for j in range(RIRs.shape[2]):
                RIR_len = len(computed_rirs[i][j])
                if RIR_len > self.p.RIR_length:
                    RIRs[:, i, j] = computed_rirs[i][j][: self.p.RIR_length]
                else:
                    RIRs[:RIR_len, i, j] = computed_rirs[i][j]

        return RIRs, np.array(sources)
