import numpy as np
import scipy.linalg as spla
import scipy.signal as signal
from tqdm import tqdm
import utils.vad
import warnings


def MWF_fd(
    audio: np.ndarray,
    noise: np.ndarray,
    e1: np.ndarray,
    STFTObj: signal.ShortTimeFFT,
    GEVD: bool = False,
    Gamma: float = 0,
    mu: float = 1,
    vad: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given the audio and noise (=output of "create_micsigs") alongside a
    selection matrix indicating what audio signals are of interest, construct
    the MWF and apply it to the desired signals and the noise separately.

    This is the frequency domain version of the MWF, hence the data is first
    transformed into the frequency domain by means of the STFT before it is
    actually applied. In the STFT domain a WOLA procedure is applied to
    computing the actual samples.

    Parameters
    -----------
        audio: np.ndarray, 2D
            The influence of the desired signal on each mic.
            Should be of shape ["nb Mics" x "signal length"]

        noise: np.ndarray, 2D
            Similar to the "audio" parameter, but now for the noise
            contributions, both spatial noise sources (separated), and any
            potential measurement noise is also added into this array.

        e1: np.ndarray, 2D
            Selection matrix indicating what signals are of interest for the
            specific node. Should be the "E" matrix as denoted in the original
            DANSE paper by A. Bertrand.

        STFTObj: signal.ShortTimeFFT
            The ShortTimeFFT object that should be used for computing the stft
            and istft, this way the window etc. should no longer be passed in
            separately and consistency is guaranteed.

        GEVD: bool, optional
            Whether or not to use a GEVD-based computation for the filter
            computation. Defaults to False.

        Gamma: float, optional
            A regularization constant to be used during the computation of the
            MWF (L2 regularization): would be added to both Rnn and Ryy, meaning
            that during the update $W = Ryy^{-1} (Ryy - Rnn)E$ only the Ryy^{-1}
            term is affected. However, see the toy examples for a demonstration
            of why it could be important: else stability would purely depend on
            the amount of self-noise. Defaults to 0.

        mu: float, optional
            A constant that allows to focus more on keeping the speech signal
            intact (0 <= mu < 1), or focus more on noise reduction (1 < mu). The
            value defaults to 1, which is just the regular MWF. More
            specifically: $W = (Rss + mu Rnn)^{-1} (Rss) E$, but since only Ryy
            and Rnn are known, this can be rewritten as
            $W = (Ryy + (mu - 1)Rnn)^{-1} (Ryy - Rnn) E$

        vad: np.ndarray, 1D, optional
            Numpy array (1D) of the same length as "audio" indicating activity
            of the desired source. If None, it is assumed all frames are speech
            + noise. A remark that should be made here is that, in contrast to
            the vad in the DANSE_network, a vad of all ones also implies using
            groundtruth to compute Rnn (it is not possible to rely on prior
            information in this case which was possible there). THIS IS A SMALL,
            BUT IMPORTANT, INCONSISTENCY!

    Returns
    ----------
        A tuple consisting of the MWF alongside that filter applied to the
        desired part of "audio" and the "noise" alongside the unwanted
        contributions of other audio sources.
    """
    # Bookkeeping
    if audio.ndim != 2 or noise.ndim != 2:
        raise ValueError("audio and noise should be 2-dimensional arrays")
    if audio.shape[0] != noise.shape[0] or audio.shape[1] != noise.shape[1]:
        raise ValueError("audio and noise should have the same dimensions")
    if e1.ndim != 2 or e1.shape[0] != audio.shape[0]:
        raise ValueError(
            f"e1 should be a 2D array of size {audio.shape[0]} x nb desired channels, got {e1.shape} instead"
        )
    if np.any(np.sum(e1, axis=0) != 1):
        raise ValueError(
            "All columns should contain exactly one entry corresponding to that microphone signal being desired"
        )
    if Gamma < 0:
        raise ValueError(f"Gamma should be non-negative! Got {Gamma}.")
    if mu < 0:
        raise ValueError(f"mu for SDW should be >= 0! Got {mu}.")
    if vad is not None and (vad.ndim != 1 or vad.shape[0] != audio.shape[1]):
        raise ValueError("The VAD should be a 1D array of same size as audio")

    # construct vad if not passed in
    if vad is None:
        vad = np.ones((audio.shape[1]), dtype=bool)
    vad = utils.vad.transformVAD(vad, STFTObj.mfft, 1 - STFTObj.hop / STFTObj.mfft)

    # transform to frequency domain
    desFreq = STFTObj.stft(audio, axis=1).transpose(1, 0, 2)
    interferenceFreq = STFTObj.stft(noise, axis=1).transpose(1, 0, 2)
    yFreq = desFreq + interferenceFreq

    # keep the lFFT bins in front to solve jointly
    Ryy = (yFreq[:, :, vad] @ np.conj(yFreq[:, :, vad].transpose(0, 2, 1))) / np.sum(
        vad
    )
    if np.any(~vad):
        Rnn = (
            yFreq[:, :, ~vad] @ np.conj(yFreq[:, :, ~vad].transpose(0, 2, 1))
        ) / np.sum(~vad)
    else:
        Rnn = (
            interferenceFreq @ np.conj(interferenceFreq.transpose(0, 2, 1))
        ) / vad.shape[0]

    if GEVD:
        W_k = GEVD_MWF_computation(Ryy, Rnn, e1, Gamma, mu)
    else:
        W_k = MWF_computation(Ryy, Rnn, e1, Gamma, mu)
    W_k_H = np.conj(W_k.transpose(0, 2, 1))

    desiredFiltered = W_k_H @ desFreq
    interferenceFiltered = W_k_H @ interferenceFreq

    desiredOut = np.real_if_close(STFTObj.istft(desiredFiltered, f_axis=0, t_axis=-1))
    interferenceOut = np.real_if_close(
        STFTObj.istft(interferenceFiltered, f_axis=0, t_axis=-1)
    )

    return (
        W_k,
        desiredOut.T[:, : audio.shape[1]],
        interferenceOut.T[:, : audio.shape[1]],
    )


def MWF_fd_online(
    audio: np.ndarray,
    noise: np.ndarray,
    e1: np.ndarray,
    STFTObj: signal.ShortTimeFFT,
    deltaUpdate: int,
    lmbd: float,
    updateMode: str = "exponential",
    GEVD: bool = False,
    Gamma: float = 0,
    mu: float = 1,
    vad: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Same function as "MWF_fd", but now computes the filter in an online mode
    fashion.

    Parameters
    ----------
        audio: np.ndarray
            refer to `MWF_fd()`

        noise: np.ndarray
            refer to `MWF_fd()`

        e1: np.ndarray
            refer to `MWF_fd()`

        STFTObj: signal.ShortTimeFFT
            refer to `MWF_fd()`

        deltaUpdate: int
            The number of frames before an update of the weights happens

        lmbd: int
            The smoothing parameter for the exponential averaging of the
            autocorrelation matrices.

        updateMode: str, optional
            A string indicating how to update the correlation matrices. In a
            windowed fashion or using exponential averaging, defaults to the
            latter.

        GEVD: bool, optional
            whether or not to use a GEVD-based updating rule, defaults to False

        Gamma: float, optional
            refer to `MWF_fd()`

        mu: float, optional
            refer to `MWF_fd()`

        vad: np.ndarray, optional
            refer to `MWF_fd()`

    Returns
    -------
    The order is the same as `MWF_fd()`, but now returns a set of lists instead
    of all data at once. First a set of filters is returned, then the processed
    outputs.
    """
    # Bookkeeping
    if audio.ndim != 2 or noise.ndim != 2:
        raise ValueError("audio and noise should be 2-dimensional arrays")
    if audio.shape[0] != noise.shape[0] or audio.shape[1] != noise.shape[1]:
        raise ValueError("audio and noise should have the same dimensions")
    if e1.ndim != 2 or e1.shape[0] != audio.shape[0]:
        raise ValueError(
            f"e1 should be a 2D array of size {audio.shape[0]} x nb desired channels, got {e1.shape} instead"
        )
    if np.any(np.sum(e1, axis=0) != 1):
        raise ValueError(
            "All columns should contain exactly one entry corresponding to that microphone signal being desired"
        )
    if Gamma < 0:
        raise ValueError(f"Gamma should be non-negative! Got {Gamma}.")
    if mu < 0:
        raise ValueError(f"mu for SDW should be >= 0! Got {mu}.")
    if deltaUpdate <= 0:
        raise ValueError("deltaUpdate should be strictly positive")
    if lmbd <= 0 or lmbd >= 1:
        raise ValueError("lmbd should be between 0 and 1")
    if updateMode not in ["exponential", "windowed"]:
        raise ValueError(f"updateMode should be either exponential or windowed!")
    if vad is not None and (vad.ndim != 1 or vad.shape[0] != audio.shape[1]):
        raise ValueError("The VAD should be a 1D array of same size as audio")

    if GEVD:  # bind the function for easier handling
        compute_MWF = GEVD_MWF_computation
    else:
        compute_MWF = MWF_computation

    # construct vad if not passed in
    if vad is None:
        vad = np.ones((audio.shape[1]), dtype=bool)
    vad = utils.vad.transformVAD(vad, STFTObj.mfft, 1 - STFTObj.hop / STFTObj.mfft)

    # transform to frequency domain
    desFreq = STFTObj.stft(audio, axis=1).transpose(1, 0, 2)
    interferenceFreq = STFTObj.stft(noise, axis=1).transpose(1, 0, 2)

    # initialize the correlation matrices and weights
    Ryy = np.zeros(
        (int(np.ceil(STFTObj.mfft / 2 + 1)), desFreq.shape[1], desFreq.shape[1]),
        dtype=np.complex128,
    )
    Rnn = np.zeros_like(Ryy)
    idx = np.arange(Ryy.shape[1])
    Ryy[:, idx, idx] = 1e0
    Rnn[:, idx, idx] = 1e-1

    W_mwf = compute_MWF(Ryy, Rnn, e1, Gamma, mu)

    # preallocate memory
    nUpdates = int(np.ceil(desFreq.shape[2] / deltaUpdate))
    audioSlices: list[np.ndarray] = [None for _ in range(nUpdates)]
    noiseSlices: list[np.ndarray] = [None for _ in range(nUpdates)]
    vadSlices: list[np.ndarray] = [None for _ in range(nUpdates)]

    audioOut: list[np.ndarray] = [None for _ in range(nUpdates)]
    noiseOut: list[np.ndarray] = [None for _ in range(nUpdates)]
    Ws: list[np.ndarray] = [None for _ in range(nUpdates + 1)]
    Ws[0] = W_mwf

    # segment data
    for i in range(nUpdates):
        audioSlices[i] = desFreq[:, :, i * deltaUpdate : (i + 1) * deltaUpdate]
        noiseSlices[i] = interferenceFreq[:, :, i * deltaUpdate : (i + 1) * deltaUpdate]
        vadSlices[i] = vad[i * deltaUpdate : (i + 1) * deltaUpdate]

    # run over segments, compute statistics and update
    for i in tqdm(range(nUpdates), desc="Performing iterations", leave=False):
        d = audioSlices[i]
        n = noiseSlices[i]
        vadSlice = vadSlices[i]
        y = d + n

        # filter for output
        W_mwf_H = np.conj(Ws[i].transpose(0, 2, 1))
        audioOut[i] = W_mwf_H @ d
        noiseOut[i] = W_mwf_H @ n

        # compute updates and update weights
        if updateMode == "exponential":
            for j in range(y.shape[2]):
                if vadSlice[j]:
                    Ryy = lmbd * Ryy + (1 - lmbd) * (
                        y[:, :, [j]] @ np.conj(y[:, :, [j]].transpose(0, 2, 1))
                    )
                else:
                    Rnn = lmbd * Rnn + (1 - lmbd) * (
                        y[:, :, [j]] @ np.conj(y[:, :, [j]].transpose(0, 2, 1))
                    )
        else:  # windowed updating
            nActive = np.sum(vadSlice)
            if np.sum(nActive) > y.shape[1]:  # only update if full rank
                Ryy = (
                    y[:, :, vadSlice] @ np.conj(y[:, :, vadSlice].transpose(0, 2, 1))
                ) / nActive
            if nActive < vadSlice.shape[0]:  # some noise samples, update if present
                Rnn = (
                    y[:, :, vadSlice] @ np.conj(y[:, :, vadSlice].transpose(0, 2, 1))
                ) / (vadSlice.shape[0] - nActive)

        Ws[i + 1] = compute_MWF(Ryy, Rnn, e1, Gamma, mu)

    return Ws, audioOut, noiseOut


def MWF_computation(
    Ryy: np.ndarray, Rnn: np.ndarray, e1: np.ndarray, Gamma: float = 0, mu: float = 1
) -> np.ndarray:
    """
    Method for computing an MWF if the autocorrelations are already known.

    It is expected that either 1 bin is used (2D numpy arrays), or that the data
    is already in the freqency domain where the frequency axis is in front.

    The user should indicate which channels are of interest himself (using e1)
    """
    ryd = (Ryy - Rnn) @ e1
    Ryy_prime = Ryy + (mu - 1) * Rnn  # speech distortion weighting

    if Gamma == 0:
        try:
            W_sol = np.linalg.solve(Ryy_prime, ryd)
        except np.linalg.LinAlgError:
            warnings.warn("Solving system failed, reverting to Pinv!")
            W_sol = np.linalg.pinv(Ryy_prime) @ ryd
    else:
        regMtx = np.zeros_like(Ryy_prime)
        diagIdcs = np.arange(0, Ryy_prime.shape[1])
        regMtx[:, diagIdcs, diagIdcs] = Gamma * np.ones((Ryy_prime.shape[1]))
        try:
            W_sol = np.linalg.solve(Ryy_prime + regMtx, ryd)
        except np.linalg.LinAlgError:
            warnings.warn("Solving system failed, reverting to Pinv!")
            W_sol = np.linalg.pinv(Ryy_prime) @ ryd

    return W_sol


def GEVD_MWF_computation(
    Ryy: np.ndarray, Rnn: np.ndarray, e1: np.ndarray, Gamma: float = 0, mu: float = 1
) -> np.ndarray:
    """
    Only works in the frequency domain (3D numpy arrays, frequency axis on the
    first dimension). e1 is expected to be 2D.

    Follows formula (9) - (17) in the GEVD-DANSE paper by A. Hassani, but uses
    a slightly different criterion as it incorporates an SDW-MWF computation for
    which the criterion is $W = (Rss + mu * Rnn)^{-1} Rss e1$. For Rss a 
    low-rank approximation is made with the GEVD, and $(Rss + mu * Rnn)^{-1}$ is 
    replaced by $Ryy + (mu - 1) Rnn$. Concretely, $\Delta$ is computed, but 
    $L$ isn't used. 

    /!\\ `Gamma` is just added as an argument, not used as of yet since GEVD-based
    DANSE appears to be stable enough for now, mostly here for the sake of ease
    of use later on /!\\
    """
    # work per bin: scipy doesn't allow broadcasting of the GEVD
    w_MWF = np.zeros((Ryy.shape[0], Ryy.shape[1], e1.shape[1]), dtype=np.complex128)

    for i in range(w_MWF.shape[0]):
        # compute GEVD and make low-rank approximation of Rss, X = Q⁻H in the paper.
        vals, X = spla.eigh(Ryy[i, :, :], Rnn[i, :, :])
        # sort eigenvalues in descending order
        idcs = np.flip(np.argsort(vals))
        X: np.ndarray = X[:, idcs]

        # compute the diagonal matrix of formula (16)
        DeltaDiag = vals[idcs] - 1
        # low-rank approximation: force the lower-rank eigenvalues to 0
        DeltaDiag[e1.shape[1] :] = 0
        Delta = np.diag(DeltaDiag)
        Q_H = np.linalg.inv(X)

        Rss_lr = np.conj(Q_H.T) @ Delta @ Q_H
        Rnn_w_lr = Ryy[i, :, :] - Rss_lr

        w_MWF[i, :, :] = np.linalg.inv(Ryy[i, :, :] + (mu - 1) * Rnn_w_lr) @ Rss_lr @ e1

    return w_MWF
