"""
Pruned version of the code that can be found in the simulator as well for
computing some variants of the MWF
"""

import numpy as np
import scipy.linalg as spla
import warnings


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

        w_MWF[i, :, :] = (
            np.linalg.inv(Ryy[i, :, :] + (mu - 1) * Rnn[i, :, :])
            @ np.conj(Q_H.T)
            @ Delta
            @ Q_H
            @ e1
        )

    return w_MWF
