import librosa
import numpy as np
from scipy import stats


def acoustic_complexity_index(S):
    flux = np.abs(np.diff(S, axis=1))
    return np.sum(flux, axis=1)/np.sum(S, axis=1)


def average_energy(S):
    return np.sqrt(np.mean(S, axis=1))


def entropy_based_concentration(S):
     entropy = np.array([stats.entropy(row) for row in S])
     return (1 - entropy/np.log(S.shape[1]))


def maximum_energy(S):
    return np.max(S, axis=1)


def maximum_flux(S):
    flux = np.abs(np.diff(S, axis=1))
    return np.max(flux, axis=1)


def maximum_pcen(S, **kwargs):
    kwargs["return_zf"] = True
    pcen, zf = librosa.pcen(S, **kwargs)
    return np.max(pcen, axis=1), zf
