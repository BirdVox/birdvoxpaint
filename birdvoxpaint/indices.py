import librosa
import numpy as np
from scipy import stats


def acoustic_complexity_index(S):
    S = np.abs(S) ** 2
    flux = np.abs(np.diff(S, axis=1))
    aci = np.sum(flux, axis=1) / np.sum(S, axis=1)
    return aci[np.newaxis, :]


def average_energy(S):
    S = np.abs(S) ** 2
    return np.sqrt(np.mean(S, axis=1))[np.newaxis, :]


def average_pcen(S, **kwargs):
    S = np.abs(S) ** 2
    pcen = librosa.pcen(S, **kwargs)
    return np.mean(pcen, axis=1)[np.newaxis, :]


def entropy_based_concentration(S):
    S = np.abs(S) ** 2
    entropy = np.array([stats.entropy(row) for row in S])
    concentration = 1 - entropy / np.log(S.shape[1])
    return concentration[np.newaxis, :]


def maximum_energy(S):
    S = np.abs(S) ** 2
    return np.max(S, axis=1)[np.newaxis, :]


def maximum_db(S):
    S = np.abs(S) ** 2
    max_db = 10 * np.log10(np.max(S, axis=1) + 1e-17)
    return max_db[np.newaxis, :]


def maximum_flux(S):
    S = np.abs(S) ** 2
    flux = np.abs(np.diff(S, axis=1))
    return np.max(flux, axis=1)[np.newaxis, :]


def maximum_pcen(S, **kwargs):
    S = np.abs(S) ** 2
    pcen = librosa.pcen(S, **kwargs)
    return np.max(pcen, axis=1)
