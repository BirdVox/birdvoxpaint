import librosa
import numpy as np


def acoustic_complexity_index(S):
    spectrogram = np.abs(S)**2
    flux = np.abs(np.diff(spectrogram, axis=1))
    return np.sum(flux, axis=1)/np.sum(spectrogram, axis=1)


def average_energy(S):
    spectrogram = np.abs(S)**2
    return np.sqrt(np.mean(spectrogram, axis=1))


def entropy_based_concentration(S):
     spectrogram = np.abs(S)**2
     entropy = np.array([scipy.stats.entropy(row) for row in spectrogram])
     return (1 - entropy/np.log(spectrogram.shape[1]))


def maximum_energy(S):
    spectrogram = np.abs(S)**2
    return np.max(spectrogram, axis=1)


def maximum_flux(S):
    spectrogram = np.abs(S)**2
    flux = np.abs(np.diff(spectrogram, axis=1))
    return np.max(flux, axis=1)


def maximum_pcen(S, **kwargs):
    spectrogram = np.abs(S)**2
    kwargs["return_zf"] = True
    pcen, zf = librosa.pcen(spectrogram, **kwargs)
    return np.max(pcen, axis=1), zf
