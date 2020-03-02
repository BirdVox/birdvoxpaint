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
    return np.max(pcen, axis=1)[np.newaxis, :]


def towsey_rgb(S, **kwargs):
    # Compute spectrogram as squared complex modulus.
    S = np.abs(S) ** 2

    # Compute spectral flux.
    flux = np.abs(np.diff(S, axis=1))

    # Note that, contrary to Towsey et al., we estimate background noise by averaging
    # amplitudes over time. This "pink noise model" may not reflect the perceived level of
    # noise in eco-acoustic recordings, but it is conceptually simpler (and faster) than
    # the approach of Towsey, which involves the computation of quantiles.
    background_noise_level = np.sum(S, axis=1)

    # Compute acoustic complexity index, as quotient of averaged flux to averaged energy.
    acoustic_complexity_index = np.sum(flux, axis=1) / background_noise_level

    # Compute temporal entropy index.
    rowwise_entropy = np.array([stats.entropy(row) for row in S])
    entropy_based_concentration = 1 - rowwise_entropy / np.log(S.shape[1])

    # Compute count of acoustic event count as proportion of frames above background noise.
    acoustic_event_threshold = 2.0 * background_noise_level
    acoustic_event_count = np.mean(S > acoustic_event_threshold[:, np.newaxis], axis=1)

    # Stack ACI, ENT, and CVR into a tensor.
    rgb_tensor = np.stack([
        acoustic_complexity_index,
        entropy_based_concentration,
        acoustic_event_count
    ])
    return rgb_tensor
