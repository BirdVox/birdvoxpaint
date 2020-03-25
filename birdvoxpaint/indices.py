import librosa
import numpy as np
from scipy import stats


def acoustic_complexity_index(S):
    # Note that, contrary to Towsey et al., we estimate background noise by
    # averaging amplitudes over time. This "pink noise model" may not reflect
    # the perceived level of noise in eco-acoustic recordings, but it is
    # conceptually simpler (and faster) than the approach of Towsey,
    # which involves the computation of quantiles.
    S = np.abs(S) ** 2
    flux = np.abs(np.diff(S, axis=1))
    aci = np.sum(flux, axis=1) / np.sum(S, axis=1)
    return aci


def acoustic_event_count(S):
    acoustic_event_threshold = 2.0 * np.mean(S, axis=1)
    return np.mean(
        (S > acoustic_event_threshold[:, np.newaxis]).astype('float'), axis=1
    )


def average_energy(S):
    S = np.abs(S) ** 2
    return np.sqrt(np.mean(S, axis=1))


def average_pcen(S, **kwargs):
    S = np.abs(S) ** 2
    pcen = librosa.pcen(S, **kwargs)
    return np.mean(pcen, axis=1)


def entropy_based_concentration(S):
    S = np.abs(S) ** 2
    entropy = np.array([stats.entropy(row) for row in S])
    concentration = 1 - entropy / np.log(S.shape[1])
    return concentration


def maximum_energy(S):
    S = np.abs(S) ** 2
    return np.max(S, axis=1)


def maximum_db(S):
    S = np.abs(S) ** 2
    max_db = 10 * np.log10(np.max(S, axis=1) + 1e-17)
    return max_db


def maximum_flux(S):
    S = np.abs(S) ** 2
    flux = np.abs(np.diff(S, axis=1))
    return np.max(flux, axis=1)


def maximum_pcen(S, **kwargs):
    S = np.abs(S) ** 2
    pcen = librosa.pcen(S, **kwargs)
    return np.max(pcen, axis=1)
