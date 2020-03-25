import datetime
import numpy as np

import librosa
from librosa.util.exceptions import ParameterError

import joblib
import inspect
from functools import wraps


# TODO: cache this instead of pre-calling it in `transform`
def freq_slice(fmin, fmax, sr, n_fft):
    '''Calculate the slice needed to select a frequency band.

    Arguments:
        fmin, fmax (int): the frequency bounds
        sr (int): the sample rate
        n_fft (int): the fft size

    Returns:
        slice(i[fmin], i[fmax])
    '''
    if not sr or not n_fft:
        raise ParameterError("You must set a sr=({}) and n_fft=({})".format(sr, n_fft))

    if fmin and fmin < 0:
        raise ParameterError("fmin={} must be nonnegative".format(fmin))

    if fmax and fmax > (sr / 2):
        raise ParameterError(
            "fmax={} must be smaller than nyquist, f={}".format(fmax, sr)
        )

    fft_frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bin_start = np.where(fft_frequencies >= fmin)[0][0] if fmin else None
    bin_stop = np.where(fft_frequencies < fmax)[0][-1] if fmax else None
    return slice(bin_start, bin_stop)


DAY = 60 * 60 * 24
MONTH = DAY * 30


def format_based_on_scale(t, scale, offset=None):
    '''Convert seconds to a formatted time string of some semantic format.
    For example, if the overall scale of the axis is only an hour, we don't
    care about showing the year so we can format the time to only show minute,
    second, etc.
    '''
    if offset:
        t = datetime.datetime.fromtimestamp(t + offset)

        if scale < DAY:
            return t.strftime('%H:%M:%S')
        if scale < 2 * DAY:
            return t.strftime('%d %H:')
        elif scale < MONTH:
            return t.strftime('%m/%d')
        else:
            return t.strftime('%m/%d/%Y')
    else:
        if t < 0:  # TODO: use librosa.display.TimeFormatter here ?
            return '-' + str(datetime.timedelta(seconds=-t))
        return str(datetime.timedelta(seconds=t))
