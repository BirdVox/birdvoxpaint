from functools import partial

import joblib
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import tqdm

from . import indices, util
from .indices import *
from .display import *


def transform(
    filename=None,
    frame_length=2048,
    hop_length=512,
    n_mels=None,
    fmin=None,
    fmax=None,
    indices=[average_energy],
    segment_duration=10,
    verbose=True,
    n_jobs=None,
):
    '''Extract spectrotemporal acoustic indices in a monophonic audio file

    Arguments:

    '''

    # if n_jobs is None or -1, parallelize across all available CPUs
    if not n_jobs or n_jobs < 0:
        n_jobs = joblib.cpu_count()

    # measure sample rate and segment length
    sr = librosa.get_samplerate(filename)
    segment_length = segment_duration * sr
    n_frames_per_segment = int(segment_length / hop_length)

    # adjust segment_duration so that it matches the unit roundoff
    # of the Euclidean division above
    segment_duration = n_frames_per_segment * frame_length

    # create a librosa generator object to loop through blocks
    librosa_generator = librosa.core.stream(
        filename,
        block_length=n_frames_per_segment,
        frame_length=frame_length,
        hop_length=hop_length,
        fill_value=0,
    )

    # measure duration of the recording
    total_duration = librosa.get_duration(filename=filename)
    n_segments = int(total_duration / segment_duration)

    # contruct tqdm generator from librosa generator
    # this allows to display a progress bar
    tqdm_generator = tqdm.tqdm(librosa_generator, total=n_segments, disable=not verbose)

    # define frequency slicing function
    # this function reduces the STFT or melspectrogram to a
    # specific subband [fmin, fmax], measured in Hertz.
    slice_fun = util.freq_slice(fmin, fmax, sr, frame_length)

    # define spectrogram function.
    spec_fun = partial(
        spec,
        n_fft=frame_length,
        hop_length=hop_length,
        win_length=frame_length,
        n_mels=n_mels,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        _fft_slice=slice_fun,
    )

    # define a closure for computing acoustic indices of a segment y.
    # note that we use librosa.util.stack instead of np.stack
    # to combine acoustic indices. This is to preserve
    # C contiguity.
    indices_fun = lambda y: [
        librosa.util.stack([acoustic_index(S) for acoustic_index in indices], axis=-1)
        for S in [spec_fun(y)]
    ][0]

    # delay execution of the closure above
    delayed_indices_fun = joblib.delayed(indices_fun)

    # construct joblib generator from delayed joblib object.
    joblib_generator = (delayed_indices_fun(y) for y in tqdm_generator)

    # construct joblib Parallel object.
    parallel_fun = joblib.Parallel(n_jobs=n_jobs)

    # execute
    S = librosa.util.stack(parallel_fun(joblib_generator), axis=1)

    return S


def spec(
    y, sr, n_fft, win_length, hop_length, n_mels=128, fmin=0, fmax=None, _fft_slice=None
):

    # mel spectrogram
    if n_mels:
        S = librosa.feature.melspectrogram(
            y,
            sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=False,
            n_mels=n_mels,
            fmin=fmin or 0,
            fmax=fmax or None,
            power=1,
        )

    else:
        # base spectrogram
        S = librosa.stft(
            y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False
        )

        # NOTE: I'm passing _fft_slice so that we don't need to repetitively calculate it,
        #       but we can also just do: `S[util.freq_slice(fmin, fmax, sr, frame_length)]`
        #       if we prefer that style.
        S = S[_fft_slice or slice(None)]
    return S
