from functools import partial

import librosa
import librosa.display
import numpy as np

from tqdm.auto import tqdm
import joblib

from . import util
from .indices import *
from .display import *

MAX_JOBS = 40

def transform(filename=None, y=None, sr=None,
              frame_length=2048, hop_length=512,
              n_mels=None, fmin=None, fmax=None,
              indices=[average_energy],
              segment_duration=10,
              verbose=True, n_jobs=None):
    '''Extract spectrotemporal acoustic indices in a monophonic audio file

    Arguments:

    '''

    if not n_jobs or n_jobs < 0: # automatically use available cpus, limited by MAX_JOBS
        n_jobs = min(joblib.cpu_count(), MAX_JOBS)

    # load audio frames - generates one block at a time
    y_blocks, n_blocks, sr = util.block_stream(
        filename, y, sr,
        frame_length=frame_length, hop_length=hop_length,
        segment_duration=segment_duration, n_blocks=n_jobs) # yields: batch*time*n_fft

    y_blocks = tqdm(y_blocks, total=n_blocks, disable=not verbose)

    # prepare spectrogram function
    _fft_slice = util.freq_slice(fmin, fmax, sr, frame_length)
    f_spec = partial(spec,
        n_fft=frame_length, hop_length=hop_length,
        win_length=frame_length, n_mels=n_mels,
        sr=sr, fmin=fmin, fmax=fmax,
         _fft_slice=_fft_slice)

    # convert audio blocks to spectrograms
    S_blocks = (f_spec(y) for y in y_blocks) # yields: freq, time*batch

    # calculate the number of frames in a segment (segment == block length / n_jobs)
    segment_length = librosa.core.samples_to_frames(
        max(segment_duration * sr, frame_length), frame_length, hop_length)

    # break spectrogram into blocks and apply indices
    S = util.apply_indices(S_blocks, indices, segment_length, n_jobs=n_jobs)

    return S


def spec(y, sr, n_fft,
         win_length, hop_length,
         n_mels=128, fmin=0, fmax=None,
         _fft_slice=None):

    # mel spectrogram
    if n_mels:
        S = librosa.feature.melspectrogram(
            y, sr, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, center=False,
            n_mels=n_mels,
            fmin=fmin or 0,
            fmax=fmax or None, power=1)

    else:
        # base spectrogram
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, center=False)

        # NOTE: I'm passing _fft_slice so that we don't need to repetitively calculate it,
        #       but we can also just do: `S[util.freq_slice(fmin, fmax, sr, frame_length)]`
        #       if we prefer that style.
        S = S[_fft_slice or slice(None)]
    return S
