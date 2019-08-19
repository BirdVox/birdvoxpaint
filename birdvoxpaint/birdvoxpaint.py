from functools import partial

import librosa
import librosa.display
import numpy as np

import tqdm
import joblib

from . import util
from .indices import *
from .display import *

MAX_JOBS = 40

def transform(filename=None, y=None, sr=None,
              frame_length=2048, hop_length=512,

              n_mels=256, fmin=None, fmax=None,
              time_constant=0.4, use_pcen=True, use_mels=True,

              indices=[average_energy],
              segment_duration=10,
              verbose=True, n_jobs=None, return_plot=False):
    '''Convert audio into a false color spectrogram.

    Arguments:
        
    '''

    if not n_jobs or n_jobs < 0: # automatically use available cpus, limited by MAX_JOBS
        n_jobs = min(joblib.cpu_count(), MAX_JOBS)

    # load audio frames - generates one block at a time
    y_blocks, sr = util.block_stream(
        filename, y, sr,
        frame_length=frame_length, hop_length=hop_length,
        segment_duration=segment_duration, n_blocks=n_jobs) # yields: batch*time*n_fft

    y_blocks = tqdm.tqdm(y_blocks, disable=not verbose)

    # prepare spectrogram function
    fft_state = {}
    _fft_slice = util.freq_slice(fmin, fmax, sr, frame_length)
    f_spec = partial(spec,
        n_fft=frame_length, hop_length=hop_length,
        win_length=frame_length, n_mels=n_mels,
        sr=sr, fmin=fmin, fmax=fmax,
        time_constant=time_constant,
        use_mels=use_mels, use_pcen=use_pcen,
         _fft_slice=_fft_slice, state=fft_state)

    # convert audio blocks to spectrograms
    S_blocks = (f_spec(y) for y in y_blocks) # yields: freq, time*batch

    # calculate the number of frames in a segment (segment == block length / n_jobs)
    segment_length = util.sample2frame(max(segment_duration * sr, frame_length), frame_length, hop_length)

    # break spectrogram into blocks and apply indices
    S = util.apply_indices(S_blocks, indices, segment_length, n_jobs=n_jobs)

    if return_plot:
        # for convenience, compose the plotting function with all of the arguments passed. DRY!
        plot_ = partial(specshow,
            sr=sr, segment_duration=segment_duration,
            fmin=fmin, fmax=fmax,
            y_axis='mels' if use_mels else 'frequency')
        return S, plot_
    return S


def spec(y, sr, n_fft,
         win_length, hop_length,
         n_mels=128, time_constant=1,
         fmin=0, fmax=None,
         use_mels=False, use_pcen=False,
         _fft_slice=None, state=None):
    state = state if state is not None else {}

    # mel spectrogram
    if use_mels:
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
        S = np.abs(S)

    if use_pcen:
        # TODO: this isn't working ? I think it has something to do with the windowing not lining up
        S, state['zf'] = librosa.core.pcen(
            S * (2**31), sr=sr, hop_length=hop_length,
            time_constant=time_constant, max_size=1,
            axis=-1, zi=state.get('zf'), return_zf=True)

    # NOTE: I moved the square into here seeing as every index we had expected power not magnitude
    S = S**2
    return S
