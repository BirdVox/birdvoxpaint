import datetime
import numpy as np

import librosa
from librosa.util.exceptions import ParameterError

import joblib
import inspect
from functools import wraps



def block_stream(filename=None, y=None, sr=None,
                 frame_length=2048, hop_length=512,
                 segment_duration=1, n_blocks=1, full_frames=True):
    '''Load audio into frames. If given a filename, it will load from file in blocks.
    If y and sr are given, slice into blocks

    Arguments:
        filename (str, optional):
        y, sr (np.ndarray, int, optional):
        frame_length (int):
        hop_length (int):
        block_duration (float)

    '''
    # load audio frame generator

    if filename is not None:
        if y is not None:
            raise ParameterError(
                'Either y or filename must be equal to None')

        # get blocks from file
        duration = librosa.get_duration(filename=filename)
        orig_sr = librosa.get_samplerate(filename)
        sr = sr or orig_sr

        # see: https://librosa.github.io/librosa/_modules/librosa/core/audio.html#stream
        # block_length is in units of `frames` so reverse calculation
        block_length = max(segment_duration * orig_sr, frame_length)
        block_n_frames = librosa.core.samples_to_frames(
            block_length, frame_length, hop_length)

        n_total = duration * orig_sr / librosa.core.frames_to_samples(
            block_n_frames, frame_length, hop_length)
        n_total = int(n_total / n_blocks)

        y_blocks = librosa.stream(filename,
            block_length=block_n_frames * n_blocks,
            frame_length=frame_length,
            hop_length=hop_length)

        # will throw an error if audio is not valid
        y_blocks = (y for y in y_blocks if librosa.util.valid_audio(y, mono=True))

        if sr != orig_sr: # resample if we have a different sr
            y_blocks = (librosa.resample(y, orig_sr, sr) for y in y_blocks)

    else:
        if y is None or sr is None:
            raise ParameterError(
                'At least one of (y, sr) or filename must be provided')

        librosa.util.valid_audio(y, mono=True)

        # get block length, make it evenly divisible into frames (with hop)
        block_length = max(segment_duration * sr, frame_length) * n_blocks # min block size = 1 frame
        block_length = librosa.core.samples_to_frames(
            block_length, frame_length, hop_length) # convert to even frames
        block_length = librosa.core.frames_to_samples(
            block_length, frame_length, hop_length) # convert back

        # get frames from array
        y_blocks = librosa.util.frame(y, block_length, block_length).T

        n_total = len(y_blocks)

    if full_frames: # drop any frames that are incomplete
        y_blocks = (y for y in y_blocks if y.size == block_length)

    return y_blocks, n_total, sr


def multifile_stream(filenames, segment_duration, frame_length, hop_length, sr=None):
    '''Chain multiple file audio block streams into a single stream. It uses samples
    from the next file to complete a previously incomplete block
    '''
    # TODO: need to do this for every file and add up n_total


    all_y_blocks = []
    N_total = 0 # TODO: this doesn't take into account `remainder`
    remainder = None
    for filename in filenames:
        offset = 0

        # get blocks from file
        duration = librosa.get_duration(filename=filename)
        orig_sr = librosa.get_samplerate(filename)
        sr = sr or orig_sr

        block_length = max(segment_duration * orig_sr, frame_length)
        block_n_frames = librosa.core.samples_to_frames(
            block_length, frame_length, hop_length)

        # TODO: this needs to be made a part of the generator expression
        if remainder is not None:
            offset = 1. * (block_length - len(remainder)) / orig_sr
            rem_completed, _ = librosa.load(filename, sr=orig_sr, duration=offset)
            if sr != orig_sr: # resample if we have a different sr
                rem_completed = librosa.resample(rem_completed, orig_sr, sr)
            remainder = np.concatenate([remainder, rem_completed])

            if len(remainder) == block_length: # there were enough samples in the new file
                yield remainder
                remainder = None # clear
            else: # super short file ??? - better to be robust to all types of weather.
                continue

        n_total = duration * orig_sr / librosa.core.frames_to_samples(
            block_n_frames, frame_length, hop_length)
        n_total = int(n_total / n_blocks)
        N_total += n_total

        y_blocks = librosa.stream(filename,
            offset=offset,
            block_length=block_n_frames,
            frame_length=frame_length,
            hop_length=hop_length)

        # will throw an error if audio is not valid
        y_blocks = (y for y in y_blocks if librosa.util.valid_audio(y, mono=True))

        if sr != orig_sr: # resample if we have a different sr
            y_blocks = (librosa.resample(y, orig_sr, sr) for y in y_blocks)

        for y in y_blocks:
            yield y # TODO: return y_blocks, n_total, sr



def spec_frame(S, segment_length):
    '''Break a spectrogram into equal segments.

    Returns:
        Spectrograms with the window applied
        shape: (freq, frame * time) => (frame, freq, time)

    '''
    # NOTE: Depricated: awaiting librosa==0.7.1 .... :)
    # only feed in full length frames into the pipeline
    truncated_length = (S.shape[1] // segment_length) * segment_length
    S = S[:,:truncated_length] # removes incomplete frames
    S = S.T.reshape(-1, segment_length, len(S)) # shape: frame, segment, freq
    S = S.transpose((0, 2, 1)) # shape: frame, freq, segment
    return S




def apply_indices(S, indices, segment_length, n_jobs=1):
    '''Takes an iterable yielding blocks of spectrograms, breaks them into blocks of
    size: segment_length. The number of blocks per yield should equal n_jobs.

    Arguments:
        S (iterable(np.ndarray[freq, time])): the iterable of spectrograms
        indices (callable/list of callables): the indices to calculate. If a
            single callable, it should return all indices as the final axis.
        segment_length (int): the length in time (# frames) to be passed to
            indices at a time.

    '''
    # reshape to blocks of spectrograms (# blocks == n_jobs)
    S_blocks = (spec_frame(spec, segment_length) for spec in S) # yields: block, freq, time

    # allow user to pass a custom indices function - or as a list of functions
    if callable(indices):
        calc_indices = binder(indices, state={})
    else:
        indices = [binder(f, state={}) for f in indices]
        calc_indices = lambda x: np.stack([f(x) for f in indices], axis=-1)

    # build job to calculate indices
    if n_jobs == 1:
        jobs = joblib.Parallel(n_jobs=n_jobs)
        calc_indices = joblib.delayed(calc_indices)
    else:
        jobs = lambda x: x # dummy function so that we can call jobs below like normal.

    S_blocks = (
        np.stack(list(jobs(map(calc_indices, spec))), axis=1) # shape: freq, block
        for spec in S_blocks # yields: block, freq, time
        if spec.size # removes empty blocks
    )

    S = np.concatenate(list(S_blocks), axis=1) # shape: freq, block, indices
    return S

def _shape(X):
    for x in X:
        print(x.shape)
        yield x

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
        raise ParameterError(
            "You must set a sr=({}) and n_fft=({})".format(sr, n_fft))

    if fmin and fmin < 0:
        raise ParameterError("fmin={} must be nonnegative".format(fmin))

    if fmax and fmax > (sr/2):
        raise ParameterError(
            "fmax={} must be smaller than nyquist, f={}".format(fmax, sr))

    fft_frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bin_start = np.where(fft_frequencies >= fmin)[0][0] if fmin else None
    bin_stop = np.where(fft_frequencies < fmax)[0][-1] if fmax else None
    return slice(bin_start, bin_stop)



DAY = 60*60*24
MONTH = DAY * 30
def format_based_on_scale(t, scale, offset=None):
    '''Convert seconds to a formatted time string of some semantic format.

    For example, if the overall scale of the axis is only an hour, we don't
    care about showing the year so we can format the time to only show minute, second, etc.
    '''
    if offset:
        t = datetime.datetime.fromtimestamp(t + offset)

        if scale < DAY:
            return t.strftime('%H:%M:%S')
        if scale < 2*DAY:
            return t.strftime('%d %H:')
        elif scale < MONTH:
            return t.strftime('%m/%d')
        else:
            return t.strftime('%m/%d/%Y')
    else:
        if t < 0: # TODO: use librosa.display.TimeFormatter here ?
            return '-' + str(datetime.timedelta(seconds=-t))
        return str(datetime.timedelta(seconds=t))



def binder(__func__=None, __cfg__=None, **kw):
    '''Provide arguments from config to function only if the function supports them.

    This is essentially `functools.partial` except that it will only fill in
    arguments that are in the function signature, ignoring others.

    NOTE: Does not pass anything extra to **kwargs.

    e.g.

        def my_max_index(S):
            return np.max(S, 1)

        def my_pcen_index(S, state):
            S, state['zi'] = librosa.pcen(S, zi=state.get('zf'), return_zf=True)
            ...
            return S

        def example_transform(funcs=[my_pcen_index, my_max_index], ...):
            ...
            # create function with bound state
            funcs = [binder(f, state={}) for f in funcs]

            for S in S_blocks:
                S = func(S) # it works for both functions !
                ...
                if i_want_to_reset_the_state:
                    for f in funcs:
                        f.cfg_['state'] = {}
    '''

    def outer(func):
        # extract the available arguments
        spec = inspect.getfullargspec(func)
        avail_args = set(spec.args) | set(spec.kwonlyargs)
        # grab the available arguments from config
        cfg = {}
        if __cfg__:
            cfg.update({k: __cfg__[k] for k in avail_args & set(__cfg__)})
        if kw:
            cfg.update({k: kw[k] for k in avail_args & set(kw)})

        func.cfg_ = cfg
        # TODO: allow mutations from the original bound dictionary `config` to
        #       propagate to function. Which would also allow functions
        #       to share a state.

        @wraps(func)
        def inner(*a, **kw):
            return func(*a, **dict(func.cfg_, **kw))
        return inner

    return outer(__func__) if __func__ is not None else outer
