import datetime
import numpy as np

import librosa
from librosa.util.exceptions import ParameterError

import joblib


sample2frame = lambda x, window, hop: int((x - window) / hop) + 1
frame2sample = lambda x, window, hop: int((x - 1) * hop) + window

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

        orig_sr = librosa.get_samplerate(filename)
        sr = sr or orig_sr

        # see: https://librosa.github.io/librosa/_modules/librosa/core/audio.html#stream
        # block_length is in units of `frames` so reverse calculation
        block_length = max(segment_duration * orig_sr, frame_length) * n_blocks
        block_n_frames = sample2frame(block_length, frame_length, hop_length)
        y_blocks = librosa.stream(filename,
            block_length=block_n_frames,
            frame_length=frame_length,
            hop_length=hop_length)

        def maybe_resample(y_blocks):
            # will throw an error if audio is not valid
            # resample if we have a different sr
            for y in y_blocks:
                librosa.util.valid_audio(y, mono=True)
                if sr != orig_sr:
                    y = librosa.resample(y, orig_sr, sr)
                yield y

        y_blocks = maybe_resample(y_blocks)

    else:
        if y is None or sr is None:
            raise ParameterError(
                'At least one of (y, sr) or filename must be provided')

        librosa.util.valid_audio(y, mono=True)

        # get block length, remove frame overhang (reuses conversions which does flooring)
        block_length = max(segment_duration * sr, frame_length) * n_blocks # min block size = 1 frame
        block_length = sample2frame(block_length, frame_length, hop_length) # drop overhang
        block_length = frame2sample(block_length, frame_length, hop_length) # convert back

        # get frames from array
        y_blocks = librosa.util.frame(y, block_length, block_length).T

    if full_frames: # drop any frames that are incomplete
        y_blocks = (y for y in y_blocks if y.size == block_length)

    return y_blocks, sr


# def multifile_stream(filenames, block_length):
#     '''Chain multiple file audio block streams into a single stream. It uses samples
#     from the next file to complete a previously incomplete block
#     '''
#     raise NotImplemented("I haven't really played around or tested this. I just drafted it.")
#     remaining = None
#     for filename in filenames:
#         offset = 0 # TODO: uhhh I guess ? - idk
#         # TODO: Load up to block_length -
#         if remaining is not None:
#             offset = block_length - len(remaining)
#             y_complete, _ = librosa.load(filename, duration=offset)
#             remaining = np.concatenate([remaining, y_complete])
#
#             if len(remaining) == block_length: # there were enough samples in the new file
#                 yield remaining
#                 remaining = None # clear
#             else: # super short file ??? - better to be robust to all types of weather.
#                 continue
#
#         y_blocks = librosa.stream(filename,
#             offset=offset,
#             block_length=block_n_frames,
#             frame_length=frame_length,
#             hop_length=hop_length)
#
#         for y in y_blocks:
#             librosa.util.valid_audio(y, mono=True)
#             if sr != orig_sr:
#                 librosa.resample(y, orig_sr, sr)
#             yield y



def spec_frame(S, segment_length):
    '''Break a spectrogram into equal segments.

    Returns:
        Spectrograms with the window applied
        shape: (freq, frame * time) => (frame, freq, time)

    '''

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
        calc_indices = indices
    else:
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
            "fmax={} must be smaller than sample rate sr={}".format(fmax, sr))

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
        if t < 0:
            return '-' + str(datetime.timedelta(seconds=-t))
        return str(datetime.timedelta(seconds=t))
