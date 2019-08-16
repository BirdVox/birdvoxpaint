import librosa
from librosa.util.exceptions.ParameterError
import numpy as np


def transform(filename=None, y=None, sr=22050,
        n_fft=256, hop_length=32, frame_length=256, fmin=1000, fmax=10000,
        indices=[average_energy], segment_duration=10,
        verbose=False, n_jobs=-1):

    if n_jobs=-1:
        n_jobs = joblib.cpu_count()

    if filename is not None:
        if y is not None:
            raise ParameterError(
                'Either y or filename must be equal to None')
        file_duration = librosa.get_duration(filename=filename)
        orig_sr = librosa.get_samplerate(filename)
        block_length = segment_duration * orig_sr * n_jobs
        y_blocks = librosa.stream(filename, block_length=block_length,
            frame_length=frame_length, hop_length=hop_length)
        if sr is None:
            sr = orig_sr
    else:
        if (y is None) or (sr is None):
            raise ParameterError(
                'At least one of (y, sr) or filename must be provided')
        librosa.util.valid_audio(y, mono=True)
        block_length = segment_duration * sr * n_jobs
        file_duration = librosa.get_duration(y=y, sr=sr)
        y_blocks = librosa.util.frame(y,
            frame_length=block_length, hop_length=block_length)

    if fmin < 0:
        raise ParameterError("fmin={} must be nonnegative".format(fmin))

    if fmax > (sr/2):
        raise ParameterError(
            "fmax={} must be smaller than sample rate sr={}".format(fmax, sr))

    n_indices = len(indices)
    n_blocks = int(np.ceil(file_duration / block_duration))
    fft_frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bin_start = np.where(fft_frequencies>=fmin)[0][0]
    bin_stop = np.where(fft_frequencies<fmax)[0][-1]
    n_freqs = bin_stop - bin_start
    feature_map = joblib.delayed(
        lambda x: np.stack([feature_lambda(x) for feature_lambda in indices]))
    joblib_parallel = joblib.Parallel(n_jobs=n_jobs)
    X_list = []

    for block_id in tqdm.tqdm(range(n_blocks), disable=not verbose):
        if filename is not None:
            y_block = next(y_blocks)
            librosa.util.valid_audio(y_block, mono=True)
            if sr!=orig_sr:
                y_block = librosa.resample(y_block, orig_sr, sr)
        else:
            y_block = y_blocks[:, block_id]
        S = librosa.stft(y_block, n_fft=n_fft,
            hop_length=hop_length, win_length=frame_length, center=False)
        truncated_length = (S_tensor.shape[1]//segment_length) * segment_length
        if truncated_length == 0:
            continue
        else:
            S = S[bin_start:bin_stop, :truncated_length]
        S_tensor = np.reshape(S.T, (-1, segment_length, n_freqs)).T
        n_segments = S_tensor.shape[2]
        job_generator = (feature_map(S_tensor[:, :, segment_id])
            for segment_id in range(n_segments))
        X_list.append(np.stack(joblib_parallel(job_generator), axis=-1))

    X_tensor = np.concatenate(X_list, axis=-1)

    return X_tensor
