import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from . import util


def specshow(S, sr, period=60, offset=0,
             segment_duration=10, date_offset=None,
             fmin=0, fmax=None,
             row_height=None, normalize=False,
             x_axis=None, y_axis=None):
    '''Plot a false color spectrogram (or any long spectrogram for that matter).

    Arguments:
        S (np.ndarray):
        sr (int):
    '''
    nfreq, nseg, nidxs = S.shape
    if nidxs == 1: # use cmap
        S = S[:,:,0]

    if normalize: # imshow expects floats to be in [0, 1]
        S = (S - S.min()) / (S.max() - S.min())

    duration = nseg * segment_duration # total duration of spectrogram
    n_blank, offset = divmod(offset, period) # remove blank rows from offset
    smin, smax = S.min(), S.max()

    nrows = int(np.ceil((duration + offset) / period))
    fmax = fmax or sr / 2

    if row_height:
        plt.gcf().set_figheight(row_height * nrows)

    # plt.suptitle('{}')

    gs = gridspec.GridSpec(nrows, 1, wspace=0.1, hspace=0.02)
    for i in range(nrows):
        ax = plt.subplot(gs[i])

        # get the spectrogram row
        j, k = (i * period - offset), ((i + 1) * period - offset)
        S_i = S[:,max(int(j / segment_duration), 0):
                      int(k / segment_duration)]


        plt.imshow(S_i, cmap='magma', aspect='auto', origin='lower', vmin=smin, vmax=smax,
                   extent=[max(0, j), min(duration, k), fmin, fmax])
        plt.xlim([j, k])

        # format the x axis labels
        tj = util.format_based_on_scale(j - n_blank * period, duration, date_offset)
        tk = util.format_based_on_scale(k - n_blank * period, duration, date_offset)

        # move the x bounds to the side (uses xlabel)
        plt.xticks([])
        plt.yticks([])
        if x_axis:
            plt.xlabel('{} to \n{}'.format(tj, tk), ha='left', va='bottom')
            ax.xaxis.set_label_coords(1.01, 0.01)
        if y_axis:
            librosa.display.__scale_axes(ax, y_axis, 'y')
            librosa.display.__decorate_axis(ax.yaxis, y_axis)

        # remove box
        for spine in ax.spines.values():
            spine.set_visible(False)

    # pretty
    plt.tight_layout()
