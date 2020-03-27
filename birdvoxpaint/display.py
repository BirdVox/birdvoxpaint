import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from . import util


def rgbshow(
    data,
    x_coords=None,
    y_coords=None,
    sr=22050,
    segment_duration=10,
    fmin=None,
    fmax=None,
    ax=None,
    **kwargs
):
    data = np.clip(data, 0, 1)

    kwargs.setdefault('rasterized', True)
    kwargs.setdefault('edgecolors', 'None')
    kwargs.setdefault('shading', 'flat')

    all_params = dict(
        kwargs=kwargs,
        sr=sr / 1000,
        fmin=fmin,
        fmax=fmax,
        hop_length=sr / 1000 * segment_duration,
    )

    # Get the x and y coordinates
    y_axis = 'hz'
    y_coords = librosa.display.__mesh_coords(
        y_axis, y_coords, data.shape[0], **all_params
    )
    x_axis = 'time'
    x_coords = librosa.display.__mesh_coords(
        x_axis, x_coords, data.shape[1], **all_params
    )

    color_tuples = np.array(
        [data[:, :, 0].flatten(), data[:, :, 1].flatten(), data[:, :, 2].flatten()]
    ).transpose()

    axes = librosa.display.__check_axes(ax)
    out = axes.pcolormesh(
        x_coords, y_coords, data[:, :, 0], color=color_tuples, **kwargs
    )
    librosa.display.__set_current_image(ax, out)

    axes.set_xlim(x_coords.min(), x_coords.max())
    axes.set_ylim(y_coords.min(), y_coords.max())

    # Set up axis scaling
    librosa.display.__scale_axes(axes, x_axis, 'x')
    librosa.display.__scale_axes(axes, y_axis, 'y')

    # Construct tickers and locators
    librosa.display.__decorate_axis(axes.xaxis, x_axis)
    last_xtick = plt.xticks()[0][-1]
    if last_xtick < 60:
        axes.xaxis.set_label_text('Time (s)')
    elif last_xtick < 3600:
        axes.xaxis.set_label_text('Time (mm:ss)')
    else:
        axes.xaxis.set_label_text('Time (hh:mm:ss)')
    librosa.display.__decorate_axis(axes.yaxis, y_axis)
    axes.yaxis.set_label_text('Frequency (kHz)')

    return axes
