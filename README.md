birdvoxpaint
============
False-color spectrograms for long-duration bioacoustic monitoring


## Installation

BirdVoxPaint is hosted on PyPI. To install, run the following command in your Python environment:

```bash
$ pip install birdvoxpaint
```

To install the latest version from source clone the repository and, from the top-level `birdvoxpaint` folder, call:

```bash
$ python setup.py install
```


## Usage

To analyze an audio file:

```python
import librosa
import birdvoxpaint as bvp
import matplotlib.pyplot as plt

# get the path to the audio file you want to use
audio_path = librosa.example('nutcracker')

# calculate acoustic indices
X = bvp.transform(
    audio_path, segment_duration=3, indices=[
        bvp.acoustic_complexity_index,
        bvp.entropy_based_concentration,
        bvp.acoustic_event_count,
    ])

print(X.shape)

# show
bvp.rgbshow(X)
plt.show()
```
