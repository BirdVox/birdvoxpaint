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

To analyze a WAV file at location `wav_path`:

```python
import birdvoxpaint as bvp

X = bvp.transform(
    wav_path,
    indices=[
        bvp.acoustic_complexity_index,
        bvp.entropy_based_concentration,
        bvp.acoustic_event_count,
    ],
)
bvp.rgbshow(X)
```
