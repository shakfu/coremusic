# Import Guide

Where everything lives, and how to import it.

## The Rule

Import from the subpackage that owns the domain:

```python
--8<-- "examples/guides/imports/common.py:example"
```

The top-level `coremusic` package deliberately exports nothing but
`__version__`. There is no flat namespace: `coremusic.AudioFile` does not
exist, and neither does the `coremusic.objects` package that earlier versions
had. See the [Migration Guide](migration.md) if you are updating code written
against either.

## Package Map

```
coremusic/
├── __init__.py         # version only
├── capi.pyx            # functional C API - every CoreAudio/CoreMIDI call
├── base.py             # CoreAudioObject, AudioPlayer, NUMPY_AVAILABLE
├── exceptions.py       # exception hierarchy
├── shortcuts.py        # one-call helpers (play, convert, get_info, ...)
├── constants/          # enumerated CoreAudio/CoreMIDI constants
│
├── audio/              # audio domain
│   ├── core.py         # AudioFile, AudioFormat, AudioQueue, AudioConverter
│   ├── units.py        # AudioUnit, AudioComponent
│   ├── graph.py        # AUGraph
│   ├── devices.py      # AudioDevice, AudioDeviceManager
│   ├── clock.py        # AudioClock, ClockTimeFormat
│   ├── utilities.py    # conversion helpers, AudioEffectsChain
│   ├── async_io.py     # AsyncAudioFile, AsyncAudioQueue
│   ├── analysis.py     # AudioAnalyzer, LivePitchDetector
│   ├── slicing.py      # AudioSlicer, SliceCollection
│   ├── streaming.py    # AudioInputStream, AudioOutputStream, StreamGraph
│   ├── visualization.py# WaveformPlotter, SpectrogramPlotter
│   ├── buffer_pool.py  # BufferPool
│   ├── mmap_file.py    # MMapAudioFile
│   └── audiounit_host.py # AudioUnitHost, AudioUnitPlugin, AudioUnitChain
│
├── midi/               # MIDI domain
│   ├── core.py         # MIDIClient, MIDIPort, MIDIEndpoint
│   ├── player.py       # MusicPlayer, MusicSequence, MusicTrack
│   ├── utilities.py    # MIDISequence, MIDITrack, MIDIEvent, MIDIRouter
│   ├── transform.py    # Pipeline, Transpose, Quantize, Humanize, ...
│   └── link.py         # LinkMIDIClock, LinkMIDISequencer
│
├── music/              # music theory
│   └── theory.py       # Note, Scale, Chord, Interval, TimeSignature
│
├── utils/              # helpers
│   ├── scipy.py        # SciPy-backed DSP
│   ├── fourcc.py       # FourCC conversion
│   └── batch.py        # parallel batch processing
│
├── cli/                # command-line interface
└── link.pyx            # Ableton Link: LinkSession, SessionState, Clock
```

Classes are re-exported from each subpackage's `__init__`, so
`from coremusic.audio import AudioAnalyzer` and
`from coremusic.audio.analysis import AudioAnalyzer` both work. Prefer the
short form; reach for the module path when you want to be explicit about where
something comes from.

## Audio

Core objects:

```python
--8<-- "examples/guides/imports/audio.py:objects"
```

Async I/O:

```python
--8<-- "examples/guides/imports/audio.py:async"
```

Analysis:

```python
--8<-- "examples/guides/imports/audio.py:analysis"
```

Slicing:

```python
--8<-- "examples/guides/imports/audio.py:slicing"
```

Visualization (requires `coremusic[visualization]`):

```python
--8<-- "examples/guides/imports/audio.py:visualization"
```

AudioUnit hosting:

```python
--8<-- "examples/guides/imports/audio.py:audiounit-host"
```

## MIDI

Clients, ports, and endpoints:

```python
--8<-- "examples/guides/imports/midi.py:objects"
```

MIDI files and events:

```python
--8<-- "examples/guides/imports/midi.py:files"
```

Transformation pipeline:

```python
--8<-- "examples/guides/imports/midi.py:transform"
```

Ableton Link integration:

```python
--8<-- "examples/guides/imports/midi.py:link"
```

## Music Theory

```python
--8<-- "examples/guides/imports/support.py:music"
```

## Constants and Exceptions

```python
--8<-- "examples/guides/imports/support.py:constants"
```

```python
--8<-- "examples/guides/imports/support.py:exceptions"
```

## Utilities

SciPy-backed DSP (requires `coremusic[analysis]`):

```python
--8<-- "examples/guides/imports/support.py:scipy"
```

FourCC conversion:

```python
--8<-- "examples/guides/imports/support.py:fourcc"
```

## Ableton Link

```python
--8<-- "examples/guides/imports/support.py:link"
```

## Shortcuts

For the common one-liners, `coremusic.shortcuts` wraps the objects above:

```python
--8<-- "examples/guides/imports/support.py:shortcuts"
```

## Functional API

Every CoreAudio and CoreMIDI call is available from `coremusic.capi`. The
object layer is built on it, and the two interoperate: objects expose their
underlying id through `object_id`, and functions that take an id accept it.

```python
import coremusic.capi as capi

# Direct C function calls
file_id = capi.audio_file_open_url("audio.wav")
data, count = capi.audio_file_read_packets(file_id, 0, 1024)
capi.audio_file_close(file_id)

# Constants are exposed as get_* functions
property_id = capi.get_audio_file_property_data_format()
format_id = capi.fourchar_to_int('lpcm')
```

## Best Practices

**Import what you use, from where it lives.**

```python
from coremusic.audio import AudioFile
from coremusic.audio.analysis import AudioAnalyzer
from coremusic.midi import MIDIClient
```

**Do not use wildcard imports.** They pull in a large namespace and hide where
a name came from:

```python
from coremusic.audio import *   # don't
```

**Group imports** in the usual order - standard library, third party, then
coremusic:

```python
import time
from pathlib import Path

import numpy as np

from coremusic.audio import AudioFile
from coremusic.midi import MIDIClient
from coremusic import link
```

## Optional Dependencies

Some modules need extras. Import them anyway - each exposes a flag rather than
failing at import time:

| Module | Extra | Flag |
| --- | --- | --- |
| `coremusic.audio.analysis` | `coremusic[analysis]` | `NUMPY_AVAILABLE`, `SCIPY_AVAILABLE` |
| `coremusic.utils.scipy` | `coremusic[analysis]` | `SCIPY_AVAILABLE` |
| `coremusic.audio.visualization` | `coremusic[visualization]` | `MATPLOTLIB_AVAILABLE` |

```python
from coremusic.audio import NUMPY_AVAILABLE

if NUMPY_AVAILABLE:
    ...
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'coremusic'`** - the package is not
installed in the interpreter you are running. `pip install coremusic`, or
`uv sync` in a checkout.

**`ModuleNotFoundError: No module named 'coremusic.objects'`** - that package
was removed in 0.2.3. Its contents moved to `coremusic.audio`,
`coremusic.midi`, `coremusic.exceptions`, and `coremusic.base`. See the
[Migration Guide](migration.md).

**`ImportError: cannot import name 'AudioFile' from 'coremusic'`** - there is
no flat namespace. Import from the subpackage: `from coremusic.audio import
AudioFile`.

**`ImportError` from a `.so` file** - the Cython extension is missing or built
for another Python version. In a checkout, rebuild with `make build`.

## See Also

- [Migration Guide](migration.md) - updating code written for older versions
- [API Reference](../api/index.md) - the classes themselves
- [Quick Start](../quickstart.md) - a five-minute tour
