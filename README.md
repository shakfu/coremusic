# CoreMusic: Python bindings for Apple CoreAudio

[![PyPI version](https://badge.fury.io/py/coremusic.svg)](https://badge.fury.io/py/coremusic)
[![License](https://img.shields.io/github/license/shakfu/coremusic.svg)](https://github.com/shakfu/coremusic/blob/main/LICENSE)

A zero-dependency music development toolkit for macOS providing Python access to Apple's CoreAudio and CoreMIDI frameworks via Cython. Offers both functional (C-style) and object-oriented (Pythonic) APIs with automatic resource management.

## Features

| Framework | Capabilities |
|-----------|-------------|
| **CoreAudio** | Hardware abstraction, device management, format handling |
| **AudioToolbox** | AudioFile I/O, AudioQueue streaming, AudioComponent discovery |
| **AudioUnit** | Plugin hosting, real-time processing, render callbacks, MIDI instrument control |
| **CoreMIDI** | Device/endpoint management, UMP (MIDI 1.0/2.0), thru connections, transforms |
| **Ableton Link** | Network tempo sync, beat-accurate playback/sequencing |

**Audio**: File I/O (WAV, AIFF, MP3), real-time processing, analysis (peak, RMS, tempo, key), buffer pool, memory-mapped I/O

**MIDI**: Device discovery, virtual devices, routing, transformation pipeline (transpose, quantize, humanize, harmonize)

**Music Theory**: 25+ scales, 35+ chords, Note/Interval/Scale/Chord classes

## Installation

```bash
pip install coremusic
```

**Requirements:** macOS, Python 3.11+

### Building from Source

```bash
git clone https://github.com/shakfu/coremusic.git
cd coremusic
make        # Build
make test   # Run tests
```

## Command Line Interface

```bash
coremusic <command> [options]
```

| Command    | Description                                                      |
|----------- |------------------------------------------------------------------|
| `audio`    | Audio file operations (info, play, record, duration, metadata)   |
| `devices`  | Audio device management (list, info, volume, mute, set-default)  |
| `plugin`   | AudioUnit plugins (list, find, info, params, process, render)    |
| `analyze`  | Audio analysis (levels, tempo, key, spectrum, loudness, onsets)  |
| `convert`  | Audio conversion (file, batch, normalize, trim)                  |
| `midi`     | MIDI operations (devices, monitor, record, send, file, output)   |
| `sequence` | MIDI sequence operations (info, play, tracks)                    |

### Examples

```bash
# Audio
coremusic audio play song.wav --loop
coremusic audio record -o recording.wav -d 10
coremusic analyze tempo song.wav
coremusic convert normalize input.wav output.wav --target -1.0

# Devices
coremusic devices list
coremusic devices volume "MacBook Pro Speakers" 0.5

# Plugins
coremusic plugin list --type effect
coremusic plugin list --name-only | grep -i reverb
coremusic plugin process "AUDelay" input.wav -o output.wav
coremusic plugin render "DLSMusicDevice" song.mid -o rendered.wav

# MIDI
coremusic midi devices
coremusic midi input monitor
coremusic midi output panic
coremusic midi file play song.mid
coremusic midi file quantize input.mid -o quantized.mid --grid 1/16

# JSON output for scripting
coremusic --json plugin list --type instrument
```

## Quick Start

### Audio Files

```python
import coremusic as cm

with cm.AudioFile("audio.wav") as f:
    print(f"Duration: {f.duration:.2f}s, Rate: {f.format.sample_rate}Hz")
    data, count = f.read_packets(0, 1000)
```

### Audio Playback

```python
import coremusic as cm

player = cm.AudioPlayer()
player.load_file("audio.wav")
player.setup_output()
player.play()
player.set_looping(True)
# ... later
player.stop()
```

### AudioUnit Plugins

```python
import coremusic as cm

# Discover and use plugins
host = cm.AudioUnitHost()
effects = host.discover_plugins(type='effect')

with host.load_plugin("DLSMusicDevice", type='instrument') as synth:
    synth.note_on(channel=0, note=60, velocity=100)
    time.sleep(1.0)
    synth.note_off(channel=0, note=60)
```

### MIDI

```python
import coremusic as cm

client = cm.MIDIClient("My App")
output_port = client.create_output_port("Output")
output_port.send_data(destination, b'\x90\x3C\x7F')  # Note On
client.dispose()
```

### MIDI Transformation

```python
from coremusic.midi.utilities import MIDISequence
from coremusic.midi.transform import Pipeline, Transpose, Quantize, Humanize

seq = MIDISequence.load("input.mid")
pipeline = Pipeline([
    Transpose(semitones=5),
    Quantize(grid=0.125, strength=0.8),
    Humanize(timing=0.02, velocity=10),
])
pipeline.apply(seq).save("output.mid")
```

### Music Theory

```python
from coremusic.music.theory import Note, Scale, ScaleType, Chord, ChordType

c4 = Note.from_name("C4")
c_major = Scale(c4, ScaleType.MAJOR)
cmaj7 = Chord(c4, ChordType.MAJOR_7)
```

### Ableton Link

```python
import coremusic as cm

with cm.link.LinkSession(bpm=120.0) as session:
    state = session.capture_app_session_state()
    current_time = session.clock.micros()
    beat = state.beat_at_time(current_time, quantum=4.0)
    print(f"Beat: {beat:.2f}, Tempo: {state.tempo:.1f} BPM")
```

## API Overview

### Object-Oriented API (Recommended)

Pythonic wrappers with automatic resource management:

- Context managers (`with` statements) for automatic cleanup
- Type-safe classes instead of integer IDs
- Properties, iteration, operators
- IDE autocompletion and type hints

```python
import coremusic as cm
```

### Functional API (Advanced)

Direct access to CoreAudio C functions for maximum control:

- Direct mapping to CoreAudio C APIs
- Fine-grained resource management
- Familiar for CoreAudio developers

```python
import coremusic.capi as capi

audio_file = capi.audio_file_open_url("audio.wav")
format_data = capi.audio_file_get_property(audio_file, capi.get_audio_file_property_data_format())
capi.audio_file_close(audio_file)
```

Both APIs interoperate - OO objects expose underlying IDs when needed.

## Architecture

```
src/coremusic/
  __init__.py          # Package entry, OO API
  capi.pyx/pxd         # Cython bindings
  objects.py           # OO wrappers
  audio/               # Analysis, buffer pool, streaming
  midi/                # Utilities, transforms, Link integration
  music/               # Theory (scales, chords)
  link.pyx             # Ableton Link bindings
```

Linked frameworks: CoreAudio, AudioToolbox, AudioUnit, CoreMIDI, CoreFoundation

## Testing

```bash
make test           # Fast tests
make test-all       # All tests (1600+)
```

## Documentation

- **[Link Integration Guide](docs/link_integration.md)**: Ableton Link with CoreAudio/CoreMIDI
- **[Error Handling Guide](docs/ERROR_DECORATOR.md)**: OSStatus codes, exceptions

## Resources

- [AudioToolbox Documentation](https://developer.apple.com/documentation/AudioToolbox)
- [AudioUnit Programming Guide](https://developer.apple.com/library/archive/documentation/MusicAudio/Conceptual/AudioUnitProgrammingGuide/Introduction/Introduction.html)
- [Ableton Link](https://github.com/Ableton/link)

## License

MIT License - see LICENSE file.
