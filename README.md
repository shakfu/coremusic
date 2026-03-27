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

**Music Theory**: 25+ scales, 35+ chords, Note/Interval/Scale/Chord classes, time signatures, rhythmic patterns

## Installation

```bash
pip install coremusic
```

**Requirements:** macOS, Python 3.11+

### Optional Dependencies

CoreMusic has zero runtime dependencies by default. Optional features require additional packages:

```bash
# Audio analysis (beat detection, pitch detection, key detection)
pip install coremusic[analysis]

# Visualization (waveform plots, spectrograms)
pip install coremusic[visualization]

# All optional features
pip install coremusic[all]
```

Check feature availability at runtime:

```python
from coremusic.objects import NUMPY_AVAILABLE
from coremusic.audio.analysis import AudioAnalyzer
import coremusic.utils.scipy as spu

if NUMPY_AVAILABLE:
    # NumPy-based features available
    data = audio.read_as_numpy()

if spu.SCIPY_AVAILABLE:
    # SciPy-based analysis available
    analyzer = AudioAnalyzer("song.wav")
    tempo = analyzer.detect_beats().tempo
```

### Building from Source

```bash
git clone https://github.com/shakfu/coremusic.git
cd coremusic
make        # Build
make test   # Run tests
```

## Command Line Interface

```bash
% coremusic --help
usage: coremusic [-h] [--version] [--json] <command> ...

CoreMusic - Python bindings for Apple CoreAudio.

positional arguments:
  <command>
    audio     Audio file operations
    device    Audio device management
    plugin    AudioUnit plugin discovery
    analyze   Audio analysis and feature extraction
    convert   Convert audio files between formats
    midi      MIDI operations
    sequence  MIDI sequence operations
    completion
              Generate shell completion scripts

options:
  -h, --help  show this help message and exit
  --version   show program's version number and exit
  --json      Output in JSON format
```

| Command    | Description                                                      |
|----------- |------------------------------------------------------------------|
| `audio`    | Audio file operations (info, play, record, duration, metadata)   |
| `device`   | Audio device management (list, info, volume, mute, set-default, monitor) |
| `plugin`   | AudioUnit plugins (list, find, info, params, process, chain, render) |
| `analyze`  | Audio analysis (levels, tempo, key, spectrum, loudness, onsets)  |
| `convert`  | Audio conversion (file, batch, normalize, trim)                  |
| `midi`     | MIDI operations (list, info, play, quantize, receive, monitor, send, panic) |
| `sequence` | MIDI sequence operations (info, play, tracks)                    |
| `completion` | Generate shell completion scripts (bash, zsh, fish)            |

### Shell Completion

Enable tab completion for commands and options:

```bash
# Bash (add to ~/.bashrc)
eval "$(coremusic completion bash)"

# Zsh (add to ~/.zshrc)
eval "$(coremusic completion zsh)"

# Fish (add to ~/.config/fish/config.fish)
coremusic completion fish | source
```

### Examples

```bash
# Audio
coremusic audio play song.wav --loop
coremusic audio record -o recording.wav -d 10
coremusic analyze tempo song.wav
coremusic convert normalize input.wav output.wav --target -1.0

# Devices
coremusic device list
coremusic device volume "MacBook Pro Speakers" 0.5

# Plugins
coremusic plugin list --type effect
coremusic plugin list --name-only | grep -i reverb
coremusic plugin process "AUDelay" input.wav -o output.wav
coremusic plugin chain input.wav -p "AUDelay:Delay Time=0.5" -p "AUReverb2" -o out.wav
coremusic plugin render "DLSMusicDevice" song.mid -o rendered.wav

# MIDI
coremusic midi list
coremusic midi monitor                              # Human-readable MIDI input
coremusic midi receive                              # Display incoming MIDI
coremusic midi receive -o recording.mid             # Save to MIDI file
coremusic midi receive --plugin "DLSMusicDevice"    # Route to synth plugin
coremusic midi play song.mid
coremusic midi quantize input.mid -o quantized.mid --grid 1/16
coremusic midi panic

# Device monitoring
coremusic device monitor                            # Watch for device changes

# JSON output for scripting
coremusic --json plugin list --type instrument
```

## Quick Start

### One-Liner Convenience Functions

```python
from coremusic.shortcuts import play, play_background, convert
from coremusic.shortcuts import analyze_tempo, analyze_key, get_info
from coremusic.shortcuts import list_devices, list_plugins

# Quick playback
play("song.wav")                       # Blocking playback
handle = play_background("song.wav")   # Non-blocking, returns control handle
handle.stop()                          # Stop when done

# Quick analysis
tempo = analyze_tempo("song.wav")      # Get BPM
key, mode = analyze_key("song.wav")    # Get musical key
info = get_info("song.wav")            # Get file metadata

# Quick conversion
convert("input.wav", "output.mp3")

# List resources
devices = list_devices()
plugins = list_plugins(type='effect')
```

### Audio Files

```python
from coremusic.objects import AudioFile

with AudioFile("audio.wav") as f:
    print(f"Duration: {f.duration:.2f}s, Rate: {f.format.sample_rate}Hz")
    data, count = f.read_packets(0, 1000)
```

### Audio Playback

```python
from coremusic.objects import AudioPlayer

player = AudioPlayer()
player.load_file("audio.wav")
player.setup_output()
player.play()
player.set_looping(True)
# ... later
player.stop()
```

### AudioUnit Plugins

```python
from coremusic.audio.audiounit_host import AudioUnitHost

# Discover and use plugins
host = AudioUnitHost()
effects = host.discover_plugins(type='effect')

with host.load_plugin("DLSMusicDevice", type='instrument') as synth:
    synth.note_on(channel=0, note=60, velocity=100)
    time.sleep(1.0)
    synth.note_off(channel=0, note=60)
```

### MIDI

```python
from coremusic.objects import MIDIClient

client = MIDIClient("My App")
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

c4 = Note.from_midi(60)  # Middle C
c_major = Scale(c4, ScaleType.MAJOR)
cmaj7 = Chord(c4, ChordType.MAJOR_7)
```

### Rhythm and Meter

```python
from coremusic.music.theory import TimeSignature, NoteValue, Duration, RhythmPattern

ts = TimeSignature(4, 4)
dotted_quarter = Duration(NoteValue.QUARTER, dots=1)
triplet = Duration.triplet(NoteValue.EIGHTH)

pattern = RhythmPattern.straight_eighths(8)
onset_times = pattern.scale_to_tempo(120)  # At 120 BPM
```

### Ableton Link

```python
from coremusic import link

with link.LinkSession(bpm=120.0) as session:
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
from coremusic.objects import AudioFile, AudioFormat, MIDIClient
from coremusic.constants import AudioFileProperty, AudioFormatID
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
  __init__.py          # Package entry, OO API exports
  capi.pyx/pxd         # Cython bindings to CoreAudio/CoreMIDI
  objects/             # Object-oriented wrappers
    audio.py           # AudioFile, AudioFormat, AudioQueue, etc.
    audiounit.py       # AudioUnit, AudioComponent
    midi.py            # MIDIClient, MIDIPort
    devices.py         # AudioDevice, AudioDeviceManager
    music.py           # MusicPlayer, MusicSequence, MusicTrack
    exceptions.py      # Exception hierarchy
  audio/               # Analysis, buffer pool, streaming
  midi/                # Utilities, transforms, Link integration
  music/               # Theory (scales, chords)
  cli/                 # Command-line interface
  link.pyx             # Ableton Link bindings
```

Linked frameworks: CoreAudio, AudioToolbox, AudioUnit, CoreMIDI, CoreFoundation

## Testing

```bash
make test           # Fast tests
make test-all       # All tests (1600+)
```

## Documentation

### Getting Started

- **[Quick Start Guide](docs/quickstart.rst)**: 5-minute introduction to coremusic

### Tutorials

Step-by-step guides for common tasks:

- **[Audio Playback](docs/tutorials/audio_playback.rst)**: Simple to advanced playback, looping, streaming, effects
- **[Audio Recording](docs/tutorials/audio_recording.rst)**: Recording from input devices, monitoring, formats
- **[MIDI Basics](docs/tutorials/midi_basics.rst)**: Devices, messages, sending/receiving MIDI
- **[Effects Processing](docs/tutorials/effects_processing.rst)**: AudioUnit effects chains, parameters, presets

### Reference

- **[Link Integration Guide](docs/link_integration.md)**: Ableton Link with CoreAudio/CoreMIDI
- **[Error Handling Guide](docs/ERROR_DECORATOR.md)**: OSStatus codes, exceptions

### Executable Examples

The `tests/tutorials/` directory contains doctest-based tutorials that serve as both documentation and tests:

```bash
# Run all tutorial doctests
pytest tests/tutorials/ --doctest-modules -v

# Run specific tutorial
pytest tests/tutorials/test_audio_file_basics.py --doctest-modules -v
```

Available tutorials: `test_quickstart.py`, `test_audio_file_basics.py`, `test_midi_basics.py`, `test_effects_processing.py`, `test_music_theory.py`

## Resources

- [AudioToolbox Documentation](https://developer.apple.com/documentation/AudioToolbox)
- [AudioUnit Programming Guide](https://developer.apple.com/library/archive/documentation/MusicAudio/Conceptual/AudioUnitProgrammingGuide/Introduction/Introduction.html)
- [Ableton Link](https://github.com/Ableton/link)

## License

CoreMusic is licensed under the GNU General Public License v3.0 (GPLv3) - see [LICENSE](LICENSE) file, and includes or links against the following third-party components:

- **Ableton Link** in `coremusic.link` licensed under the GNU General Public License v2.0 (GPLv2), see `thirdparty/link/LICENSE.md` and `thirdparty/link/GNU-GPL-v2.0.md`.
