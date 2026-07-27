# CoreMusic Documentation

**CoreMusic** is a comprehensive Cython wrapper for Apple's CoreAudio and CoreMIDI ecosystem, providing both functional and object-oriented Python bindings for professional audio and MIDI development on macOS.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://www.apple.com/macos/)

## Key Features

- **Dual API Design**: Both functional (C-style) and object-oriented (Pythonic) APIs
- **Complete Framework Coverage**: CoreAudio, AudioToolbox, AudioUnit, and CoreMIDI
- **High Performance**: Cython-based with near-native C performance
- **Automatic Resource Management**: Context managers and automatic cleanup
- **Professional Audio Support**: Real-time processing, multi-channel audio, hardware control
- **Comprehensive MIDI**: MIDI 1.0/2.0 support, device management, advanced routing
- **Precise Timing & Sync**: CoreAudioClock for audio/MIDI synchronization and tempo control
- **Music Theory**: Notes, scales, chords, intervals, and harmonic analysis
- **Command Line Interface**: CLI for audio analysis, conversion, and MIDI operations

## Quick Start

### Installation

```bash
pip install coremusic
```

Or build from source:

```bash
git clone https://github.com/shakfu/coremusic.git
cd coremusic
make
```

### Basic Audio File Operations

```python
--8<-- "examples/index/audio_file.py:example"
```

### AudioUnit Processing

```python
--8<-- "examples/index/audiounit.py:example"
```

### MIDI Operations

```python
--8<-- "examples/index/midi.py:example"
```

### Audio/MIDI Synchronization

```python
--8<-- "examples/index/audio_clock.py:example"
```

### Music Theory

```python
--8<-- "examples/index/music_theory.py:example"
```

### Command Line Interface

```bash
# Audio playback and recording
coremusic audio play song.wav --loop
coremusic audio record -o recording.wav -d 10

# Analyze audio
coremusic analyze levels song.wav
coremusic analyze tempo song.wav

# AudioUnit plugins
coremusic plugin list --type effect
coremusic plugin process "AUDelay" input.wav -o output.wav

# MIDI operations
coremusic midi list
coremusic midi monitor
coremusic midi panic

# Render MIDI through an instrument plugin to audio
coremusic plugin render "DLSMusicDevice" song.mid -o rendered.wav

# Diagnose the installation
coremusic doctor
```

See [CLI Guide](guides/cli.md) for complete CLI documentation.

## Documentation Contents
