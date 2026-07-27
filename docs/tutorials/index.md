# Tutorials

Step-by-step tutorials for common audio and MIDI tasks with coremusic.

## Tutorial Overview

### Getting Started

Start here if you're new to coremusic:

1. [Audio File Basics](audio_file_basics.md) - Read and inspect audio files
2. [Audio Playback](audio_playback.md) - Play audio files
3. [Audio Recording](audio_recording.md) - Record audio from microphones
4. [MIDI Basics](midi_basics.md) - Send and receive MIDI messages

### Audio Processing

- [Audio File Basics](audio_file_basics.md) - Read, write, and analyze audio files
- [Audio Playback](audio_playback.md) - Simple to advanced audio playback
- [Audio Recording](audio_recording.md) - Capture audio from input devices
- [Effects Processing](effects_processing.md) - Apply AudioUnit effects to audio
- [Async Audio](async_audio.md) - Non-blocking audio operations

### MIDI

- [MIDI Basics](midi_basics.md) - MIDI fundamentals: devices, messages, sending/receiving
- [MIDI Transform](midi_transform.md) - Transform MIDI with composable pipelines (transpose, quantize, humanize)

### Music Theory and Generative

- [Music Theory](music_theory.md) - Notes, intervals, scales, chords, progressions
- Generative algorithms: arpeggiators, Euclidean rhythms, Markov chains

## Quick Reference

### Audio Files

```python
--8<-- "examples/index/audio_file.py:example"
```

### Audio Playback

```python
--8<-- "examples/quickstart/play_audio.py:player"
```

### Audio Recording

```python
--8<-- "examples/quickstart/record_audio.py:example"
```

### Effects Processing

```python
--8<-- "examples/quickstart/effects_chain.py:example"
```

### MIDI

```python
--8<-- "examples/quickstart/send_midi_notes.py:example"
```

### Command Line Examples

```bash
# Play audio
coremusic audio play music.wav

# Record audio
coremusic audio record -o recording.wav --duration 10

# Apply effect
coremusic plugin process AUReverb2 input.wav -o output.wav

# Monitor MIDI
coremusic midi monitor

# List devices
coremusic device list
```

## See Also

- [Getting Started](../getting_started.md) - Installation and setup
- [Cookbook](../cookbook/index.md) - Ready-to-use recipes
- [API Reference](../api/index.md) - Complete API reference
- [CLI Guide](../guides/cli.md) - Command-line interface guide
