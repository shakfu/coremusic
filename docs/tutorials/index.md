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
import coremusic as cm

# Read audio file
with cm.AudioFile("audio.wav") as audio:
    print(f"Duration: {audio.duration}s")
    print(f"Sample rate: {audio.format.sample_rate}")
    data, count = audio.read_packets(0, 1024)
```

### Audio Playback

```python
import coremusic as cm

player = cm.AudioPlayer()
player.load_file("audio.wav")
player.setup_output()
player.start()

while player.is_playing():
    import time
    time.sleep(0.1)
```

### Audio Recording

```python
import coremusic as cm

recorder = cm.AudioRecorder()
recorder.setup(sample_rate=44100.0, channels=2, output_path="recording.wav")
recorder.start()

import time
time.sleep(10)  # Record for 10 seconds

recorder.stop()
```

### Effects Processing

```python
import coremusic as cm

chain = cm.AudioEffectsChain()
reverb = chain.add_effect_by_name("AUReverb2")
output = chain.add_output()
chain.connect(reverb, output)

chain.open()
chain.initialize()
chain.start()
```

### MIDI

```python
import coremusic as cm

client = cm.MIDIClient("My App")
port = client.create_output_port("Output")

# Send Note On (middle C)
dest = cm.midi_get_destination(0)
port.send(dest, bytes([0x90, 60, 100]))

client.dispose()
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
coremusic midi input monitor

# List devices
coremusic device list
```

## See Also

- [Getting Started](../getting_started.md) - Installation and setup
- [Cookbook](../cookbook/index.md) - Ready-to-use recipes
- [API Reference](../api/index.md) - Complete API reference
- [CLI Guide](../guides/cli.md) - Command-line interface guide
