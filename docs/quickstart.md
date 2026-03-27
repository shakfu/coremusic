# Quick Start Guide

Get up and running with coremusic in 5 minutes.

## Installation

```bash
# Install from PyPI
pip install coremusic

# Or with uv
uv add coremusic
```

## Verify Installation

```bash
# Check CLI works
coremusic --version

# List audio devices
coremusic device list

# List MIDI devices
coremusic midi list
```

## Your First Script

Create a file called `hello_audio.py`:

```python
import coremusic as cm

# Open and inspect an audio file
with cm.AudioFile("path/to/audio.wav") as audio:
    print(f"Duration: {audio.duration:.2f} seconds")
    print(f"Sample Rate: {audio.format.sample_rate} Hz")
    print(f"Channels: {audio.format.channels_per_frame}")
    print(f"Bit Depth: {audio.format.bits_per_channel}")
```

Run it:

```bash
python hello_audio.py
```

## Common Tasks

### Play Audio

**Command Line:**

```bash
coremusic audio play music.wav
```

**Python:**

```python
import coremusic as cm
import time

player = cm.AudioPlayer()
player.load_file("music.wav")
player.setup_output()
player.start()

while player.is_playing():
    time.sleep(0.1)
```

### Record Audio

**Command Line:**

```bash
coremusic audio record -o recording.wav --duration 10
```

**Python:**

```python
import coremusic as cm
import time

recorder = cm.AudioRecorder()
recorder.setup(sample_rate=44100.0, channels=2, output_path="recording.wav")
recorder.start()
time.sleep(10)
recorder.stop()
```

### Convert Audio Format

**Command Line:**

```bash
# Convert to mono WAV
coremusic convert format input.wav output.wav --channels 1
```

**Python:**

```python
import coremusic as cm

# Convert stereo to mono
output_format = cm.AudioFormatPresets.wav_44100_mono()
cm.convert_audio_file("input.wav", "output.wav", output_format)
```

### Apply Audio Effects

**Command Line:**

```bash
# Add reverb
coremusic plugin process AUReverb2 input.wav -o output.wav
```

**Python:**

```python
import coremusic as cm

chain = cm.AudioEffectsChain()
reverb = chain.add_effect_by_name("AUReverb2")
output = chain.add_output()
chain.connect(reverb, output)
# ... process audio through chain
```

### List Available Plugins

**Command Line:**

```bash
coremusic plugin list
```

**Python:**

```python
import coremusic as cm

# List all AudioUnits
units = cm.list_available_audio_units()
for unit in units:
    print(f"{unit['name']} ({unit['type']})")

# List effect names only
effects = cm.get_audiounit_names(filter_type='aufx')
```

### Monitor MIDI Input

**Command Line:**

```bash
coremusic midi input monitor
```

**Python:**

```python
import coremusic as cm

# List MIDI sources
for i in range(cm.midi_get_number_of_sources()):
    source = cm.midi_get_source(i)
    name = cm.midi_object_get_string_property(source, cm.get_midi_property_name())
    print(f"Source {i}: {name}")
```

### Send MIDI Notes

```python
import coremusic as cm
import time

client = cm.MIDIClient("My App")
port = client.create_output_port("Output")
dest = cm.midi_get_destination(0)

# Send Note On (middle C, velocity 100)
port.send(dest, bytes([0x90, 60, 100]))
time.sleep(0.5)

# Send Note Off
port.send(dest, bytes([0x80, 60, 0]))

client.dispose()
```

## API Patterns

### Context Managers (Recommended)

```python
# Automatic resource cleanup
with cm.AudioFile("audio.wav") as audio:
    data = audio.read_packets(0, 1000)
# File automatically closed
```

### Error Handling

```python
import coremusic as cm

try:
    with cm.AudioFile("audio.wav") as audio:
        data = audio.read_packets(0, 1000)
except cm.AudioFileError as e:
    print(f"Audio error: {e}")
except FileNotFoundError:
    print("File not found")
```

### NumPy Integration

```python
import coremusic as cm

if cm.NUMPY_AVAILABLE:
    import numpy as np

    with cm.AudioFile("audio.wav") as audio:
        # Read as NumPy array
        data = audio.read_as_numpy()
        print(f"Shape: {data.shape}")
        print(f"Peak: {np.max(np.abs(data))}")
```

## CLI Command Reference

```text
Audio Commands:
  coremusic audio play <file>              Play audio file
  coremusic audio record -o <file>         Record audio

Device Commands:
  coremusic device list                    List audio devices
  coremusic device info <name>             Device details

Plugin Commands:
  coremusic plugin list                    List AudioUnits
  coremusic plugin process <name> <file>   Apply effect

MIDI Commands:
  coremusic midi list                      List MIDI devices
  coremusic midi input monitor             Monitor MIDI input

Analysis Commands:
  coremusic analyze info <file>            Audio file info
  coremusic analyze loudness <file>        LUFS measurement
```

## Next Steps

- [Getting Started](getting_started.md) - Detailed installation and setup
- [Tutorials](tutorials/index.md) - Step-by-step tutorials
- [Cookbook](cookbook/index.md) - Ready-to-use recipes
- [API Reference](api/index.md) - Complete API reference

## Getting Help

- Check the [API Reference](api/index.md) for detailed documentation
- See [Tutorials](tutorials/index.md) for worked examples
- Report issues at https://github.com/shakfu/coremusic/issues
