# CoreMusic: Python bindings for Apple CoreAudio

[![PyPI version](https://badge.fury.io/py/coremusic.svg)](https://badge.fury.io/py/coremusic)
[![License](https://img.shields.io/github/license/shakfu/coremusic.svg)](https://github.com/shakfu/coremusic/blob/main/LICENSE)

## Overview

`coremusic` is a zero-dependency music development toolkit for macOS providing direct Python access via cython to Apple's CoreAudio and CoreMIDI frameworks with near-native performance. It offers both functional (C-style) and object-oriented (Pythonic) APIs with automatic resource management.

Current features include:

- Low-level I/O: CoreAudio, AudioToolbox, CoreMIDI bindings
- Sync: Ableton Link for tempo/beat sync across devices
- MIDI: Sequence manipulation, transforms, file I/O
- Theory: Scales, chords, notes (coremusic.music.theory)


### Frameworks

| Framework | Capabilities |
|-----------|-------------|
| **CoreAudio** | Hardware abstraction, device management, format handling |
| **AudioToolbox** | AudioFile I/O, AudioQueue streaming, AudioComponent discovery |
| **AudioUnit** | Plugin hosting, real-time processing, render callbacks, MIDI instrument control |
| **CoreMIDI** | Device/endpoint management, UMP (MIDI 1.0/2.0), thru connections, message transforms |
| **Ableton Link** | Network tempo sync, beat-accurate playback/sequencing, sub-ms precision |

### Features

**Audio**
- File I/O (WAV, AIFF, MP3, etc.) with format detection/conversion
- Real-time processing with low-latency callbacks
- Analysis: peak, RMS, spectral, tempo, key detection
- Buffer pool, memory-mapped I/O, async operations

**MIDI**
- Device discovery, virtual devices, routing
- Transformation pipeline: transpose, quantize, humanize, filter, harmonize
- AudioUnit instrument control: notes, CC, program change, pitch bend (16 channels)

**Music Theory**
- 25+ scales, 35+ chords, progressions
- Note, Interval, Scale, Chord classes

**Experimental Features (in `examples/`)**

The following features are available as experimental code in the `examples/` directory:
- Generative algorithms: Arpeggiator, Euclidean, Markov, Melody, Polyrhythm
- Markov/Bayesian MIDI analysis with variant generation
- DAW abstractions: Timeline, Track, Clip, Automation

## Installation

### From PyPI (Recommended)

Install the latest version from PyPI:

```bash
pip install coremusic
```

**Supported Python versions:** 3.11, 3.12, 3.13, 3.14

**Platform:** macOS only (CoreAudio frameworks are macOS-specific)

### Prerequisites for Building from Source

- macOS (CoreAudio frameworks are macOS-specific)
- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Xcode command line tools

### Building from Source

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/coremusic.git
    cd coremusic
    ```

2. Install dependencies and build (using uv):

    ```bash
    make
    # or manually
    uv sync --reinstall-package coremusic
    ```

3. Run tests to verify installation:

    ```bash
    make test
    # or manually:
    uv run pytest
    # or run tests individually
    uv run pytest -v tests/test_coremidi.py
    ```

## API Overview

`coremusic` provides two complementary APIs that can be used together or independently:

### Functional API (Advanced)

The functional API provides direct access to CoreAudio C functions with minimal wrapping. This approach offers:

- Direct mapping to CoreAudio C APIs
- Maximum performance and control
- Familiar interface for CoreAudio developers
- Fine-grained resource management
- **Requires explicit import**: `import coremusic.capi as capi`

### Object-Oriented API (Modern)

The object-oriented API provides Pythonic wrappers with automatic resource management:

- **Automatic cleanup** with context managers and destructors
- **Type safety** with proper Python classes instead of integer IDs
- **Pythonic patterns** with properties, iteration, and operators
- **Resource safety** preventing common memory leaks and handle errors
- **Developer experience** with IDE autocompletion and type hints

Both APIs can be used together - the object-oriented layer is built on top of the functional API and maintains full compatibility.

## Command Line Interface

CoreMusic includes a comprehensive CLI for common audio and MIDI operations:

```bash
coremusic <command> [options]
```

### Available Commands

| Command    | Description                                                      |
|----------- |------------------------------------------------------------------|
| `audio`    | Audio file operations (info, duration, metadata)                 |
| `devices`  | Audio device management (list, default, info)                    |
| `plugins`  | AudioUnit plugin discovery (list, find, info, params)            |
| `analyze`  | Audio analysis (peak, rms, silence, tempo, spectrum, key, mfcc)  |
| `convert`  | Convert audio files between formats (file, batch)                |
| `midi`     | MIDI device discovery (devices, inputs, outputs, send, file)     |
| `sequence` | MIDI sequence operations (info, play, tracks)                    |

### CLI Examples

```bash
# Get audio file information
coremusic audio info song.wav

# List audio devices
coremusic devices list

# Find AudioUnit plugins
coremusic plugins find "reverb"

# Analyze audio levels
coremusic analyze levels song.wav

# Detect tempo
coremusic analyze tempo song.wav

# Convert audio format
coremusic convert file input.wav output.mp3 --format mp3

# List MIDI devices
coremusic midi devices

# Show MIDI file info
coremusic sequence info song.mid

# JSON output for scripting
coremusic --json devices list
```

## Quick Start

### Audio File Operations

#### Object-Oriented API (Recommended)

```python
import coremusic as cm

# Simple context manager approach
with cm.AudioFile("path/to/audio.wav") as audio_file:
    print(f"Duration: {audio_file.duration:.2f} seconds")
    print(f"Format: {audio_file.format}")

    # Read audio data
    data, count = audio_file.read_packets(0, 1000)
    print(f"Read {count} packets, {len(data)} bytes")

# Alternative explicit management
audio_file = cm.AudioFile("path/to/audio.wav")
audio_file.open()
try:
    # Work with file
    format_info = audio_file.format
    print(f"Sample rate: {format_info.sample_rate}")
    print(f"Channels: {format_info.channels_per_frame}")
finally:
    audio_file.close()
```

#### Functional API (Advanced)

```python
import coremusic.capi as capi
import coremusic as cm

# Open an audio file
audio_file = capi.audio_file_open_url("path/to/audio.wav")

# Get file format information
format_data = capi.audio_file_get_property(
    audio_file,
    capi.get_audio_file_property_data_format()
)

# Parse format data to a python dict
format_dict = cm.parse_audio_stream_basic_description(format_data)

# Read audio packets
packet_data, packets_read = capi.audio_file_read_packets(audio_file, 0, 1000)

# Close the file (manual cleanup required)
capi.audio_file_close(audio_file)
```

### AudioUnit Operations

#### Object-Oriented API (Recommended)

```python
import coremusic as cm

# Context manager approach with automatic cleanup
with cm.AudioUnit.default_output() as unit:
    # Unit is automatically initialized

    # Configure audio format
    format = cm.AudioFormat(
        sample_rate=44100.0,
        format_id='lpcm',
        channels_per_frame=2,
        bits_per_channel=16
    )
    unit.set_stream_format(format)

    # Start audio processing
    unit.start()
    # ... perform audio operations ...
    unit.stop()

# Unit is automatically cleaned up

# Alternative explicit management
unit = cm.AudioUnit.default_output()
try:
    unit.initialize()
    unit.start()
    # ... audio processing ...
    unit.stop()
    unit.uninitialize()
finally:
    unit.dispose()
```

#### Functional API (Advanced)

```python
import coremusic.capi as capi

# Find default output AudioUnit
description = {
    'type': capi.fourchar_to_int('auou'),
    'subtype': capi.fourchar_to_int('def '),
    'manufacturer': capi.fourchar_to_int('appl'),
    'flags': 0,
    'flags_mask': 0
}

component_id = capi.audio_component_find_next(description)
audio_unit = capi.audio_component_instance_new(component_id)

# Initialize and start
capi.audio_unit_initialize(audio_unit)
capi.audio_output_unit_start(audio_unit)

# ... perform audio operations ...

# Manual cleanup required
capi.audio_output_unit_stop(audio_unit)
capi.audio_unit_uninitialize(audio_unit)
capi.audio_component_instance_dispose(audio_unit)
```

### AudioQueue Operations

#### Object-Oriented API (Recommended)

```python
import coremusic as cm

# Create audio format
format = cm.AudioFormat(
    sample_rate=44100.0,
    format_id='lpcm',
    channels_per_frame=2,
    bits_per_channel=16
)

# Create audio queue for output
queue = cm.AudioQueue.new_output(format)
try:
    # Allocate buffers
    buffers = []
    for i in range(3):
        buffer = queue.allocate_buffer(1024)
        buffers.append(buffer)

        # Fill buffer with audio data and enqueue
        # buffer.data = audio_data  # Fill with actual audio data
        queue.enqueue_buffer(buffer)

    # Start playback
    queue.start()
    # ... playback operations ...
    queue.stop()

finally:
    queue.dispose()
```

#### Functional API (Advanced)

```python
import coremusic.capi as capi

# Define audio format
format_dict = {
    'sample_rate': 44100.0,
    'format_id': 'lpcm',
    'channels_per_frame': 2,
    'bits_per_channel': 16
}

# Create audio queue
queue_id = capi.audio_queue_new_output(format_dict)
try:
    # Allocate and enqueue buffers
    buffer_id = capi.audio_queue_allocate_buffer(queue_id, 1024)
    capi.audio_queue_enqueue_buffer(queue_id, buffer_id)

    # Start and stop playback
    capi.audio_queue_start(queue_id)
    # ... playback operations ...
    capi.audio_queue_stop(queue_id)

finally:
    capi.audio_queue_dispose(queue_id)
```

### MIDI Operations

#### Object-Oriented API (Recommended)

```python
import coremusic as cm

# Create MIDI client
client = cm.MIDIClient("My MIDI App")
try:
    # Create input and output ports
    input_port = client.create_input_port("Input")
    output_port = client.create_output_port("Output")

    # Send MIDI data
    note_on_data = b'\x90\x60\x7F'  # Note On, Middle C, Velocity 127
    output_port.send_data(destination, note_on_data)

finally:
    client.dispose()
```

#### Functional API (Advanced)

```python
import coremusic.capi as capi

# Create MIDI client
client_id = capi.midi_client_create("My MIDI App")
try:
    # Create ports
    input_port_id = capi.midi_input_port_create(client_id, "Input")
    output_port_id = capi.midi_output_port_create(client_id, "Output")

    # Send MIDI data
    note_on_data = b'\x90\x60\x7F'
    capi.midi_send(output_port_id, destination_id, note_on_data, 0)

    # Clean up ports
    capi.midi_port_dispose(input_port_id)
    capi.midi_port_dispose(output_port_id)

finally:
    capi.midi_client_dispose(client_id)
```

### Audio Player Example

```python
import coremusic as cm

# Create an audio player
player = cm.AudioPlayer()

# Load an audio file
player.load_file("path/to/audio.wav")

# Setup audio output
player.setup_output()

# Start playback
player.play()

# Control playback
player.set_looping(True)
print(f"Playing: {player.is_playing()}")
print(f"Progress: {player.get_progress():.2f}")

# Stop playback
player.stop()
```

### AudioUnit Instrument MIDI Control

```python
import coremusic as cm
import time

# Discover available instrument plugins
host = cm.AudioUnitHost()
instruments = host.discover_plugins(type='instrument')
print(f"Found {len(instruments)} instrument plugins")

# Load an instrument (Apple DLSMusicDevice - General MIDI synth)
with host.load_plugin("DLSMusicDevice", type='instrument') as synth:
    # Play a note
    synth.note_on(channel=0, note=60, velocity=100)  # Middle C
    time.sleep(1.0)
    synth.note_off(channel=0, note=60)

    # Play a chord
    notes = [60, 64, 67]  # C major (C, E, G)
    for note in notes:
        synth.note_on(channel=0, note=note, velocity=90)
    time.sleep(1.5)
    synth.all_notes_off(channel=0)

    # Change instrument (General MIDI program change)
    synth.program_change(channel=0, program=0)   # Acoustic Grand Piano
    synth.program_change(channel=0, program=40)  # Violin
    synth.program_change(channel=0, program=56)  # Trumpet

    # Control volume with MIDI CC
    synth.control_change(channel=0, controller=7, value=100)  # Full volume
    synth.control_change(channel=0, controller=7, value=50)   # Half volume

    # Pitch bend
    synth.note_on(channel=0, note=60, velocity=100)
    synth.pitch_bend(channel=0, value=8192)   # Center (no bend)
    synth.pitch_bend(channel=0, value=12288)  # Bend up
    synth.pitch_bend(channel=0, value=8192)   # Back to center
    synth.note_off(channel=0, note=60)

    # Multi-channel orchestration
    synth.program_change(channel=0, program=0)   # Piano
    synth.program_change(channel=1, program=48)  # Strings
    synth.program_change(channel=2, program=56)  # Trumpet

    # Play arrangement on different channels
    synth.note_on(channel=0, note=60, velocity=90)  # Piano: C
    synth.note_on(channel=1, note=64, velocity=70)  # Strings: E
    synth.note_on(channel=2, note=67, velocity=80)  # Trumpet: G
    time.sleep(2.0)

    # Clean stop
    for ch in range(3):
        synth.all_notes_off(channel=ch)
```

### Ableton Link Integration

```python
import coremusic as cm

# Create Link session with context manager
with cm.link.LinkSession(bpm=120.0) as session:
    print(f"Link enabled, {session.num_peers} peers connected")

    # Get synchronized timing information
    state = session.capture_app_session_state()
    current_time = session.clock.micros()
    beat = state.beat_at_time(current_time, quantum=4.0)
    phase = state.phase_at_time(current_time, quantum=4.0)

    print(f"Beat: {beat:.2f}, Phase: {phase:.2f}, Tempo: {state.tempo:.1f} BPM")

# Link + Audio: Beat-accurate playback
with cm.link.LinkSession(bpm=120.0) as session:
    player = cm.AudioPlayer(link_session=session)
    player.load_file("loop.wav")
    player.setup_output()

    # Get Link timing in audio callback
    timing = player.get_link_timing(quantum=4.0)
    print(f"Audio beat: {timing['beat']:.2f}")

    player.play()

# Link + MIDI: Clock synchronization
from coremusic import link_midi

with cm.link.LinkSession(bpm=120.0) as session:
    # Create MIDI client and port
    client = cm.capi.midi_client_create("Link MIDI Demo")
    port = cm.capi.midi_output_port_create(client, "Clock Out")
    dest = cm.capi.midi_get_destination(0)

    # Send MIDI clock synchronized to Link
    clock = link_midi.LinkMIDIClock(session, port, dest)
    clock.start()  # Sends MIDI Start + Clock messages (24 per quarter note)

    # Clock automatically follows Link tempo changes
    time.sleep(10)

    clock.stop()  # Sends MIDI Stop message

# Link + MIDI: Beat-accurate sequencing
with cm.link.LinkSession(bpm=120.0) as session:
    client = cm.capi.midi_client_create("Link Sequencer Demo")
    port = cm.capi.midi_output_port_create(client, "Seq Out")
    dest = cm.capi.midi_get_destination(0)

    # Schedule MIDI events on Link beat grid
    seq = link_midi.LinkMIDISequencer(session, port, dest, quantum=4.0)

    # C major arpeggio (beat-accurate)
    seq.schedule_note(beat=0.0, channel=0, note=60, velocity=100, duration=0.9)  # C4
    seq.schedule_note(beat=1.0, channel=0, note=64, velocity=100, duration=0.9)  # E4
    seq.schedule_note(beat=2.0, channel=0, note=67, velocity=100, duration=0.9)  # G4
    seq.schedule_note(beat=3.0, channel=0, note=72, velocity=100, duration=0.9)  # C5

    seq.start()  # Events play at precise Link beat positions
    time.sleep(5)
    seq.stop()
```

### Performance Features

#### Zero-Copy Buffer Pool

```python
import coremusic as cm
from coremusic.audio.buffer_pool import AudioBufferPool

# Create buffer pool with memory limit (100 MB)
pool = AudioBufferPool(max_memory_mb=100)

# Get buffer from pool (zero-copy reuse)
buffer = pool.get_buffer(size=4096, channels=2)

# Use buffer for audio processing
# ... process audio ...

# Return buffer to pool (automatic cleanup)
pool.return_buffer(buffer)

# Pool automatically manages memory limits
print(f"Pool size: {pool.current_size_mb:.2f} MB")
print(f"Active buffers: {pool.num_active}")
```

#### Memory-Mapped File I/O

```python
from coremusic.audio.mmap_file import MMapAudioFile

# Open large audio file with memory mapping
with MMapAudioFile("large_audio.wav") as mmap_file:
    # Get file info
    print(f"Duration: {mmap_file.duration:.2f}s")
    print(f"Sample rate: {mmap_file.sample_rate}Hz")

    # Efficient random access (no full file load)
    chunk = mmap_file.read_frames(start=100000, count=4096)

    # Process in chunks with zero-copy
    for chunk in mmap_file.iter_chunks(chunk_size=8192):
        # Process audio chunk
        process_audio(chunk)
```

#### Optimized Audio Operations

```python
import coremusic.capi as capi
import numpy as np

# Optimized buffer operations (Cython-accelerated)
left = np.array([0.5, 0.3, 0.8], dtype=np.float32)
right = np.array([0.4, 0.6, 0.7], dtype=np.float32)

# Fast stereo mixing with pan
result = capi.mix_stereo_buffers(left, right, pan=0.5)

# Fast gain application
capi.apply_gain(left, gain=0.8)

# Fast fade operations
capi.apply_fade_in(left, fade_samples=1000)
capi.apply_fade_out(right, fade_samples=1000)
```

### DAW Timeline and Multi-Track Operations (Experimental)

> **Note:** DAW features are experimental and located in `examples/daw/`. They are not part of the core package.

```python
# Add examples/ to your Python path first
import sys
sys.path.insert(0, '/path/to/coremusic/examples')

from daw import Timeline, Track, Clip

# Create DAW timeline
timeline = Timeline(sample_rate=48000, tempo=128.0)

# Add tracks
drums = timeline.add_track("Drums", "audio")
vocals = timeline.add_track("Vocals", "audio")

# Add clips with trimming and fades
drums.add_clip(Clip("drums.wav"), start_time=0.0)
vocals.add_clip(
    Clip("vocals.wav").trim(2.0, 26.0).set_fades(0.5, 1.0),
    start_time=8.0
)

# Transport control
timeline.play()
timeline.stop()
```

### Music Theory

```python
from coremusic.music.theory import Note, Scale, ScaleType, Chord, ChordType

# Music Theory Basics
c4 = Note.from_name("C4")
print(f"MIDI: {c4.midi}, Frequency: {c4.frequency:.2f} Hz")

# Scales (25+ types available)
c_major = Scale(Note.from_name("C4"), ScaleType.MAJOR)
d_dorian = Scale(Note.from_name("D4"), ScaleType.DORIAN)
a_blues = Scale(Note.from_name("A3"), ScaleType.BLUES_MINOR)

# Chords (35+ types available)
cmaj7 = Chord(Note.from_name("C4"), ChordType.MAJOR_7)
dm7 = Chord(Note.from_name("D4"), ChordType.MINOR_7)
g7 = Chord(Note.from_name("G4"), ChordType.DOMINANT_7)

# Export to MIDI file
from coremusic.midi.utilities import MIDISequence
sequence = MIDISequence()
track = sequence.create_track("Melody")
# Add events to track and save
sequence.save("composition.mid")
```

### Generative Algorithms (Experimental)

> **Note:** Generative music features are experimental and located in `examples/generative/`. They are not part of the core package.

See `examples/generative/` for:
- Arpeggiator (10 patterns: UP, DOWN, UP_DOWN, RANDOM, etc.)
- Euclidean rhythm generator (Bjorklund's algorithm)
- Markov chain melody generation
- Bayesian network MIDI analysis
- Polyrhythm generator
- Bit shift register sequencing

### MIDI Transformation Pipeline

```python
from coremusic.midi.utilities import MIDISequence
from coremusic.midi.transform import (
    Pipeline, Transpose, Quantize, Humanize, VelocityScale,
    Harmonize, Reverse, NoteFilter, Arpeggiate
)

# Load existing MIDI file
seq = MIDISequence.load("input.mid")

# Create transformation pipeline
pipeline = Pipeline([
    Transpose(semitones=5),              # Up a perfect fourth
    Quantize(grid=0.125, strength=0.8),  # Quantize to 16th notes (with 80% strength)
    VelocityScale(min_vel=40, max_vel=100),  # Compress velocity range
    Humanize(timing=0.02, velocity=10),  # Add human feel
])

# Apply and save
transformed = pipeline.apply(seq)
transformed.save("output.mid")

# Individual transformers can also be used directly
reversed_seq = Reverse().transform(seq)
harmonized = Harmonize([4, 7]).transform(seq)  # Add thirds and fifths

# Filter notes by range
bass_only = NoteFilter(min_note=24, max_note=48).transform(seq)

# Filter to only notes in a scale (scale mask)
from coremusic.music.theory import Note, Scale, ScaleType
from coremusic.midi.transform import ScaleFilter, filter_to_scale

c_major = Scale(Note('C', 4), ScaleType.MAJOR)
in_scale = ScaleFilter(c_major).transform(seq)  # Only C, D, E, F, G, A, B notes pass through

# Or use the convenience function
a_pent = Scale(Note('A', 3), ScaleType.MINOR_PENTATONIC)
pentatonic_only = filter_to_scale(seq, a_pent)

# Convert chords to arpeggios
arpeggiated = Arpeggiate(pattern='up_down', note_duration=0.1).transform(seq)

# Convenience functions for common operations
from coremusic.midi.transform import transpose, quantize, humanize
result = humanize(quantize(transpose(seq, 12), 0.25), timing=0.01)
```

### CoreMIDI Basic Usage

```python
import coremusic.capi as capi

# Create MIDI 1.0 Universal Packets
ump = capi.midi1_channel_voice_message(
    group=0,
    status=capi.get_midi_status_note_on(),
    channel=0,
    data1=60,  # Middle C
    data2=100  # Velocity
)
print(f"MIDI UMP: {ump:08X}")

# Get MIDI device information
device_count = capi.midi_get_number_of_devices()
print(f"MIDI devices: {device_count}")

source_count = capi.midi_get_number_of_sources()
print(f"MIDI sources: {source_count}")

# Create a virtual MIDI device
try:
    device = capi.midi_device_create("My Virtual Device")
    print(f"Created device: {device}")
except RuntimeError as e:
    print(f"Device creation failed: {e}")
```

### MIDI Thru Connection Example

```python
import coremusic.capi as capi

# Initialize thru connection parameters
params = capi.midi_thru_connection_params_initialize()

# Configure channel mapping (route channel 1 to channel 2)
params['channelMap'][0] = 1  # Channel 1 (0-indexed) -> Channel 2

# Add a note number transform (transpose up one octave)
params['noteNumber'] = {
    'transform': capi.get_midi_transform_add(),
    'value': 12  # Add 12 semitones
}

# Create the thru connection
try:
    connection = capi.midi_thru_connection_create_with_params(params)
    print(f"Created thru connection: {connection}")

    # Clean up
    capi.midi_thru_connection_dispose(connection)
except RuntimeError as e:
    print(f"Thru connection failed: {e}")
```

## Tests, Examples and Demos

The project includes a comprehensive test suite with 1600+ tests covering all major functionality, as well as examples and demos in the `tests/demos/` directory. 

## Architecture

### Core Files

#### Functional API Layer

- **`src/coremusic/capi.pyx`**: Main Cython implementation with Python wrapper functions and audio player with render callbacks

- **`src/coremusic/capi.pxd`**: Main Cython header importing all framework declarations

- **`src/coremusic/coremidi.pxd`**: CoreMIDI framework declarations and structures

- **`src/coremusic/corefoundation.pxd`**: CoreFoundation framework declarations

- **`src/coremusic/audiotoolbox.pxd`**: AudioToolbox framework declarations

- **`src/coremusic/coreaudiotypes.pxd`**: CoreAudio types and structures

#### Object-Oriented API Layer

- **`src/coremusic/objects.py`**: Object-oriented wrappers with automatic cleanup and context managers

- **`src/coremusic/__init__.py`**: Package entry point exposing OO API (functional API via `capi` submodule)

- **`src/coremusic/audio/`**: High-level audio abstractions:
  - `analysis.py`: Audio analysis tools (peak, RMS, spectral)
  - `async_io.py`: Asynchronous audio I/O operations
  - `audiounit_host.py`: AudioUnit plugin hosting infrastructure
  - `buffer_pool.py`: Zero-copy buffer pool with memory limits
  - `mmap_file.py`: Memory-mapped file I/O for large audio files
  - `slicing.py`: Audio slicing and time-stretching utilities
  - `streaming.py`: Streaming audio processing
  - `utilities.py`: Common audio processing utilities
  - `visualization.py`: Audio visualization helpers

- **`src/coremusic/midi/`**: High-level MIDI abstractions:
  - `link.py`: Ableton Link + MIDI integration
  - `utilities.py`: MIDI file I/O, sequencing, and routing
  - `transform.py`: MIDI transformation pipeline (Transpose, Quantize, Humanize, etc.)

- **`src/coremusic/music/`**: Music theory:
  - `theory.py`: Note, Interval, Scale (25+ types), Chord (35+ types), ChordProgression

#### Ableton Link Integration

- **`src/coremusic/link.pyx`**: Cython wrapper for Ableton Link C++ API with context manager support

- **`thirdparty/link/`**: Ableton Link library (C++ headers and implementation)

#### High-Level Features

- **`src/coremusic/buffer_utils.py`**: Audio buffer utilities and conversion helpers

- **`src/coremusic/os_status.py`**: OSStatus error code mappings and handling

- **`src/coremusic/constants.py`**: CoreAudio constant definitions

- **`src/coremusic/log.py`**: Logging utilities for audio operations

- **`src/coremusic/utils/`**: Utility modules (FourCC conversion, type helpers, etc.)

#### Experimental Features (examples/)

The `examples/` directory contains experimental code not part of the core package:

- **`examples/daw/`**: DAW essentials (Timeline, Track, Clip, Automation)
- **`examples/generative/`**: Generative music algorithms (Arpeggiator, Euclidean, Markov, Bayes)

#### Build Configuration

- **`setup.py`**: Build configuration linking CoreAudio and CoreMIDI frameworks

### API Architecture

The project uses a layered architecture:

1. **C Framework Layer**: Direct access to Apple's CoreAudio and CoreMIDI frameworks

2. **Functional API Layer**: Cython wrappers providing direct C function access with Python calling conventions

3. **Object-Oriented API Layer**: Pure Python classes built on the functional layer, providing:
   - Automatic resource management via `__dealloc__` in Cython base class
   - Context manager support (`with` statements)
   - Pythonic interfaces with properties and methods
   - Type safety with proper class hierarchies
   - IDE support with autocompletion and type hints

### Framework Dependencies

The project links against macOS frameworks:

- CoreServices
- CoreFoundation
- AudioUnit
- AudioToolbox
- CoreAudio
- CoreMIDI

Third-party integrations:

- **Ableton Link**: Network tempo synchronization (C++ library, included in `thirdparty/link/`)

Required libraries: m, dl, pthread

### Build Process

To build just:

```sh
make
```

The extension is built using Cython with the following process:

1. Cython compiles `.pyx` files to C (including render callbacks and audio player)

2. C compiler links against CoreAudio frameworks

3. Python extension module is created

All audio playback functionality, including real-time render callbacks, is implemented purely in Cython without requiring separate C source files

To build a wheel:

```sh
make wheel
```

### Testing

Run the complete test suite:

```sh
make test           # Run fast tests (skip slow ones)
make test-all       # Run all tests including slow ones
```

Test clean installation without optional dependencies:

```sh
make test-clean-install
```

This creates a fresh virtual environment in `/tmp`, installs coremusic without numpy/scipy/matplotlib, and verifies all core functionality imports correctly. Useful for CI/CD and verifying optional dependency handling.

## API Migration and Best Practices

### Choosing Between APIs

**Use Object-Oriented API when:**

- Building new applications
- Rapid prototyping and development
- You want automatic resource management
- Working with complex audio workflows
- Team development where code safety is important

**Use Functional API when:**

- Maximum performance is critical
- Porting existing CoreAudio C code
- Need fine-grained control over resource lifetimes
- Working within existing functional codebases
- Building low-level audio processing components

### Best Practices

- **Resource Management**: Always use context managers (`with` statements) when possible

- **Error Handling**: Both APIs provide consistent exception types (`AudioFileError`, `AudioUnitError`, etc.)

- **Performance**: Object-oriented layer adds minimal overhead - choose based on development needs

- **Mixing APIs**: Both APIs can be used together - OO objects expose their underlying IDs when needed

## Performance

coremusic provides near-native performance through both APIs with recent optimizations:

**Functional API:**

- Direct C API access with zero Python overhead
- Explicit memory management for optimal control
- Maximum performance for real-time audio processing

**Object-Oriented API:**

- Minimal overhead layer built on functional API
- Automatic resource management without performance penalty
- Efficient Cython-based cleanup via `__dealloc__`

**Recent Performance Enhancements (October 2025):**

- **Cython-Optimized Operations**: Core audio buffer operations (mixing, gain, pan, fade) now implemented in optimized Cython integrated into `capi.pyx`

- **Zero-Copy Buffer Pool**: Efficient buffer reuse with configurable memory limits, reducing allocation overhead

- **Memory-Mapped I/O**: Large file handling without loading entire files into memory

- **Reduced Allocations**: Buffer pool prevents repeated allocation/deallocation cycles

- **Benchmarking Suite**: Comprehensive performance benchmarks in `benchmarks/` for measuring optimization impact

**Common Performance Features:**

- Optimized audio processing pipelines
- Real-time audio callback support
- Efficient memory management
- Direct framework integration
- Low-latency audio processing (< 10ms typical)

## Contributing

Contributions are welcome! Please see the project structure and existing code patterns when adding new features.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `make test`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on Apple's CoreAudio framework
- Inspired by various CoreAudio examples and tutorials
- Built with Cython for high performance
- Test audio file (amen.wav) from public domain sources

## Documentation

### Complete Guides

- **[Link Integration Guide](docs/link_integration.md)**: Comprehensive documentation for Ableton Link integration with CoreAudio and CoreMIDI including examples for:
  - Network tempo synchronization
  - Beat-accurate audio playback
  - MIDI clock synchronization
  - Beat-accurate MIDI sequencing
  - Combined audio + MIDI synchronized workflows

- **[Error Handling Guide](docs/ERROR_DECORATOR.md)**: Comprehensive error handling system documentation covering:
  - OSStatus error code mappings
  - Error decorator patterns
  - Exception hierarchy
  - Best practices for error handling

### API References

- **Functional API**: See `src/coremusic/capi.pyx` for complete C API bindings

- **Object-Oriented API**: See `src/coremusic/objects.py` for Pythonic wrappers

- **Link API**: See `src/coremusic/link.pyx` for Link integration

- **Audio Utilities**: See `src/coremusic/audio/` for high-level audio processing modules

- **MIDI Utilities**: See `src/coremusic/midi/` for MIDI processing utilities

## Resources

For more information about CoreAudio and Link development:

- The [AudioToolbox framework](https://developer.apple.com/documentation/AudioToolbox) provides interfaces for recording, playback, and stream parsing

- The [AudioUnit framework](https://developer.apple.com/documentation/audiounit) -- use the refernce above.

- [AudioUnit App Extension](https://developer.apple.com/library/archive/documentation/General/Conceptual/ExtensibilityPG/AudioUnit.html#//apple_ref/doc/uid/TP40014214-CH22-SW1)

- [AudioUnit Programming Guide](https://developer.apple.com/library/archive/documentation/MusicAudio/Conceptual/AudioUnitProgrammingGuide/Introduction/Introduction.html)

- [Ableton Link](https://www.ableton.com/en/link/) - Official Link website and documentation

- [Link GitHub Repository](https://github.com/Ableton/link) - Link source code and examples
