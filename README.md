# coremusic

A Cython wrapper for Apple's CoreAudio and CoreMIDI ecosystem, providing Python bindings for audio development on macOS. This project exposes a subset of CoreAudio and CoreMIDI C APIs through Python, enabling audio applications, real-time processing, and audio development on the Apple platform.

## Overview

`coremusic` is a c-based Python extension that provides direct access to Apple's CoreAudio frameworks. Built with Cython, it offers near-native performance while maintaining the ease of use of Python. The wrapper covers the complete CoreAudio ecosystem, from low-level hardware control to high-level audio processing units.

### Key Features

- **Complete CoreAudio Framework Coverage**: Full access to CoreAudio, AudioToolbox, and AudioUnit APIs

- **High Performance**: Cython-based implementation with near-native C performance

- **Professional Audio Support**: Real-time audio processing, multi-channel audio, and hardware control

- **Audio File I/O**: Support for WAV, AIFF, MP3, and other audio formats through CoreAudio

- **AudioUnit Integration**: Complete AudioUnit discovery, instantiation, and lifecycle management

- **AudioQueue Support**: High-level audio queue management for streaming and playback

- **Hardware Abstraction**: Direct access to audio hardware and device management

- **Format Detection**: Automatic audio format detection and conversion

- **Real-time Processing**: Low-latency audio processing capabilities

## Supported Frameworks

### CoreAudio

- Core audio types and hardware abstraction
- Audio device management and control
- Hardware object manipulation
- Audio format handling and conversion

### AudioToolbox

- AudioFile operations (open, read, write, close)
- AudioQueue creation and management
- AudioComponent discovery and management
- High-level audio services

### AudioUnit

- AudioUnit discovery and instantiation
- Real-time audio processing units
- Audio effects and generators
- Hardware audio output control
- Render callback infrastructure

## Installation

### Prerequisites

- macOS (CoreAudio frameworks are macOS-specific)
- Python 3.6 or higher
- Cython
- Xcode command line tools

### Building from Source

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/coremusic.git
    cd coremusic
    ```

2. Install dependencies:

    ```bash
    pip install cython
    ```

3. Build the extension:

    ```bash
    make
    # or manually:
    python3 setup.py build_ext --inplace
    ```

4. Run tests to verify installation:

    ```bash
    make test
    # or manually:
    pytest
    ```

## Quick Start

### Basic Audio File Operations

```python
import coreaudio as ca

# Open an audio file
audio_file = ca.audio_file_open_url(
    "path/to/audio.wav",
    ca.get_audio_file_read_permission(),
    ca.get_audio_file_wave_type()
)

# Get file format information
format_data = ca.audio_file_get_property(
    audio_file,
    ca.get_audio_file_property_data_format()
)

# Read audio packets
packet_data, packets_read = ca.audio_file_read_packets(audio_file, 0, 1000)

# Close the file
ca.audio_file_close(audio_file)
```

### AudioUnit Setup

```python
import coreaudio as ca

# Find default output AudioUnit
description = {
    'type': ca.get_audio_unit_type_output(),
    'subtype': ca.get_audio_unit_subtype_default_output(),
    'manufacturer': ca.get_audio_unit_manufacturer_apple(),
    'flags': 0,
    'flags_mask': 0
}

component_id = ca.audio_component_find_next(description)
audio_unit = ca.audio_component_instance_new(component_id)

# Initialize and start
ca.audio_unit_initialize(audio_unit)
ca.audio_output_unit_start(audio_unit)

# ... perform audio operations ...

# Cleanup
ca.audio_output_unit_stop(audio_unit)
ca.audio_unit_uninitialize(audio_unit)
ca.audio_component_instance_dispose(audio_unit)
```

### Audio Player Example

```python
import coreaudio as ca

# Create an audio player
player = ca.AudioPlayer()

# Load an audio file
player.load_file("path/to/audio.wav")

# Setup audio output
player.setup_output()

# Start playback
player.start()

# Control playback
player.set_looping(True)
print(f"Playing: {player.is_playing()}")
print(f"Progress: {player.get_progress():.2f}")

# Stop playback
player.stop()
```

## API Reference

### Audio File Operations

#### `audio_file_open_url(file_path, permissions, file_type_hint)`

Open an audio file for reading or writing.

**Parameters:**

- `file_path` (str): Path to the audio file
- `permissions` (int): File access permissions (use `get_audio_file_read_permission()`)
- `file_type_hint` (int): File type hint (use `get_audio_file_wave_type()` for WAV files)

**Returns:** Audio file ID (int)

#### `audio_file_get_property(audio_file_id, property_id)`

Get a property from an audio file.

**Parameters:**

- `audio_file_id` (int): Audio file ID from `audio_file_open_url()`
- `property_id` (int): Property ID (use `get_audio_file_property_data_format()` for format info)

**Returns:** Property data as bytes

#### `audio_file_read_packets(audio_file_id, start_packet, num_packets)`

Read audio packets from a file.

**Parameters:**

- `audio_file_id` (int): Audio file ID
- `start_packet` (int): Starting packet number
- `num_packets` (int): Number of packets to read

**Returns:** Tuple of (packet_data, packets_read)

#### `audio_file_close(audio_file_id)`

Close an audio file.

### AudioUnit Operations

#### `audio_component_find_next(description)`

Find an audio component matching the description.

**Parameters:**

- `description` (dict): Component description with keys: type, subtype, manufacturer, flags, flags_mask

**Returns:** Component ID (int) or None

#### `audio_component_instance_new(component_id)`

Create a new instance of an audio component.

**Parameters:**

- `component_id` (int): Component ID from `audio_component_find_next()`

**Returns:** AudioUnit instance ID (int)

#### `audio_unit_initialize(audio_unit_id)`

Initialize an AudioUnit.

#### `audio_output_unit_start(audio_unit_id)`

Start audio output.

#### `audio_output_unit_stop(audio_unit_id)`

Stop audio output.

### AudioQueue Operations

#### `audio_queue_new_output(audio_format)`

Create a new output audio queue.

**Parameters:**

- `audio_format` (dict): Audio format specification

**Returns:** AudioQueue ID (int)

#### `audio_queue_allocate_buffer(queue_id, buffer_size)`

Allocate a buffer for an audio queue.

#### `audio_queue_start(queue_id)`

Start an audio queue.

#### `audio_queue_stop(queue_id, immediate)`

Stop an audio queue.

### Utility Functions

#### `fourchar_to_int(code)`

Convert a four-character code string to integer.

**Parameters:**

- `code` (str): Four-character code (e.g., 'WAVE', 'TEXT')

**Returns:** Integer representation

#### `int_to_fourchar(n)`

Convert an integer to a four-character code string.

**Parameters:**

- `n` (int): Integer value

**Returns:** Four-character code string

### AudioPlayer Class

The `AudioPlayer` class provides a high-level interface for audio playback:

#### `AudioPlayer()`

Create a new AudioPlayer instance.

#### `load_file(file_path)`

Load an audio file for playback.

#### `setup_output()`

Setup the audio output unit.

#### `start()`

Start audio playback.

#### `stop()`

Stop audio playback.

#### `set_looping(loop)`

Enable or disable looping playback.

#### `is_playing()`

Check if audio is currently playing.

#### `get_progress()`

Get current playback progress (0.0 to 1.0).

#### `reset_playback()`

Reset playback to the beginning.

## Examples and Demos

The project includes comprehensive demonstration scripts in the `tests/demos/` directory:

### Unified Audio Demo

```bash
python3 tests/demos/unified_audio_demo.py
```

This comprehensive demo showcases:

- CoreAudio constants and utilities
- Audio file operations and format detection
- AudioUnit infrastructure testing
- AudioQueue operations
- Real audio playback using AudioPlayer
- Advanced CoreAudio features

### Test Suite

```bash
pytest tests/test_coremusic.py -v
```

The test suite covers:

- Audio file I/O operations
- AudioUnit lifecycle management
- AudioQueue functionality
- Error handling and edge cases
- Performance characteristics

## Architecture

### Core Files

- **`coreaudio.pyx`**: Main Cython implementation with Python wrapper functions
- **`coreaudio.pxd`**: Cython header file with C declarations for CoreAudio APIs
- **`audio_player.c/h`**: C implementation of audio player with render callbacks
- **`setup.py`**: Build configuration linking CoreAudio frameworks

### Framework Dependencies

The project links against macOS frameworks:

- CoreServices
- CoreFoundation  
- AudioUnit
- AudioToolbox
- CoreAudio

Required libraries: m, dl, pthread

### Build Process

The extension is built using Cython with the following process:

1. Cython compiles `.pyx` files to C
2. C compiler links against CoreAudio frameworks
3. Python extension module is created

## Performance

coremusic provides near-native performance through:

- Direct C API access without Python overhead
- Efficient memory management
- Optimized audio processing pipelines
- Real-time audio callback support

## Use Cases

### Professional Audio Applications

- Digital Audio Workstations (DAWs)
- Audio effects and processors
- Real-time audio synthesis
- Multi-channel audio handling

### Audio Analysis and Processing

- Audio format conversion
- Spectral analysis
- Audio file manipulation
- Batch audio processing

### Educational and Research

- Audio algorithm development
- CoreAudio API exploration
- Audio programming education
- Prototype audio applications

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

## Resources

For more information about CoreAudio development:

- The [AudioToolbox framework](https://developer.apple.com/documentation/AudioToolbox) provides interfaces for recording, playback, and stream parsing

- The [AudioUnit framework](https://developer.apple.com/documentation/audiounit) -- use the refernce above.

- [AudioUnit App Extension](https://developer.apple.com/library/archive/documentation/General/Conceptual/ExtensibilityPG/AudioUnit.html#//apple_ref/doc/uid/TP40014214-CH22-SW1)

- [AudioUnit Programming Guide](https://developer.apple.com/library/archive/documentation/MusicAudio/Conceptual/AudioUnitProgrammingGuide/Introduction/Introduction.html)

