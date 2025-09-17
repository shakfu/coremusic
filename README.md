# coremusic

An early stage Cython wrapper for Apple's CoreAudio and CoreMIDI ecosystem, providing Python bindings for professional audio and MIDI development on macOS. This project exposes a subset of CoreAudio and CoreMIDI C APIs through Python, enabling advanced audio applications, real-time processing, MIDI routing, and professional audio software development.

## Overview

`coremusic` is a c-based Python extension that provides direct access to Apple's CoreAudio frameworks. Built with Cython, it offers near-native performance while maintaining the ease of use of Python. The wrapper covers the CoreAudio ecosystem, from low-level hardware control to high-level audio processing units.

### Key Features

- **CoreAudio Framework Coverage**: Full access to CoreAudio, AudioToolbox, and AudioUnit APIs

- **High Performance**: Cython-based implementation with near-native C performance

- **Professional Audio Support**: Real-time audio processing, multi-channel audio, and hardware control

- **Audio File I/O**: Support for WAV, AIFF, MP3, and other audio formats through CoreAudio

- **AudioUnit Integration**:  AudioUnit discovery, instantiation, and lifecycle management

- **AudioQueue Support**: High-level audio queue management for streaming and playback

- **Hardware Abstraction**: Direct access to audio hardware and device management

- **Format Detection**: Automatic audio format detection and conversion

- **Real-time Processing**: Low-latency audio processing capabilities

- ** CoreMIDI Framework Coverage**: Full access to MIDI services, device management, and advanced routing

- **Universal MIDI Packet Support**: MIDI 1.0 and 2.0 message creation and handling in UMP format

- **MIDI Device Management**:  device and entity discovery, creation, and control

- **MIDI Routing and Transformation**: Advanced MIDI thru connections with filtering and transformation

- **MIDI Driver APIs**: Access to MIDI driver development and device integration functions

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

### CoreMIDI

- MIDI Services:  CoreMIDI framework integration with device and endpoint management
- Universal MIDI Packets: MIDI 1.0 and 2.0 message creation and handling in UMP format
- Message Creation: Channel voice, system, and meta message construction with type safety
- Device Management: MIDI device and entity discovery, creation, and property management
- MIDI Setup: External device handling, device list management, and system configuration
- Driver Support: Access to MIDI driver APIs for device creation and endpoint management
- Thru Connections: Advanced MIDI routing with filtering, transformation, and channel mapping
- Message Transformation: Scale, filter, add, and remap MIDI messages with flexible transforms
- Real-time MIDI: Low-latency MIDI processing with proper timestamp handling

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
    # or run tests individually
    pytest -v tests/test_coremidi.py
    ```

## Quick Start

### Basic Audio File Operations

```python
import coremusic as cm

# Open an audio file
audio_file = cm.audio_file_open_url(
    "path/to/audio.wav",
    cm.get_audio_file_read_permission(),
    cm.get_audio_file_wave_type()
)

# Get file format information
format_data = cm.audio_file_get_property(
    audio_file,
    cm.get_audio_file_property_data_format()
)

# Read audio packets
packet_data, packets_read = cm.audio_file_read_packets(audio_file, 0, 1000)

# Close the file
cm.audio_file_close(audio_file)
```

### AudioUnit Setup

```python
import coremusic as cm

# Find default output AudioUnit
description = {
    'type': cm.get_audio_unit_type_output(),
    'subtype': cm.get_audio_unit_subtype_default_output(),
    'manufacturer': cm.get_audio_unit_manufacturer_apple(),
    'flags': 0,
    'flags_mask': 0
}

component_id = cm.audio_component_find_next(description)
audio_unit = cm.audio_component_instance_new(component_id)

# Initialize and start
cm.audio_unit_initialize(audio_unit)
cm.audio_output_unit_start(audio_unit)

# ... perform audio operations ...

# Cleanup
cm.audio_output_unit_stop(audio_unit)
cm.audio_unit_uninitialize(audio_unit)
cm.audio_component_instance_dispose(audio_unit)
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
player.start()

# Control playback
player.set_looping(True)
print(f"Playing: {player.is_playing()}")
print(f"Progress: {player.get_progress():.2f}")

# Stop playback
player.stop()
```

### CoreMIDI Basic Usage

```python
import coremusic as cm

# Create MIDI 1.0 Universal Packets
ump = cm.midi1_channel_voice_message(
    group=0,
    status=cm.get_midi_status_note_on(),
    channel=0,
    data1=60,  # Middle C
    data2=100  # Velocity
)
print(f"MIDI UMP: {ump:08X}")

# Get MIDI device information
device_count = cm.midi_get_number_of_devices()
print(f"MIDI devices: {device_count}")

source_count = cm.midi_get_number_of_sources()
print(f"MIDI sources: {source_count}")

# Create a virtual MIDI device
try:
    device = cm.midi_device_create("My Virtual Device")
    print(f"Created device: {device}")
except RuntimeError as e:
    print(f"Device creation failed: {e}")
```

### MIDI Thru Connection Example

```python
import coremusic as cm

# Initialize thru connection parameters
params = cm.midi_thru_connection_params_initialize()

# Configure channel mapping (route channel 1 to channel 2)
params['channelMap'][0] = 1  # Channel 1 (0-indexed) -> Channel 2

# Add a note number transform (transpose up one octave)
params['noteNumber'] = {
    'transform': cm.get_midi_transform_add(),
    'value': 12  # Add 12 semitones
}

# Create the thru connection
try:
    connection = cm.midi_thru_connection_create_with_params(params)
    print(f"Created thru connection: {connection}")

    # Clean up
    cm.midi_thru_connection_dispose(connection)
except RuntimeError as e:
    print(f"Thru connection failed: {e}")
```

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

### CoreMIDI Test Suite

```bash
pytest tests/test_coremidi.py -v
```

The CoreMIDI test suite includes:

- Universal MIDI Packet creation and validation
- MIDI device and entity management
- MIDI driver API functionality
- Thru connection routing and transformation
- Error handling and environment adaptation

### Test Suite

```bash
pytest tests/test_coremusic.py tests/test_coremidi.py -v
```

The complete test suite covers:

- Audio file I/O operations
- AudioUnit lifecycle management
- AudioQueue functionality
- MIDI message creation and handling
- MIDI device management and routing
- Error handling and edge cases
- Performance characteristics

## Architecture

### Core Files

- **`src/coremusic/capi.pyx`**: Main Cython implementation with Python wrapper functions
- **`src/coremusic/capi.pxd`**: Main Cython header importing all framework declarations
- **`src/coremusic/coremidi.pxd`**: CoreMIDI framework declarations and structures
- **`src/coremusic/corefoundation.pxd`**: CoreFoundation framework declarations
- **`src/coremusic/audio_player.c/h`**: C implementation of audio player with render callbacks
- **`setup.py`**: Build configuration linking CoreAudio and CoreMIDI frameworks

### Framework Dependencies

The project links against macOS frameworks:

- CoreServices
- CoreFoundation
- AudioUnit
- AudioToolbox
- CoreAudio
- CoreMIDI

Required libraries: m, dl, pthread

### Build Process

To build just:

```sh
make
```

The extension is built using Cython with the following process:

1. Cython compiles `.pyx` files to C
2. C compiler links against CoreAudio frameworks
3. Python extension module is created

To build a wheel:

```sh
make wheel
```


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
- MIDI controllers and sequencers
- Virtual instruments and synthesizers

### MIDI Applications

- MIDI routing and transformation systems
- Virtual MIDI devices and drivers
- MIDI processing and filtering
- Advanced MIDI routing matrices
- MIDI learning and mapping systems
- Real-time MIDI effects and processors

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
- MIDI protocol research and development
- Music technology experimentation

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

