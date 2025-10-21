# CoreMusic: Complete Python bindings for Apple CoreAudio

A comprehensive Cython wrapper for Apple's CoreAudio and CoreMIDI ecosystem, providing both functional and object-oriented Python bindings for professional audio and MIDI development on macOS. This project exposes the complete CoreAudio and CoreMIDI C APIs through Python, enabling advanced audio applications, real-time processing, MIDI routing, and professional audio software development.

## Overview

`coremusic` is a high-performance Python extension that provides direct access to Apple's CoreAudio frameworks. Built with Cython, it offers near-native performance while maintaining the ease of use of Python. The wrapper covers the complete CoreAudio ecosystem, from low-level hardware control to high-level audio processing units, with both traditional functional APIs and modern object-oriented interfaces.

### Key Features

- **Dual API Design**: Both functional (C-style) and object-oriented (Pythonic) APIs available

- **CoreAudio Framework Coverage**: Full access to CoreAudio, AudioToolbox, and AudioUnit APIs

- **High Performance**: Cython-based implementation with near-native C performance

- **Automatic Resource Management**: Object-oriented APIs with context managers and automatic cleanup

- **Professional Audio Support**: Real-time audio processing, multi-channel audio, and hardware control

- **Audio File I/O**: Support for WAV, AIFF, MP3, and other audio formats through CoreAudio

- **AudioUnit Integration**: AudioUnit discovery, instantiation, and lifecycle management

- **AudioQueue Support**: High-level audio queue management for streaming and playback

- **Hardware Abstraction**: Direct access to audio hardware and device management

- **Format Detection**: Automatic audio format detection and conversion

- **Real-time Processing**: Low-latency audio processing capabilities

- **CoreMIDI Framework Coverage**: Full access to MIDI services, device management, and advanced routing

- **Universal MIDI Packet Support**: MIDI 1.0 and 2.0 message creation and handling in UMP format

- **MIDI Device Management**: Device and entity discovery, creation, and control

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

# Open an audio file
audio_file = capi.audio_file_open_url("path/to/audio.wav")

# Get file format information
format_data = capi.audio_file_get_property(
    audio_file,
    capi.get_audio_file_property_data_format()
)

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

## Examples and Demos

The project includes comprehensive demonstration scripts and test suites:

### Object-Oriented API Tests

```bash
# Test object-oriented audio file operations
pytest tests/test_objects_audio_file.py -v

# Test object-oriented AudioUnit operations
pytest tests/test_objects_audio_unit.py -v

# Test object-oriented MIDI operations
pytest tests/test_objects_midi.py -v

# Test complete object-oriented API
pytest tests/test_objects_comprehensive.py -v
```

### Unified Audio Demo

```bash
python3 tests/demos/unified_audio_demo.py
```

This comprehensive demo showcases:

- CoreAudio constants and utilities
- Audio file operations and format detection (both APIs)
- AudioUnit infrastructure testing (both APIs)
- AudioQueue operations (both APIs)
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

### Complete Test Suite

```bash
# Run all tests
make test

# Run specific test categories
pytest tests/test_coremidi.py tests/test_objects_*.py -v
```

The complete test suite covers:

- **Functional API**: Audio file I/O, AudioUnit lifecycle, AudioQueue functionality
- **Object-Oriented API**: Modern Pythonic wrappers with automatic resource management
- **MIDI Operations**: Message creation, device management, and routing (both APIs)
- **Integration Testing**: Cross-API compatibility and consistency
- **Resource Management**: Automatic cleanup and disposal testing
- **Error handling**: Edge cases and failure scenarios
- **Performance characteristics**: Real-time audio processing validation

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

- **`src/coremusic/objects.pyx`**: Cython extension base class for automatic resource management
- **`src/coremusic/oo.py`**: Object-oriented wrappers with automatic cleanup and context managers
- **`src/coremusic/__init__.py`**: Package entry point exposing OO API (functional API via `capi` submodule)

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

### Migration Guide

**Migrating to new namespace (functional API users):**

```python
# Before (old pattern - deprecated)
import coremusic as cm
audio_file = cm.audio_file_open_url("file.wav")  # No longer works

# After (new pattern - functional API)
import coremusic.capi as capi
audio_file = capi.audio_file_open_url("file.wav")  # Correct
```

**Migrating to object-oriented API (recommended):**

```python
# Before (Functional API)
import coremusic.capi as capi
audio_file = capi.audio_file_open_url("file.wav")
try:
    format_data = capi.audio_file_get_property(audio_file, property_id)
    data, count = capi.audio_file_read_packets(audio_file, 0, 1000)
finally:
    capi.audio_file_close(audio_file)

# After (Object-Oriented API)
import coremusic as cm
with cm.AudioFile("file.wav") as audio_file:
    format_info = audio_file.format
    data, count = audio_file.read_packets(0, 1000)
```

### Best Practices

- **Resource Management**: Always use context managers (`with` statements) when possible
- **Error Handling**: Both APIs provide consistent exception types (`AudioFileError`, `AudioUnitError`, etc.)
- **Performance**: Object-oriented layer adds minimal overhead - choose based on development needs
- **Mixing APIs**: Both APIs can be used together - OO objects expose their underlying IDs when needed

## Performance

coremusic provides near-native performance through both APIs:

**Functional API:**

- Direct C API access with zero Python overhead
- Explicit memory management for optimal control
- Maximum performance for real-time audio processing

**Object-Oriented API:**

- Minimal overhead layer built on functional API
- Automatic resource management without performance penalty
- Efficient Cython-based cleanup via `__dealloc__`

**Common Performance Features:**

- Optimized audio processing pipelines
- Real-time audio callback support
- Efficient memory management
- Direct framework integration

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
