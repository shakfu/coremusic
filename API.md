# Object-Oriented API for coremusic

This document describes the actual object-oriented implementation in the coremusic Python package, providing Pythonic wrappers around the functional API in `src/coremusic/capi.pyx`.

**Implementation Status**: This document reflects the current state of the object-oriented API as of the latest version. Items marked as "NOT YET IMPLEMENTED" are planned features that are not yet available.

## Architecture Overview

### Base Infrastructure - IMPLEMENTED

```python
class CoreAudioObject:
    """Base Cython extension class providing automatic resource management"""
    # Properties
    @property
    def is_disposed(self) -> bool
    @property
    def object_id(self) -> int

    # Methods
    def dispose(self) -> None
    def _ensure_not_disposed(self) -> None
    def _set_object_id(self, object_id: int) -> None

class AudioFormat:
    """Pythonic representation of AudioStreamBasicDescription"""
    def __init__(self, sample_rate: float, format_id: str, format_flags: int = 0,
                 bytes_per_packet: int = 0, frames_per_packet: int = 0,
                 bytes_per_frame: int = 0, channels_per_frame: int = 2,
                 bits_per_channel: int = 16)

    # Methods
    def to_dict(self) -> dict

class AudioBuffer:
    """Wrapper for audio buffer management"""
    def __init__(self, queue_id: int, buffer_size: int)

    # Properties
    @property
    def buffer_size(self) -> int
```

### Exception Hierarchy - IMPLEMENTED

```python
class CoreAudioError(Exception):
    """Base exception for all CoreAudio errors"""
    def __init__(self, message: str, status_code: int = None)

class AudioFileError(CoreAudioError):
    """Audio file operation errors"""

class AudioQueueError(CoreAudioError):
    """Audio queue operation errors"""

class AudioUnitError(CoreAudioError):
    """AudioUnit operation errors"""

class MIDIError(CoreAudioError):
    """MIDI operation errors"""

class MusicPlayerError(CoreAudioError):
    """Music player operation errors - NOT YET IMPLEMENTED"""
```

## Core Audio File Framework

### Audio File Operations - PARTIALLY IMPLEMENTED

```python
class AudioFile(CoreAudioObject):
    """High-level audio file operations with automatic resource management"""

    def __init__(self, path: Union[str, Path])

    # Properties - IMPLEMENTED
    @property
    def format(self) -> AudioFormat  # Returns placeholder AudioFormat
    @property
    def duration(self) -> float  # Returns 0.0 (placeholder implementation)

    # Properties - NOT YET IMPLEMENTED
    @property
    def frame_count(self) -> int  # NOT YET IMPLEMENTED
    @property
    def packet_count(self) -> int  # NOT YET IMPLEMENTED
    @property
    def max_packet_size(self) -> int  # NOT YET IMPLEMENTED

    # Methods - IMPLEMENTED
    def open(self) -> 'AudioFile'  # Returns self for chaining
    def close(self) -> None
    def read_packets(self, start_packet: int, packet_count: int) -> tuple[bytes, int]
    def get_property(self, property_id: int) -> bytes

    # Methods - NOT YET IMPLEMENTED
    def read_frames(self, start_frame: int = 0, num_frames: int = None) -> bytes  # NOT YET IMPLEMENTED

    # Context manager support - IMPLEMENTED
    def __enter__(self) -> 'AudioFile'
    def __exit__(self, exc_type, exc_val, exc_tb) -> None
```

### Audio File Stream Operations - PARTIALLY IMPLEMENTED

```python
class AudioFileStream(CoreAudioObject):
    """Streaming audio file parser for real-time processing"""

    def __init__(self, file_type_hint: int = 0)

    # Properties - IMPLEMENTED
    @property
    def ready_to_produce_packets(self) -> bool

    # Properties - NOT YET IMPLEMENTED
    @property
    def data_format(self) -> AudioFormat  # NOT YET IMPLEMENTED
    @property
    def packet_count(self) -> int  # NOT YET IMPLEMENTED
    @property
    def byte_count(self) -> int  # NOT YET IMPLEMENTED

    # Methods - IMPLEMENTED
    def open(self) -> 'AudioFileStream'  # Returns self for chaining
    def close(self) -> None
    def parse_bytes(self, data: bytes) -> None  # Auto-opens if not open
    def seek(self, packet_offset: int) -> None  # Raises error if not open
    def get_property(self, property_id: int) -> bytes  # Raises error if not open
```

## Audio Queue Framework

### Audio Queue Operations - PARTIALLY IMPLEMENTED

```python
class AudioQueue(CoreAudioObject):
    """Audio queue for buffered playback and recording"""

    def __init__(self, audio_format: AudioFormat)

    # Factory methods - IMPLEMENTED
    @classmethod
    def new_output(cls, audio_format: AudioFormat) -> 'AudioQueue'

    # Factory methods - NOT YET IMPLEMENTED
    @classmethod
    def new_input(cls, audio_format: AudioFormat) -> 'AudioQueue'  # NOT YET IMPLEMENTED

    # Buffer management - IMPLEMENTED
    def allocate_buffer(self, buffer_size: int) -> 'AudioBuffer'
    def enqueue_buffer(self, buffer: 'AudioBuffer') -> None

    # Playback control - IMPLEMENTED
    def start(self) -> None
    def stop(self, immediate: bool = True) -> None
    def dispose(self, immediate: bool = True) -> None

    # Playback control - NOT YET IMPLEMENTED
    def pause(self) -> None  # NOT YET IMPLEMENTED
```

## Audio Component & AudioUnit Framework

### Audio Component Operations - IMPLEMENTED

```python
class AudioComponentDescription:
    """Pythonic representation of AudioComponent description"""
    def __init__(self, type: str, subtype: str, manufacturer: str, flags: int = 0, flags_mask: int = 0)

    # Methods - IMPLEMENTED
    def to_dict(self) -> dict  # Returns fourcc integer values

class AudioComponent(CoreAudioObject):
    """Audio component discovery and management"""

    def __init__(self, description: AudioComponentDescription)

    # Class methods - IMPLEMENTED
    @classmethod
    def find_next(cls, description: AudioComponentDescription) -> Optional['AudioComponent']

    # Class methods - NOT YET IMPLEMENTED
    @classmethod
    def find_all(cls, description: AudioComponentDescription) -> list['AudioComponent']  # NOT YET IMPLEMENTED

    # Methods - IMPLEMENTED
    def create_instance(self) -> 'AudioUnit'
```

### AudioUnit Operations - PARTIALLY IMPLEMENTED

```python
class AudioUnit(CoreAudioObject):
    """Audio processing unit wrapper with lifecycle management"""

    def __init__(self, description: AudioComponentDescription)

    # Factory methods - IMPLEMENTED
    @classmethod
    def default_output(cls) -> 'AudioUnit'

    # Factory methods - NOT YET IMPLEMENTED
    @classmethod
    def music_device(cls) -> 'AudioUnit'  # NOT YET IMPLEMENTED

    # Lifecycle - IMPLEMENTED
    def initialize(self) -> None
    def uninitialize(self) -> None

    # Configuration - IMPLEMENTED
    def set_property(self, property_id: int, scope: int, element: int, data: bytes) -> None
    def get_property(self, property_id: int, scope: int, element: int) -> bytes

    # Configuration - NOT YET IMPLEMENTED
    def set_stream_format(self, format: AudioFormat, scope: str = 'output', element: int = 0) -> None  # NOT YET IMPLEMENTED

    # Playback control - IMPLEMENTED
    def start(self) -> None
    def stop(self) -> None

    # Properties - IMPLEMENTED
    @property
    def is_initialized(self) -> bool

    # Context manager support - IMPLEMENTED
    def __enter__(self) -> 'AudioUnit'  # Auto-initializes
    def __exit__(self, exc_type, exc_val, exc_tb) -> None  # Auto-uninitializes and disposes

    # Music device methods - NOT YET IMPLEMENTED
    def midi_event(self, status: int, data1: int, data2: int, offset: int = 0) -> None  # NOT YET IMPLEMENTED
    def start_note(self, instrument_id: int, group_id: int, pitch: float, velocity: float) -> int  # NOT YET IMPLEMENTED
    def stop_note(self, group_id: int, note_instance_id: int) -> None  # NOT YET IMPLEMENTED
```

## Audio Services Framework - NOT YET IMPLEMENTED

### System Sound Services - NOT YET IMPLEMENTED

```python
# NOTE: This entire framework is NOT YET IMPLEMENTED

class SystemSound(CoreAudioObject):  # NOT YET IMPLEMENTED
    """System sound playbook for alerts and short sounds"""

    @classmethod
    def from_file(cls, file_path: str) -> 'SystemSound'  # NOT YET IMPLEMENTED

    @classmethod
    def user_preferred_alert(cls) -> 'SystemSound'  # NOT YET IMPLEMENTED

    @classmethod
    def vibrate(cls) -> 'SystemSound'  # NOT YET IMPLEMENTED

    def play(self) -> None  # NOT YET IMPLEMENTED
    def play_alert(self) -> None  # NOT YET IMPLEMENTED
    def get_property(self, property_id: int, specifier: int = 0) -> bytes  # NOT YET IMPLEMENTED
    def set_property(self, property_id: int, data: bytes, specifier: int = 0) -> None  # NOT YET IMPLEMENTED
```

## Music Framework - NOT YET IMPLEMENTED

### Music Player Operations - NOT YET IMPLEMENTED

```python
# NOTE: This entire framework is NOT YET IMPLEMENTED

class MusicPlayer(CoreAudioObject):  # NOT YET IMPLEMENTED
    """High-level music sequence playback controller"""

    # All properties and methods are NOT YET IMPLEMENTED
    @property
    def sequence(self) -> 'MusicSequence'  # NOT YET IMPLEMENTED
    @sequence.setter
    def sequence(self, seq: 'MusicSequence') -> None  # NOT YET IMPLEMENTED

    @property
    def time(self) -> float  # NOT YET IMPLEMENTED
    @time.setter
    def time(self, time: float) -> None  # NOT YET IMPLEMENTED

    @property
    def play_rate(self) -> float  # NOT YET IMPLEMENTED
    @play_rate.setter
    def play_rate(self, rate: float) -> None  # NOT YET IMPLEMENTED

    @property
    def is_playing(self) -> bool  # NOT YET IMPLEMENTED

    def preroll(self) -> None  # NOT YET IMPLEMENTED
    def start(self) -> None  # NOT YET IMPLEMENTED
    def stop(self) -> None  # NOT YET IMPLEMENTED
```

### All Other Music Framework Classes - NOT YET IMPLEMENTED

```python
# The following classes are NOT YET IMPLEMENTED:
# - MusicSequence
# - MusicTrack
# - MusicEvent
# - MidiNoteEvent
# - MidiChannelEvent
# - TempoEvent
```

## CoreMIDI Framework - PARTIALLY IMPLEMENTED

### MIDI Client Operations - IMPLEMENTED

```python
class MIDIClient(CoreAudioObject):
    """MIDI client for managing ports and connections"""

    def __init__(self, name: str)

    # Properties - IMPLEMENTED
    @property
    def name(self) -> str

    # Port management - IMPLEMENTED
    def create_input_port(self, port_name: str) -> 'MIDIInputPort'
    def create_output_port(self, port_name: str) -> 'MIDIOutputPort'
```

### MIDI Port Operations - IMPLEMENTED

```python
class MIDIPort(CoreAudioObject):
    """Base class for MIDI ports"""

    def __init__(self, name: str)

    # Properties - IMPLEMENTED
    @property
    def name(self) -> str

class MIDIInputPort(MIDIPort):
    """MIDI input port for receiving data"""

    # Methods - IMPLEMENTED
    def connect_source(self, source) -> None  # source is any object with object_id
    def disconnect_source(self, source) -> None

class MIDIOutputPort(MIDIPort):
    """MIDI output port for sending data"""

    # Methods - IMPLEMENTED
    def send_data(self, destination, data: bytes, timestamp: int = 0) -> None  # destination is any object with object_id
```

### All Other MIDI Framework Classes - NOT YET IMPLEMENTED

```python
# The following MIDI classes are NOT YET IMPLEMENTED:
# - MIDIDevice
# - MIDIEntity
# - MIDIEndpoint
# - MIDIObject
# - MIDISystem
# - MIDIMessage
# - MIDI1Message
# - MIDI2Message
# - MIDIThruConnection
# - MIDIThruConnectionParams
# - MIDIDeviceList
```

## Summary

### Current Implementation Status

**IMPLEMENTED (11 core classes):**
- **Base Infrastructure**: `CoreAudioObject`, `AudioFormat`, `AudioBuffer`
- **Exception Hierarchy**: `CoreAudioError`, `AudioFileError`, `AudioQueueError`, `AudioUnitError`, `MIDIError`
- **Audio File Framework**: `AudioFile` (partial), `AudioFileStream` (partial)
- **Audio Queue Framework**: `AudioQueue` (partial), `AudioBuffer`
- **AudioUnit Framework**: `AudioComponentDescription`, `AudioComponent`, `AudioUnit` (partial)
- **MIDI Framework**: `MIDIClient`, `MIDIPort`, `MIDIInputPort`, `MIDIOutputPort`

**Current Features:**
- **Automatic resource management** with Cython `__dealloc__` and context managers
- **Type safety** with proper Python classes instead of integer IDs
- **Exception hierarchy** for consistent error handling
- **Context manager support** (`with` statements) for safe resource handling
- **Backward compatibility** - functional API remains fully available

**NOT YET IMPLEMENTED:**
- Music framework (MusicPlayer, MusicSequence, MusicTrack, etc.)
- System Sound Services
- Advanced MIDI classes (MIDIDevice, MIDIEndpoint, MIDISystem, etc.)
- MIDI message construction classes
- MIDI thru connections
- Device list management
- Many AudioFile and AudioUnit advanced features

**Development Approach:**
The implementation follows a pragmatic approach, implementing core functionality first with automatic resource management and safety features. The functional API provides complete access to all CoreAudio features, while the object-oriented layer adds modern Python conveniences on top of the most commonly used operations.

The design preserves the full power of CoreAudio while providing a more accessible, safer interface for Python developers.
