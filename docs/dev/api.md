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

## Audio Converter Framework - IMPLEMENTED

### AudioConverter Operations - IMPLEMENTED

```python
class AudioConverterError(CoreAudioError):
    """Exception for AudioConverter operations"""

class AudioConverter(CoreAudioObject):
    """Audio format converter for sample rate and format conversion

    Provides high-level interface for converting between audio formats,
    sample rates, bit depths, and channel configurations.

    Supports ALL conversion types:
        - Sample rate changes (e.g., 44.1kHz → 48kHz, 48kHz → 96kHz)
        - Bit depth changes (e.g., 16-bit → 24-bit)
        - Channel count changes (stereo ↔ mono)
        - Codec conversions (via ExtendedAudioFile)
        - Combined conversions (any combination of the above)
    """

    def __init__(self, source_format: AudioFormat, dest_format: AudioFormat)
        """Create an AudioConverter

        Args:
            source_format: Source audio format
            dest_format: Destination audio format

        Raises:
            AudioConverterError: If converter creation fails
        """

    # Properties - IMPLEMENTED
    @property
    def source_format(self) -> AudioFormat
        """Get source audio format"""

    @property
    def dest_format(self) -> AudioFormat
        """Get destination audio format"""

    # Conversion methods - IMPLEMENTED
    def convert(self, audio_data: bytes) -> bytes
        """Convert audio data using simple buffer API

        Uses AudioConverterConvertBuffer for simple conversions where
        input/output sizes are predictable (channel-only conversions).

        Args:
            audio_data: Input audio data in source format

        Returns:
            Converted audio data in destination format

        Raises:
            AudioConverterError: If conversion fails

        Note:
            For complex conversions (sample rate, bit depth),
            use convert_with_callback() instead.
        """

    def convert_with_callback(self, input_data: bytes, input_packet_count: int,
                             output_packet_count: Optional[int] = None) -> bytes
        """Convert audio using callback-based API for complex conversions

        Uses AudioConverterFillComplexBuffer for complex conversions
        including sample rate changes, bit depth changes, and combinations.

        Args:
            input_data: Input audio data as bytes
            input_packet_count: Number of packets in input data
            output_packet_count: Expected output packets (auto-calculated if None)

        Returns:
            Converted audio data as bytes

        Raises:
            AudioConverterError: If conversion fails

        Example:
            ```python
            # Convert 44.1kHz to 48kHz
            source_format = AudioFormat(44100.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)
            dest_format = AudioFormat(48000.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)

            with AudioConverter(source_format, dest_format) as converter:
                # Read input data
                with AudioFile("input_44100.wav") as af:
                    input_data, packet_count = af.read_packets(0, 999999999)

                # Convert
                output_data = converter.convert_with_callback(input_data, packet_count)

                # Write output
                with ExtendedAudioFile.create("output_48000.wav", 'WAVE', dest_format) as out:
                    num_frames = len(output_data) // dest_format.bytes_per_frame
                    out.write(num_frames, output_data)
            ```
        """

    # Property access - IMPLEMENTED
    def get_property(self, property_id: int) -> bytes
        """Get a property from the converter"""

    def set_property(self, property_id: int, data: bytes) -> None
        """Set a property on the converter"""

    # State management - IMPLEMENTED
    def reset(self) -> None
        """Reset the converter to its initial state"""

    def dispose(self) -> None
        """Dispose of the audio converter"""

    # Context manager support - IMPLEMENTED
    def __enter__(self) -> 'AudioConverter'
    def __exit__(self, exc_type, exc_val, exc_tb) -> None
```

### ExtendedAudioFile Operations - IMPLEMENTED

```python
class ExtendedAudioFile(CoreAudioObject):
    """Extended audio file with automatic format conversion

    Provides high-level file I/O with automatic format conversion.
    Easier to use than AudioFile for common operations.

    Features:
        - Automatic format conversion on read/write
        - Client format property for on-the-fly conversion
        - Simplified file I/O compared to AudioFile
    """

    def __init__(self, path: Union[str, Path])
        """Create an ExtendedAudioFile"""

    # Factory methods - IMPLEMENTED
    @classmethod
    def create(cls, path: Union[str, Path], file_type: str,
              format: AudioFormat) -> 'ExtendedAudioFile'
        """Create a new audio file for writing

        Args:
            path: Path to create
            file_type: File type (e.g., 'WAVE', 'AIFF')
            format: Audio format for the file

        Returns:
            ExtendedAudioFile instance ready for writing
        """

    # Properties - IMPLEMENTED
    @property
    def file_format(self) -> AudioFormat
        """Get the file's native audio format"""

    @property
    def client_format(self) -> Optional[AudioFormat]
        """Get the client format (for automatic conversion)"""

    @client_format.setter
    def client_format(self, format: AudioFormat) -> None
        """Set the client format (enables automatic conversion)"""

    # I/O methods - IMPLEMENTED
    def open(self) -> 'ExtendedAudioFile'
        """Open the audio file"""

    def read(self, num_frames: int) -> Tuple[bytes, int]
        """Read audio frames (automatically converted if client format is set)

        Returns:
            Tuple of (audio_data, frames_read)
        """

    def write(self, num_frames: int, data: bytes) -> None
        """Write audio frames"""

    def close(self) -> None
        """Close the audio file"""

    def dispose(self) -> None
        """Dispose of the file"""

    # Property access - IMPLEMENTED
    def get_property(self, property_id: int) -> bytes
        """Get a file property"""

    def set_property(self, property_id: int, data: bytes) -> None
        """Set a file property"""

    # Context manager support - IMPLEMENTED
    def __enter__(self) -> 'ExtendedAudioFile'
    def __exit__(self, exc_type, exc_val, exc_tb) -> None
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

## High-Level Utilities - IMPLEMENTED

### Audio Analysis - IMPLEMENTED

```python
class AudioAnalyzer:
    """High-level audio analysis utilities"""

    # All methods are static and support both file paths and AudioFile objects

    @staticmethod
    def detect_silence(audio_file: Union[str, AudioFile], threshold_db: float = -40,
                      min_duration: float = 0.5) -> list[tuple[float, float]]
        """Detect silence regions in audio file

        Args:
            audio_file: File path or AudioFile object
            threshold_db: Threshold in decibels (default: -40dB)
            min_duration: Minimum silence duration in seconds (default: 0.5s)

        Returns:
            List of (start_time, end_time) tuples for silence regions

        Requires: NumPy
        """

    @staticmethod
    def get_peak_amplitude(audio_file: Union[str, AudioFile]) -> float
        """Get peak amplitude of audio file

        Returns:
            Peak amplitude (0.0 to 1.0)

        Requires: NumPy
        """

    @staticmethod
    def calculate_rms(audio_file: Union[str, AudioFile]) -> float
        """Calculate RMS (Root Mean Square) amplitude

        Returns:
            RMS amplitude (0.0 to 1.0)

        Requires: NumPy
        """

    @staticmethod
    def get_file_info(audio_file: Union[str, AudioFile]) -> dict
        """Extract comprehensive file metadata

        Returns:
            Dictionary with keys: 'path', 'format', 'duration', 'sample_rate',
            'channels', 'bits_per_channel', 'file_type'
        """
```

### Audio Format Presets - IMPLEMENTED

```python
class AudioFormatPresets:
    """Common audio format presets for convenience"""

    @staticmethod
    def wav_44100_stereo() -> AudioFormat
        """CD quality WAV (44.1kHz, 16-bit, stereo)"""

    @staticmethod
    def wav_44100_mono() -> AudioFormat
        """Mono WAV (44.1kHz, 16-bit, mono)"""

    @staticmethod
    def wav_48000_stereo() -> AudioFormat
        """Pro audio WAV (48kHz, 16-bit, stereo)"""

    @staticmethod
    def wav_96000_stereo() -> AudioFormat
        """High-res WAV (96kHz, 24-bit, stereo)"""
```

### Audio File Operations - IMPLEMENTED

```python
def convert_audio_file(input_path: str, output_path: str,
                       output_format: AudioFormat) -> None
    """Convert a single audio file to a different format

    Supports ALL conversion types:
        - Sample rate changes (e.g., 44.1kHz → 48kHz, 48kHz → 96kHz)
        - Bit depth changes (e.g., 16-bit → 24-bit)
        - Channel count changes (stereo ↔ mono)
        - Combined conversions (any combination of the above)

    Automatically chooses the optimal conversion method:
        - Simple buffer API for channel-only conversions
        - Callback-based API for complex conversions (sample rate, bit depth)

    Args:
        input_path: Input file path
        output_path: Output file path
        output_format: Target AudioFormat

    Example:
        # Sample rate conversion
        convert_audio_file("input_44100.wav", "output_48000.wav",
                          AudioFormatPresets.wav_48000_stereo())

        # Combined conversion (rate + channels)
        output_fmt = AudioFormat(48000.0, 'lpcm', channels_per_frame=1, bits_per_channel=16)
        convert_audio_file("stereo_44100.wav", "mono_48000.wav", output_fmt)
    """

def batch_convert(input_pattern: str, output_format: AudioFormat,
                  output_dir: str = ".", output_extension: str = ".wav",
                  progress_callback: Optional[Callable] = None,
                  overwrite: bool = False) -> list[tuple[str, str]]
    """Batch convert multiple files with glob patterns

    Args:
        input_pattern: Glob pattern (e.g., "*.wav", "audio/**/*.aiff")
        output_format: Target audio format
        output_dir: Output directory (created if needed)
        output_extension: Output file extension (default: ".wav")
        progress_callback: Optional callback(filename, current, total)
        overwrite: Whether to overwrite existing files

    Returns:
        List of (input_path, output_path) tuples for converted files

    Example:
        batch_convert("*.wav", AudioFormatPresets.wav_44100_mono(),
                     output_dir="converted/",
                     progress_callback=lambda f, c, t: print(f"{c}/{t}: {f}"))
    """

def trim_audio(input_path: str, output_path: str,
               start_time: float = 0.0, end_time: Optional[float] = None) -> None
    """Extract time range from audio file

    Args:
        input_path: Source audio file path
        output_path: Destination audio file path
        start_time: Start time in seconds (default: 0.0)
        end_time: End time in seconds (default: None for end of file)

    Note: Currently requires ExtendedAudioFile.write() implementation
    """
```

### Audio Effects Chain - IMPLEMENTED

```python
class AudioEffectsChain:
    """High-level wrapper for AUGraph audio processing chains"""

    def __init__(self)
        """Create a new audio effects chain (wraps AUGraph)"""

    # Node management - IMPLEMENTED
    def add_effect(self, type_code: str, subtype_code: str,
                   manufacturer_code: str) -> int
        """Add an audio effect to the chain

        Args:
            type_code: FourCC type code (e.g., 'aufx' for audio effect)
            subtype_code: FourCC subtype code (e.g., 'dely' for delay)
            manufacturer_code: FourCC manufacturer code (e.g., 'appl' for Apple)

        Returns:
            Node ID for connecting to other nodes

        Example:
            delay_node = chain.add_effect('aufx', 'dely', 'appl')
        """

    def add_effect_by_name(self, name: str) -> Optional[int]
        """Add an audio effect by name (convenience method)

        Args:
            name: AudioUnit name (e.g., 'AUDelay', 'Reverb')

        Returns:
            Node ID if found, None if AudioUnit not found

        Example:
            delay_node = chain.add_effect_by_name('AUDelay')
        """

    def add_output(self) -> int
        """Add output node to the chain

        Returns:
            Output node ID
        """

    def connect(self, source_node: int, dest_node: int,
                source_bus: int = 0, dest_bus: int = 0) -> 'AudioEffectsChain'
        """Connect two nodes in the chain

        Returns:
            Self for method chaining
        """

    # Lifecycle management - IMPLEMENTED
    def open(self) -> 'AudioEffectsChain'
        """Open the audio graph (returns self for chaining)"""

    def initialize(self) -> 'AudioEffectsChain'
        """Initialize the audio graph (returns self for chaining)"""

    def start(self) -> 'AudioEffectsChain'
        """Start audio processing (returns self for chaining)"""

    def stop(self) -> 'AudioEffectsChain'
        """Stop audio processing (returns self for chaining)"""

    def dispose(self) -> None
        """Clean up graph resources"""

    # Properties - IMPLEMENTED
    @property
    def node_count(self) -> int
        """Number of nodes in the graph"""

    @property
    def is_open(self) -> bool
        """Whether the graph is open"""

    @property
    def is_initialized(self) -> bool
        """Whether the graph is initialized"""

    @property
    def is_running(self) -> bool
        """Whether the graph is running"""

    # Context manager support - IMPLEMENTED
    def __enter__(self) -> 'AudioEffectsChain'
        """Context manager entry (auto-opens, initializes, starts)"""

    def __exit__(self, exc_type, exc_val, exc_tb) -> None
        """Context manager exit (auto-stops and disposes)"""

    # Convenience method - IMPLEMENTED
    @staticmethod
    def create_simple_effect_chain(effects: list[tuple[str, str, str]]) -> 'AudioEffectsChain'
        """Create and configure a simple effect chain

        Args:
            effects: List of (type, subtype, manufacturer) tuples

        Returns:
            Configured AudioEffectsChain (not yet opened)

        Example:
            chain = AudioEffectsChain.create_simple_effect_chain([
                ('aufx', 'dely', 'appl'),  # Delay
                ('aufx', 'rvb2', 'appl'),  # Reverb
            ])
            chain.open().initialize().start()
        """

# Module-level convenience function
def create_simple_effect_chain(effects: list[tuple[str, str, str]]) -> 'AudioEffectsChain'
    """Create and configure a simple effect chain (module-level function)

    Same as AudioEffectsChain.create_simple_effect_chain()
    """
```

### AudioUnit Discovery - IMPLEMENTED

```python
def find_audio_unit_by_name(name: str, case_sensitive: bool = False) -> Optional[AudioComponent]
    """Find an AudioUnit by name

    Args:
        name: AudioUnit name to search for (e.g., 'AUDelay', 'Reverb')
        case_sensitive: Whether to use case-sensitive matching (default: False)

    Returns:
        AudioComponent object if found, None otherwise

    Features:
        - Substring matching (searching for 'Delay' finds 'AUDelay')
        - Case-insensitive by default
        - Returns AudioComponent that can create instances directly

    Example:
        component = find_audio_unit_by_name('AUDelay')
        if component:
            unit = component.create_instance()
            unit.initialize()
            # ... use the AudioUnit
            unit.dispose()
    """

def list_available_audio_units(filter_type: Optional[str] = None) -> list[dict]
    """List all available AudioUnits on the system

    Args:
        filter_type: Optional FourCC type code to filter (e.g., 'aufx' for effects)

    Returns:
        List of dictionaries with keys: 'name', 'type', 'subtype',
        'manufacturer', 'flags'

    Example:
        # List all AudioUnits
        all_units = list_available_audio_units()
        print(f"Found {len(all_units)} AudioUnits")

        # List only audio effects
        effects = list_available_audio_units(filter_type='aufx')
        for effect in effects:
            print(f"{effect['name']}: {effect['type']}/{effect['subtype']}/{effect['manufacturer']}")
    """

def get_audiounit_names(filter_type: Optional[str] = None) -> list[str]
    """Get a simple list of AudioUnit names

    Args:
        filter_type: Optional FourCC type code to filter (e.g., 'aufx' for effects)

    Returns:
        List of AudioUnit names as strings

    Example:
        names = get_audiounit_names()
        # Returns: ['Apple PAC3 Transcoder', 'AUDelay', 'AUReverb', ...]

        effect_names = get_audiounit_names(filter_type='aufx')
        # Returns only effect names
    """
```

### Async I/O Support - IMPLEMENTED

```python
class AsyncAudioFile:
    """Asynchronous audio file operations for non-blocking I/O"""

    def __init__(self, path: Union[str, Path])

    # Async context manager - IMPLEMENTED
    async def __aenter__(self) -> 'AsyncAudioFile'
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None

    # Async methods - IMPLEMENTED
    async def open_async(self) -> 'AsyncAudioFile'
    async def close_async(self) -> None
    async def read_packets_async(self, start_packet: int, packet_count: int) -> tuple[bytes, int]
    async def read_chunks_async(self, chunk_size: int = 4096) -> AsyncIterator[bytes]
    async def read_as_numpy_async(self) -> Any  # Returns np.ndarray (requires NumPy)
    async def read_chunks_numpy_async(self, chunk_size: int = 1024) -> AsyncIterator[Any]

    # Properties - IMPLEMENTED
    @property
    def format(self) -> AudioFormat
    @property
    def duration(self) -> float

# Convenience functions - IMPLEMENTED
async def open_audio_file_async(path: Union[str, Path]) -> AsyncAudioFile
    """Open audio file asynchronously"""

class AsyncAudioQueue:
    """Asynchronous audio queue operations"""

    @classmethod
    async def new_output_async(cls, audio_format: AudioFormat) -> 'AsyncAudioQueue'
        """Create output queue asynchronously"""

    async def start_async(self) -> None
    async def stop_async(self, immediate: bool = True) -> None
    async def allocate_buffer_async(self, buffer_size: int) -> AudioBuffer

    # Async context manager - IMPLEMENTED
    async def __aenter__(self) -> 'AsyncAudioQueue'
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None
```

## Summary

### Current Implementation Status

**IMPLEMENTED (13 core classes + high-level utilities):**
- **Base Infrastructure**: `CoreAudioObject`, `AudioFormat`, `AudioBuffer`
- **Exception Hierarchy**: `CoreAudioError`, `AudioFileError`, `AudioQueueError`, `AudioUnitError`, `AudioConverterError`, `MIDIError`
- **Audio File Framework**: `AudioFile` (partial), `AudioFileStream` (partial)
- **Audio Converter Framework**: `AudioConverter`, `ExtendedAudioFile`
- **Audio Queue Framework**: `AudioQueue` (partial), `AudioBuffer`
- **AudioUnit Framework**: `AudioComponentDescription`, `AudioComponent`, `AudioUnit` (partial)
- **MIDI Framework**: `MIDIClient`, `MIDIPort`, `MIDIInputPort`, `MIDIOutputPort`
- **High-Level Utilities**: `AudioAnalyzer`, `AudioFormatPresets`, `AudioEffectsChain`
- **AudioUnit Discovery**: `find_audio_unit_by_name()`, `list_available_audio_units()`, `get_audiounit_names()`
- **File Operations**: `convert_audio_file()`, `batch_convert()`, `trim_audio()`
- **Async I/O**: `AsyncAudioFile`, `AsyncAudioQueue`

**Current Features:**
- **Automatic resource management** with Cython `__dealloc__` and context managers
- **Type safety** with proper Python classes instead of integer IDs
- **Exception hierarchy** for consistent error handling
- **Context manager support** (`with` statements) for safe resource handling
- **Complex audio conversions** supporting sample rate, bit depth, and channel changes
  - Callback-based AudioConverter API with automatic method selection
  - Duration-preserving conversions (< 0.000003s error verified)
  - Support for all PCM format conversions
- **Async/await support** for non-blocking audio operations (AsyncAudioFile, AsyncAudioQueue)
- **High-level utilities** for common audio tasks (analysis, conversion, batch processing)
- **AudioUnit discovery** by name (find AudioUnits without knowing FourCC codes)
- **Audio effects chains** with Pythonic AUGraph wrapper
- **NumPy integration** for efficient audio data processing
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
