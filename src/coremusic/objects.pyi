"""Type stubs for coremusic.objects - Object-Oriented CoreAudio API

This module provides Pythonic, object-oriented wrappers around the CoreAudio
functional API with automatic resource management and context manager support.
"""

from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

from . import capi

# NumPy types (conditional)
try:
    import numpy as np
    from numpy.typing import NDArray
except ImportError:
    NDArray = Any  # type: ignore
    np = None  # type: ignore

# Module-level flag
NUMPY_AVAILABLE: bool

# ============================================================================
# Exception Hierarchy
# ============================================================================

class CoreAudioError(Exception):
    """Base exception for CoreAudio errors"""

    status_code: int
    def __init__(self, message: str, status_code: int = 0) -> None: ...

class AudioFileError(CoreAudioError):
    """Exception for AudioFile operations"""

    ...

class AudioQueueError(CoreAudioError):
    """Exception for AudioQueue operations"""

    ...

class AudioUnitError(CoreAudioError):
    """Exception for AudioUnit operations"""

    ...

class AudioConverterError(CoreAudioError):
    """Exception for AudioConverter operations"""

    ...

class MIDIError(CoreAudioError):
    """Exception for MIDI operations"""

    ...

class MusicPlayerError(CoreAudioError):
    """Exception for MusicPlayer operations"""

    ...

class AudioDeviceError(CoreAudioError):
    """Exception for AudioDevice operations"""

    ...

class AUGraphError(CoreAudioError):
    """Exception for AUGraph operations"""

    ...

# ============================================================================
# Audio Format
# ============================================================================

class AudioFormat:
    """Pythonic representation of AudioStreamBasicDescription"""

    sample_rate: float
    format_id: str
    format_flags: int
    bytes_per_packet: int
    frames_per_packet: int
    bytes_per_frame: int
    channels_per_frame: int
    bits_per_channel: int

    def __init__(
        self,
        sample_rate: float,
        format_id: str,
        format_flags: int = 0,
        bytes_per_packet: int = 0,
        frames_per_packet: int = 0,
        bytes_per_frame: int = 0,
        channels_per_frame: int = 2,
        bits_per_channel: int = 16,
    ) -> None: ...
    @property
    def is_pcm(self) -> bool:
        """Check if this is a PCM format"""
        ...

    @property
    def is_stereo(self) -> bool:
        """Check if this is stereo (2 channels)"""
        ...

    @property
    def is_mono(self) -> bool:
        """Check if this is mono (1 channel)"""
        ...

    @property
    def is_float(self) -> bool:
        """Check if this is floating point format"""
        ...

    @property
    def is_integer(self) -> bool:
        """Check if this is integer format"""
        ...

    @property
    def is_signed(self) -> bool:
        """Check if this is signed integer format"""
        ...

    @property
    def is_interleaved(self) -> bool:
        """Check if this is interleaved format"""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AudioFormat:
        """Create AudioFormat from dictionary"""
        ...

    def to_numpy_dtype(self) -> Any:
        """Convert format to NumPy dtype (requires NumPy)"""
        ...

    @classmethod
    def pcm_16bit_stereo(cls, sample_rate: float = 44100.0) -> AudioFormat:
        """Create 16-bit stereo PCM format"""
        ...

    @classmethod
    def pcm_24bit_stereo(cls, sample_rate: float = 44100.0) -> AudioFormat:
        """Create 24-bit stereo PCM format"""
        ...

    @classmethod
    def pcm_32bit_float_stereo(cls, sample_rate: float = 44100.0) -> AudioFormat:
        """Create 32-bit float stereo PCM format"""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

# ============================================================================
# Audio File
# ============================================================================

class AudioFile(capi.CoreAudioObject):
    """High-level audio file operations with automatic resource management"""

    def __init__(self, path: Union[str, Path], mode: str = "r") -> None: ...
    def __enter__(self) -> AudioFile: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None: ...
    @property
    def path(self) -> Path:
        """File path"""
        ...

    @property
    def format(self) -> AudioFormat:
        """Audio format information"""
        ...

    @property
    def duration(self) -> float:
        """Duration in seconds"""
        ...

    @property
    def sample_rate(self) -> float:
        """Sample rate in Hz"""
        ...

    @property
    def num_channels(self) -> int:
        """Number of audio channels"""
        ...

    @property
    def frame_count(self) -> int:
        """Total number of frames"""
        ...

    @property
    def packet_count(self) -> int:
        """Total number of packets"""
        ...

    def open(self) -> None:
        """Open the audio file"""
        ...

    def close(self) -> None:
        """Close the audio file"""
        ...

    def read_frames(self, start: int = 0, count: Optional[int] = None) -> bytes:
        """Read audio frames as raw bytes"""
        ...

    def read_packets(self, num_packets: int = 1024) -> Iterator[bytes]:
        """Iterator over audio packets"""
        ...

    def read_as_numpy(
        self, start_packet: int = 0, packet_count: Optional[int] = None
    ) -> NDArray:
        """Read audio data as NumPy array (requires NumPy)"""
        ...

    def __repr__(self) -> str: ...

# ============================================================================
# Audio File Stream
# ============================================================================

class AudioFileStream(capi.CoreAudioObject):
    """Streaming audio file parser"""

    def __init__(self, file_type_hint: str = "") -> None: ...
    def __enter__(self) -> AudioFileStream: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None: ...
    @property
    def format(self) -> Optional[AudioFormat]:
        """Audio format (available after parsing begins)"""
        ...

    @property
    def is_ready(self) -> bool:
        """Check if ready to produce packets"""
        ...

    def parse_bytes(self, data: bytes, discontinuity: bool = False) -> Dict[str, Any]:
        """Parse audio data bytes"""
        ...

    def seek(self, packet_offset: int) -> int:
        """Seek to packet offset"""
        ...

    def open(self) -> None:
        """Open the stream"""
        ...

    def close(self) -> None:
        """Close the stream"""
        ...

    def __repr__(self) -> str: ...

# ============================================================================
# Audio Converter
# ============================================================================

class AudioConverter(capi.CoreAudioObject):
    """Audio format converter"""

    def __init__(
        self, source_format: AudioFormat, dest_format: AudioFormat
    ) -> None: ...
    def __enter__(self) -> AudioConverter: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None: ...
    @property
    def source_format(self) -> AudioFormat:
        """Source audio format"""
        ...

    @property
    def dest_format(self) -> AudioFormat:
        """Destination audio format"""
        ...

    def convert(self, input_data: bytes) -> bytes:
        """Convert audio data"""
        ...

    def convert_numpy(self, input_array: NDArray) -> NDArray:
        """Convert NumPy audio array (requires NumPy)"""
        ...

    def set_quality(self, quality: int) -> None:
        """Set converter quality (0-127)"""
        ...

    def reset(self) -> None:
        """Reset converter state"""
        ...

    def open(self) -> None:
        """Initialize the converter"""
        ...

    def close(self) -> None:
        """Dispose of the converter"""
        ...

    def __repr__(self) -> str: ...

# ============================================================================
# Extended Audio File
# ============================================================================

class ExtendedAudioFile(capi.CoreAudioObject):
    """High-level audio file I/O with automatic format conversion"""

    def __init__(
        self,
        path: Union[str, Path],
        mode: str = "r",
        file_type: Optional[str] = None,
        format: Optional[AudioFormat] = None,
    ) -> None: ...
    @classmethod
    def create(
        cls, path: Union[str, Path], file_type: int, format: AudioFormat
    ) -> ExtendedAudioFile:
        """Create a new audio file for writing"""
        ...
    def __enter__(self) -> ExtendedAudioFile: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None: ...
    @property
    def path(self) -> Path:
        """File path"""
        ...

    @property
    def file_format(self) -> AudioFormat:
        """File's native audio format"""
        ...

    @property
    def client_format(self) -> AudioFormat:
        """Client audio format (may differ from file format)"""
        ...

    @client_format.setter
    def client_format(self, format: AudioFormat) -> None: ...
    @property
    def frame_count(self) -> int:
        """Total number of frames"""
        ...

    def read(self, num_frames: int) -> Tuple[bytes, int]:
        """Read audio frames with automatic conversion.

        Returns:
            Tuple of (audio_data_bytes, frames_read)
        """
        ...

    def write(self, num_frames: int, audio_data: bytes) -> None:
        """Write audio frames with automatic conversion"""
        ...

    def seek(self, frame_position: int) -> None:
        """Seek to frame position"""
        ...

    def open(self) -> None:
        """Open the file"""
        ...

    def close(self) -> None:
        """Close the file"""
        ...

    def __repr__(self) -> str: ...

# ============================================================================
# Audio Queue
# ============================================================================

class AudioBuffer(capi.CoreAudioObject):
    """Audio queue buffer"""

    def __init__(self, queue_id: int, buffer_size: int) -> None: ...
    @property
    def capacity(self) -> int:
        """Buffer capacity in bytes"""
        ...

    @property
    def data_size(self) -> int:
        """Current data size in bytes"""
        ...

    def dispose(self) -> None:
        """Dispose of the buffer"""
        ...

    def __repr__(self) -> str: ...

class AudioQueue(capi.CoreAudioObject):
    """Audio queue for playback or recording"""

    def __init__(self, format: AudioFormat, is_input: bool = False) -> None: ...
    def __enter__(self) -> AudioQueue: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None: ...
    @property
    def format(self) -> AudioFormat:
        """Audio format"""
        ...

    @property
    def is_input(self) -> bool:
        """Whether this is an input queue"""
        ...

    @property
    def is_running(self) -> bool:
        """Whether the queue is running"""
        ...

    def allocate_buffer(self, buffer_size: int) -> AudioBuffer:
        """Allocate an audio buffer"""
        ...

    def start(self) -> None:
        """Start the queue"""
        ...

    def stop(self, immediate: bool = True) -> None:
        """Stop the queue"""
        ...

    def pause(self) -> None:
        """Pause the queue"""
        ...

    def flush(self) -> None:
        """Flush the queue"""
        ...

    def reset(self) -> None:
        """Reset the queue"""
        ...

    def open(self) -> None:
        """Create the queue"""
        ...

    def close(self) -> None:
        """Dispose of the queue"""
        ...

    def __repr__(self) -> str: ...

# ============================================================================
# Audio Component & AudioUnit
# ============================================================================

class AudioComponentDescription:
    """AudioComponent description for discovery"""

    component_type: str
    component_subtype: str
    component_manufacturer: str
    component_flags: int
    component_flags_mask: int

    def __init__(
        self,
        component_type: str,
        component_subtype: str,
        component_manufacturer: str = "appl",
        component_flags: int = 0,
        component_flags_mask: int = 0,
    ) -> None: ...
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary with FourCC as integers"""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> AudioComponentDescription: ...
    @classmethod
    def default_output(cls) -> AudioComponentDescription:
        """Create description for default output unit"""
        ...

    @classmethod
    def music_device(cls) -> AudioComponentDescription:
        """Create description for music device (synthesizer)"""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class AudioComponent(capi.CoreAudioObject):
    """Audio component for discovery and instantiation"""

    def __init__(self, description: AudioComponentDescription) -> None: ...
    @property
    def name(self) -> str:
        """Component name"""
        ...

    @property
    def description(self) -> AudioComponentDescription:
        """Component description"""
        ...

    def create_instance(self) -> AudioUnit:
        """Create an AudioUnit instance from this component"""
        ...

    @classmethod
    def find(cls, description: AudioComponentDescription) -> Optional[AudioComponent]:
        """Find an audio component matching description"""
        ...

    @classmethod
    def find_all(cls, description: AudioComponentDescription) -> List[AudioComponent]:
        """Find all audio components matching description"""
        ...

    def __repr__(self) -> str: ...

class AudioUnit(capi.CoreAudioObject):
    """AudioUnit for audio processing"""

    def __init__(
        self,
        component: Optional[Union[AudioComponent, AudioComponentDescription]] = None,
    ) -> None: ...
    def __enter__(self) -> AudioUnit: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None: ...
    @property
    def is_initialized(self) -> bool:
        """Whether the unit is initialized"""
        ...

    @property
    def is_running(self) -> bool:
        """Whether the unit is running"""
        ...

    def initialize(self) -> None:
        """Initialize the AudioUnit"""
        ...

    def uninitialize(self) -> None:
        """Uninitialize the AudioUnit"""
        ...

    def start(self) -> None:
        """Start the AudioUnit output"""
        ...

    def stop(self) -> None:
        """Stop the AudioUnit output"""
        ...

    def get_stream_format(self, scope: str = "output", element: int = 0) -> AudioFormat:
        """Get stream format for scope/element"""
        ...

    def set_stream_format(
        self, format: AudioFormat, scope: str = "output", element: int = 0
    ) -> None:
        """Set stream format for scope/element"""
        ...

    def get_sample_rate(self, scope: str = "output", element: int = 0) -> float:
        """Get sample rate"""
        ...

    def set_sample_rate(
        self, sample_rate: float, scope: str = "output", element: int = 0
    ) -> None:
        """Set sample rate"""
        ...

    def get_maximum_frames_per_slice(self) -> int:
        """Get maximum frames per slice"""
        ...

    def set_maximum_frames_per_slice(self, frames: int) -> None:
        """Set maximum frames per slice"""
        ...

    @classmethod
    def default_output(cls) -> AudioUnit:
        """Create default output AudioUnit"""
        ...

    @classmethod
    def music_device(cls) -> AudioUnit:
        """Create music device AudioUnit (synthesizer)"""
        ...

    def open(self) -> None:
        """Create the AudioUnit instance"""
        ...

    def close(self) -> None:
        """Dispose of the AudioUnit"""
        ...

    def __repr__(self) -> str: ...

# ============================================================================
# MIDI
# ============================================================================

class MIDIPort(capi.CoreAudioObject):
    """Base class for MIDI ports"""

    def __init__(self, client_id: int, name: str, is_input: bool) -> None: ...
    @property
    def name(self) -> str:
        """Port name"""
        ...

    @property
    def is_input(self) -> bool:
        """Whether this is an input port"""
        ...

    def dispose(self) -> None:
        """Dispose of the port"""
        ...

    def __repr__(self) -> str: ...

class MIDIInputPort(MIDIPort):
    """MIDI input port"""

    def connect_source(self, source_id: int) -> None:
        """Connect to a MIDI source"""
        ...

    def disconnect_source(self, source_id: int) -> None:
        """Disconnect from a MIDI source"""
        ...

class MIDIOutputPort(MIDIPort):
    """MIDI output port"""

    def send(self, destination_id: int, data: bytes, timestamp: int = 0) -> None:
        """Send MIDI data to destination"""
        ...

class MIDIClient(capi.CoreAudioObject):
    """MIDI client for managing MIDI connections"""

    def __init__(self, name: str) -> None: ...
    def __enter__(self) -> MIDIClient: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None: ...
    @property
    def name(self) -> str:
        """Client name"""
        ...

    def create_input_port(self, port_name: str) -> MIDIInputPort:
        """Create a MIDI input port"""
        ...

    def create_output_port(self, port_name: str) -> MIDIOutputPort:
        """Create a MIDI output port"""
        ...

    @staticmethod
    def get_sources() -> List[Tuple[int, str]]:
        """Get all MIDI sources as (id, name) tuples"""
        ...

    @staticmethod
    def get_destinations() -> List[Tuple[int, str]]:
        """Get all MIDI destinations as (id, name) tuples"""
        ...

    def open(self) -> None:
        """Create the MIDI client"""
        ...

    def close(self) -> None:
        """Dispose of the MIDI client"""
        ...

    def __repr__(self) -> str: ...

# ============================================================================
# Audio Device & Hardware
# ============================================================================

class AudioDevice(capi.CoreAudioObject):
    """Audio hardware device"""

    def __init__(self, device_id: int) -> None: ...
    @property
    def device_id(self) -> int:
        """Device ID"""
        ...

    @property
    def name(self) -> str:
        """Device name"""
        ...

    @property
    def manufacturer(self) -> str:
        """Device manufacturer"""
        ...

    @property
    def uid(self) -> str:
        """Device UID (unique identifier)"""
        ...

    @property
    def model_uid(self) -> str:
        """Device model UID"""
        ...

    @property
    def transport_type(self) -> int:
        """Transport type (USB, PCI, etc.)"""
        ...

    @property
    def sample_rate(self) -> float:
        """Current sample rate"""
        ...

    @sample_rate.setter
    def sample_rate(self, rate: float) -> None: ...
    @property
    def available_sample_rates(self) -> List[float]:
        """Available sample rates"""
        ...

    @property
    def is_alive(self) -> bool:
        """Whether device is alive"""
        ...

    @property
    def is_hidden(self) -> bool:
        """Whether device is hidden"""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class AudioDeviceManager:
    """Manager for audio devices"""

    @staticmethod
    def get_devices() -> List[AudioDevice]:
        """Get all audio devices"""
        ...

    @staticmethod
    def get_default_output_device() -> AudioDevice:
        """Get default output device"""
        ...

    @staticmethod
    def get_default_input_device() -> AudioDevice:
        """Get default input device"""
        ...

    @staticmethod
    def find_by_name(name: str) -> Optional[AudioDevice]:
        """Find device by name"""
        ...

    @staticmethod
    def find_by_uid(uid: str) -> Optional[AudioDevice]:
        """Find device by UID"""
        ...

    @staticmethod
    def set_default_output_device(device: AudioDevice) -> None:
        """Set default output device"""
        ...

    @staticmethod
    def set_default_input_device(device: AudioDevice) -> None:
        """Set default input device"""
        ...

# ============================================================================
# AUGraph
# ============================================================================

class AUGraph(capi.CoreAudioObject):
    """Audio processing graph"""

    def __init__(self) -> None: ...
    def __enter__(self) -> AUGraph: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None: ...
    @property
    def is_open(self) -> bool:
        """Whether the graph is open"""
        ...

    @property
    def is_initialized(self) -> bool:
        """Whether the graph is initialized"""
        ...

    @property
    def is_running(self) -> bool:
        """Whether the graph is running"""
        ...

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph"""
        ...

    def add_node(self, description: AudioComponentDescription) -> int:
        """Add a node to the graph"""
        ...

    def remove_node(self, node: int) -> None:
        """Remove a node from the graph"""
        ...

    def get_node_audio_unit(self, node: int) -> AudioUnit:
        """Get the AudioUnit for a node"""
        ...

    def connect_nodes(
        self, source_node: int, source_output: int, dest_node: int, dest_input: int
    ) -> None:
        """Connect two nodes"""
        ...

    def disconnect_node_input(self, dest_node: int, dest_input: int) -> None:
        """Disconnect a node input"""
        ...

    def open(self) -> None:
        """Open the graph"""
        ...

    def close(self) -> None:
        """Close the graph"""
        ...

    def initialize(self) -> None:
        """Initialize the graph"""
        ...

    def uninitialize(self) -> None:
        """Uninitialize the graph"""
        ...

    def start(self) -> None:
        """Start the graph"""
        ...

    def stop(self) -> None:
        """Stop the graph"""
        ...

    def update(self) -> bool:
        """Update the graph configuration"""
        ...

    def __repr__(self) -> str: ...
