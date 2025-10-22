#!/usr/bin/env python3
"""Object-oriented Python classes for coremusic.

This module provides Pythonic, object-oriented wrappers around the CoreAudio
functional API. These classes handle resource management automatically and
provide a more intuitive interface for CoreAudio development.

All classes inherit from capi.CoreAudioObject (Cython extension class) for
automatic resource cleanup, but are implemented as pure Python classes for
simplicity.
"""

from typing import Optional, Union, List, Dict, Any, Tuple, TYPE_CHECKING
from pathlib import Path
import struct

from . import capi

# Check if NumPy is available
try:
    import numpy as np
    from numpy.typing import NDArray

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore
    if TYPE_CHECKING:
        # For type checking purposes when NumPy isn't installed
        from numpy.typing import NDArray

# Re-export base classes and player from capi
CoreAudioObject = capi.CoreAudioObject
AudioPlayer = capi.AudioPlayer  # Audio playback utility class

# ============================================================================
# Exports
# ============================================================================
__all__ = [
    # Base class
    "CoreAudioObject",
    # Exception hierarchy
    "CoreAudioError",
    "AudioFileError",
    "AudioQueueError",
    "AudioUnitError",
    "AudioConverterError",
    "MIDIError",
    "MusicPlayerError",
    "AudioDeviceError",
    "AUGraphError",
    # Audio formats and data structures
    "AudioFormat",
    # Audio File Framework
    "AudioFile",
    "AudioFileStream",
    "ExtendedAudioFile",
    # AudioConverter Framework
    "AudioConverter",
    # Audio Queue Framework
    "AudioBuffer",
    "AudioQueue",
    # Audio Component & AudioUnit Framework
    "AudioComponentDescription",
    "AudioComponent",
    "AudioUnit",
    # MIDI Framework
    "MIDIClient",
    "MIDIPort",
    "MIDIInputPort",
    "MIDIOutputPort",
    # Audio Device & Hardware
    "AudioDevice",
    "AudioDeviceManager",
    # AUGraph Framework
    "AUGraph",
    # CoreAudioClock - Synchronization and Timing
    "AudioClock",
    "ClockTimeFormat",
    # Audio Player
    "AudioPlayer",
    # NumPy availability flag
    "NUMPY_AVAILABLE",
]
# ============================================================================
# Exception Hierarchy
# ============================================================================


class CoreAudioError(Exception):
    """Base exception for CoreAudio errors"""

    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class AudioFileError(CoreAudioError):
    """Exception for AudioFile operations"""

    pass


class AudioQueueError(CoreAudioError):
    """Exception for AudioQueue operations"""

    pass


class AudioUnitError(CoreAudioError):
    """Exception for AudioUnit operations"""

    pass


class MIDIError(CoreAudioError):
    """Exception for MIDI operations"""

    pass


class MusicPlayerError(CoreAudioError):
    """Exception for MusicPlayer operations"""

    pass


class AudioDeviceError(CoreAudioError):
    """Exception for AudioDevice operations"""

    pass


# ============================================================================
# Audio Format
# ============================================================================


class AudioFormat:
    """Pythonic representation of AudioStreamBasicDescription"""

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
    ):
        self.sample_rate = sample_rate
        self.format_id = format_id
        self.format_flags = format_flags
        self.bytes_per_packet = bytes_per_packet
        self.frames_per_packet = frames_per_packet
        self.bytes_per_frame = bytes_per_frame
        self.channels_per_frame = channels_per_frame
        self.bits_per_channel = bits_per_channel

    @property
    def is_pcm(self) -> bool:
        """Check if this is a PCM format"""
        return self.format_id == "lpcm"

    @property
    def is_stereo(self) -> bool:
        """Check if this is stereo (2 channels)"""
        return self.channels_per_frame == 2

    @property
    def is_mono(self) -> bool:
        """Check if this is mono (1 channel)"""
        return self.channels_per_frame == 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for functional API"""
        # Convert format_id string to integer using fourcc_to_int
        from . import capi

        format_id_int = (
            capi.fourchar_to_int(self.format_id)
            if isinstance(self.format_id, str)
            else self.format_id
        )

        return {
            "sample_rate": self.sample_rate,
            "format_id": format_id_int,
            "format_flags": self.format_flags,
            "bytes_per_packet": self.bytes_per_packet,
            "frames_per_packet": self.frames_per_packet,
            "bytes_per_frame": self.bytes_per_frame,
            "channels_per_frame": self.channels_per_frame,
            "bits_per_channel": self.bits_per_channel,
        }

    def to_numpy_dtype(self) -> "np.dtype[Any]":
        """
        Convert audio format to NumPy dtype for audio data arrays.

        Returns:
            NumPy dtype object suitable for audio data representation

        Raises:
            ImportError: If NumPy is not available
            ValueError: If format cannot be converted to NumPy dtype
        """
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "NumPy is not available. Install numpy to use this feature."
            )

        # Handle PCM formats
        if self.is_pcm:
            # Check if float or integer
            is_float = bool(self.format_flags & 1)  # kAudioFormatFlagIsFloat
            is_signed = not bool(
                self.format_flags & 2
            )  # kAudioFormatFlagIsSignedInteger

            if is_float:
                if self.bits_per_channel == 32:
                    return np.dtype(np.float32)
                elif self.bits_per_channel == 64:
                    return np.dtype(np.float64)
                else:
                    raise ValueError(
                        f"Unsupported float bit depth: {self.bits_per_channel}"
                    )
            else:
                # Integer formats
                if self.bits_per_channel == 8:
                    return np.dtype(np.int8 if is_signed else np.uint8)
                elif self.bits_per_channel == 16:
                    return np.dtype(np.int16)
                elif self.bits_per_channel == 24:
                    # 24-bit audio is typically padded to 32-bit
                    return np.dtype(np.int32)
                elif self.bits_per_channel == 32:
                    return np.dtype(np.int32)
                else:
                    raise ValueError(
                        f"Unsupported integer bit depth: {self.bits_per_channel}"
                    )
        else:
            raise ValueError(
                f"Cannot convert non-PCM format '{self.format_id}' to NumPy dtype"
            )

    def __repr__(self) -> str:
        return (
            f"AudioFormat({self.sample_rate}Hz, {self.format_id}, "
            f"channels={self.channels_per_frame}, bits={self.bits_per_channel})"
        )


# ============================================================================
# Audio File Operations
# ============================================================================


class AudioFile(capi.CoreAudioObject):
    """High-level audio file operations with automatic resource management"""

    def __init__(self, path: Union[str, Path]):
        super().__init__()
        self._path = str(path)
        self._format: Optional[AudioFormat] = None
        self._is_open = False

    def open(self) -> "AudioFile":
        """Open the audio file"""
        self._ensure_not_disposed()
        if not self._is_open:
            try:
                file_id = capi.audio_file_open_url(self._path)
                self._set_object_id(file_id)
                self._is_open = True
            except Exception as e:
                raise AudioFileError(f"Failed to open file {self._path}: {e}")
        return self

    def close(self) -> None:
        """Close the audio file"""
        if self._is_open:
            try:
                capi.audio_file_close(self.object_id)
            except Exception as e:
                raise AudioFileError(f"Failed to close file: {e}")
            finally:
                self._is_open = False
                self.dispose()

    def __enter__(self) -> "AudioFile":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @property
    def format(self) -> AudioFormat:
        """Get the audio format of the file"""
        self._ensure_not_disposed()
        if not self._is_open:
            self.open()

        if self._format is None:
            try:
                format_data = capi.audio_file_get_property(
                    self.object_id, capi.get_audio_file_property_data_format()
                )
                # Parse AudioStreamBasicDescription (40 bytes)
                # struct: double + 8 x UInt32
                import struct

                if len(format_data) >= 40:
                    asbd = struct.unpack("<dLLLLLLLL", format_data[:40])
                    (
                        sample_rate,
                        format_id_int,
                        format_flags,
                        bytes_per_packet,
                        frames_per_packet,
                        bytes_per_frame,
                        channels_per_frame,
                        bits_per_channel,
                        reserved,
                    ) = asbd

                    # Convert format_id from integer to fourcc string
                    format_id = capi.int_to_fourchar(format_id_int)

                    self._format = AudioFormat(
                        sample_rate=sample_rate,
                        format_id=format_id,
                        format_flags=format_flags,
                        bytes_per_packet=bytes_per_packet,
                        frames_per_packet=frames_per_packet,
                        bytes_per_frame=bytes_per_frame,
                        channels_per_frame=channels_per_frame,
                        bits_per_channel=bits_per_channel,
                    )
                else:
                    raise AudioFileError(
                        f"Invalid format data size: {len(format_data)} bytes"
                    )
            except Exception as e:
                raise AudioFileError(f"Failed to get format: {e}")

        return self._format

    def read_packets(self, start_packet: int, packet_count: int) -> Tuple[bytes, int]:
        """Read audio packets from the file"""
        self._ensure_not_disposed()
        if not self._is_open:
            self.open()

        try:
            return capi.audio_file_read_packets(
                self.object_id, start_packet, packet_count
            )
        except Exception as e:
            raise AudioFileError(f"Failed to read packets: {e}")

    def read_as_numpy(
        self, start_packet: int = 0, packet_count: Optional[int] = None
    ) -> "NDArray[Any]":
        """
        Read audio data from the file as a NumPy array.

        Args:
            start_packet: Starting packet index (default: 0)
            packet_count: Number of packets to read (default: all remaining packets)

        Returns:
            NumPy array with shape (frames, channels) for multi-channel audio,
            or (frames,) for mono audio. The dtype is determined by the audio format.

        Raises:
            ImportError: If NumPy is not available
            AudioFileError: If reading fails

        Example:
            >>> with AudioFile("audio.wav") as audio:
            ...     data = audio.read_as_numpy()
            ...     print(f"Shape: {data.shape}, dtype: {data.dtype}")
            Shape: (44100, 2), dtype: int16
        """
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "NumPy is not available. Install numpy to use this feature."
            )

        self._ensure_not_disposed()
        if not self._is_open:
            self.open()

        try:
            # Get format information
            format = self.format

            # If packet_count not specified, read all remaining packets
            if packet_count is None:
                # Get total packet count from file
                import struct

                packet_count_data = capi.audio_file_get_property(
                    self.object_id,
                    capi.get_audio_file_property_audio_data_packet_count(),
                )
                if len(packet_count_data) >= 8:
                    total_packets = struct.unpack("<Q", packet_count_data[:8])[0]
                    packet_count = total_packets - start_packet
                else:
                    raise AudioFileError("Cannot determine packet count")

            # Read the raw audio data
            data_bytes, actual_count = capi.audio_file_read_packets(
                self.object_id, start_packet, packet_count
            )

            # Get NumPy dtype from format
            dtype = format.to_numpy_dtype()

            # Convert bytes to NumPy array
            audio_data = np.frombuffer(data_bytes, dtype=dtype)

            # Reshape for multi-channel audio
            # Audio data is typically interleaved: L R L R L R ...
            if format.channels_per_frame > 1:
                # Calculate number of frames
                samples_per_frame = format.channels_per_frame
                num_frames = len(audio_data) // samples_per_frame

                # Reshape to (frames, channels)
                audio_data = audio_data[: num_frames * samples_per_frame].reshape(
                    num_frames, samples_per_frame
                )

            return audio_data

        except Exception as e:
            if isinstance(e, (ImportError, AudioFileError)):
                raise
            raise AudioFileError(f"Failed to read as NumPy array: {e}")

    def get_property(self, property_id: int) -> bytes:
        """Get a property from the audio file"""
        self._ensure_not_disposed()
        if not self._is_open:
            self.open()

        try:
            return capi.audio_file_get_property(self.object_id, property_id)
        except Exception as e:
            raise AudioFileError(f"Failed to get property: {e}")

    @property
    def duration(self) -> float:
        """Duration in seconds"""
        self._ensure_not_disposed()
        if not self._is_open:
            self.open()

        try:
            # Try to get estimated duration property
            import struct

            duration_data = capi.audio_file_get_property(
                self.object_id, capi.get_audio_file_property_estimated_duration()
            )
            if len(duration_data) >= 8:
                # Duration is a Float64 (double)
                duration = struct.unpack("<d", duration_data[:8])[0]
                return duration
            else:
                # Fallback: calculate from packet count and sample rate
                packet_count_data = capi.audio_file_get_property(
                    self.object_id,
                    capi.get_audio_file_property_audio_data_packet_count(),
                )
                if len(packet_count_data) >= 8:
                    packet_count = struct.unpack("<Q", packet_count_data[:8])[0]
                    format = self.format
                    if format.sample_rate > 0:
                        return (
                            packet_count * format.frames_per_packet / format.sample_rate
                        )
                return 0.0
        except Exception:
            # If all methods fail, return 0.0
            return 0.0

    def __repr__(self) -> str:
        status = "open" if self._is_open else "closed"
        return f"AudioFile({self._path}, {status})"

    def dispose(self) -> None:
        """Dispose of the audio file"""
        if not self.is_disposed:
            if self._is_open:
                try:
                    capi.audio_file_close(self.object_id)
                except:
                    pass  # Best effort cleanup
                finally:
                    self._is_open = False
            super().dispose()


class AudioFileStream(capi.CoreAudioObject):
    """Audio file stream for parsing audio data"""

    def __init__(self, file_type_hint: int = 0):
        super().__init__()
        self._file_type_hint = file_type_hint
        self._is_open = False

    def open(self) -> "AudioFileStream":
        """Open the audio file stream"""
        self._ensure_not_disposed()
        if not self._is_open:
            try:
                stream_id = capi.audio_file_stream_open()
                self._set_object_id(stream_id)
                self._is_open = True
            except Exception as e:
                raise AudioFileError(f"Failed to open stream: {e}")
        return self

    def close(self) -> None:
        """Close the audio file stream"""
        if self._is_open:
            try:
                capi.audio_file_stream_close(self.object_id)
            except Exception as e:
                raise AudioFileError(f"Failed to close stream: {e}")
            finally:
                self._is_open = False
                self.dispose()

    @property
    def ready_to_produce_packets(self) -> bool:
        """Check if the stream is ready to produce packets"""
        self._ensure_not_disposed()
        if not self._is_open:
            return False
        try:
            return capi.audio_file_stream_get_property_ready_to_produce_packets(
                self.object_id
            )
        except Exception:
            return False

    def parse_bytes(self, data: bytes) -> None:
        """Parse audio data bytes"""
        self._ensure_not_disposed()
        if not self._is_open:
            self.open()
        try:
            capi.audio_file_stream_parse_bytes(self.object_id, data)
        except Exception as e:
            raise AudioFileError(f"Failed to parse bytes: {e}")

    def seek(self, packet_offset: int) -> None:
        """Seek to packet offset"""
        self._ensure_not_disposed()
        if not self._is_open:
            raise AudioFileError("Stream not open")
        try:
            capi.audio_file_stream_seek(self.object_id, packet_offset)
        except Exception as e:
            raise AudioFileError(f"Failed to seek: {e}")

    def get_property(self, property_id: int) -> bytes:
        """Get a property from the audio file stream"""
        self._ensure_not_disposed()
        if not self._is_open:
            raise AudioFileError("Stream not open")
        try:
            return capi.audio_file_stream_get_property(self.object_id, property_id)
        except Exception as e:
            raise AudioFileError(f"Failed to get property: {e}")

    def dispose(self) -> None:
        """Dispose of the audio file stream"""
        if not self.is_disposed:
            if self._is_open:
                try:
                    capi.audio_file_stream_close(self.object_id)
                except:
                    pass  # Best effort cleanup
                finally:
                    self._is_open = False
            super().dispose()


# ============================================================================
# AudioConverter Framework
# ============================================================================


class AudioConverterError(CoreAudioError):
    """Exception for AudioConverter operations"""

    pass


class AudioConverter(capi.CoreAudioObject):
    """Audio format converter for sample rate and format conversion

    Provides high-level interface for converting between audio formats,
    sample rates, bit depths, and channel configurations.
    """

    def __init__(self, source_format: AudioFormat, dest_format: AudioFormat):
        """Create an AudioConverter

        Args:
            source_format: Source audio format
            dest_format: Destination audio format

        Raises:
            AudioConverterError: If converter creation fails
        """
        super().__init__()
        self._source_format = source_format
        self._dest_format = dest_format

        try:
            converter_id = capi.audio_converter_new(
                source_format.to_dict(), dest_format.to_dict()
            )
            self._set_object_id(converter_id)
        except Exception as e:
            raise AudioConverterError(f"Failed to create converter: {e}")

    @property
    def source_format(self) -> AudioFormat:
        """Get source audio format"""
        return self._source_format

    @property
    def dest_format(self) -> AudioFormat:
        """Get destination audio format"""
        return self._dest_format

    def convert(self, audio_data: bytes) -> bytes:
        """Convert audio data from source to destination format

        This method uses the simple buffer-based API (AudioConverterConvertBuffer)
        which only supports conversions where the input/output sizes are predictable.
        For complex conversions (sample rate, bit depth), use convert_with_callback().

        Args:
            audio_data: Input audio data in source format

        Returns:
            Converted audio data in destination format

        Raises:
            AudioConverterError: If conversion fails
        """
        self._ensure_not_disposed()
        try:
            return capi.audio_converter_convert_buffer(self.object_id, audio_data)
        except Exception as e:
            raise AudioConverterError(f"Failed to convert audio: {e}")

    def convert_with_callback(
        self,
        input_data: bytes,
        input_packet_count: int,
        output_packet_count: Optional[int] = None,
    ) -> bytes:
        """Convert audio using callback-based API for complex conversions

        This method supports all types of conversions including:
        - Sample rate changes (e.g., 44.1kHz -> 48kHz)
        - Bit depth changes (e.g., 16-bit -> 24-bit)
        - Channel count changes (stereo <-> mono)
        - Combinations of the above

        Args:
            input_data: Input audio data as bytes
            input_packet_count: Number of packets in input data
            output_packet_count: Expected output packets (auto-calculated if None)

        Returns:
            Converted audio data as bytes

        Raises:
            AudioConverterError: If conversion fails

        Example::

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
        """
        self._ensure_not_disposed()

        # Auto-calculate output packet count if not provided
        if output_packet_count is None:
            # Estimate based on sample rate ratio
            rate_ratio = self._dest_format.sample_rate / self._source_format.sample_rate
            output_packet_count = int(
                input_packet_count * rate_ratio * 1.1
            )  # 10% extra

        try:
            output_data, actual_packets = capi.audio_converter_fill_complex_buffer(
                self.object_id,
                input_data,
                input_packet_count,
                output_packet_count,
                self._source_format.to_dict(),
            )
            return output_data
        except Exception as e:
            raise AudioConverterError(f"Failed to convert audio: {e}")

    def get_property(self, property_id: int) -> bytes:
        """Get a property from the converter

        Args:
            property_id: Property ID

        Returns:
            Property data as bytes

        Raises:
            AudioConverterError: If getting property fails
        """
        self._ensure_not_disposed()
        try:
            return capi.audio_converter_get_property(self.object_id, property_id)
        except Exception as e:
            raise AudioConverterError(f"Failed to get property: {e}")

    def set_property(self, property_id: int, data: bytes) -> None:
        """Set a property on the converter

        Args:
            property_id: Property ID
            data: Property data as bytes

        Raises:
            AudioConverterError: If setting property fails
        """
        self._ensure_not_disposed()
        try:
            capi.audio_converter_set_property(self.object_id, property_id, data)
        except Exception as e:
            raise AudioConverterError(f"Failed to set property: {e}")

    def reset(self) -> None:
        """Reset the converter to its initial state"""
        self._ensure_not_disposed()
        try:
            capi.audio_converter_reset(self.object_id)
        except Exception as e:
            raise AudioConverterError(f"Failed to reset converter: {e}")

    def dispose(self) -> None:
        """Dispose of the audio converter"""
        if not self.is_disposed:
            try:
                capi.audio_converter_dispose(self.object_id)
            except:
                pass  # Best effort cleanup
            finally:
                super().dispose()

    def __enter__(self) -> "AudioConverter":
        """Enter context manager"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and dispose"""
        self.dispose()

    def __repr__(self) -> str:
        return f"AudioConverter({self._source_format} -> {self._dest_format})"


# ============================================================================
# ExtendedAudioFile Framework
# ============================================================================


class ExtendedAudioFile(capi.CoreAudioObject):
    """Extended audio file with automatic format conversion

    Provides high-level file I/O with automatic format conversion.
    Easier to use than AudioFile for common operations.
    """

    def __init__(self, path: Union[str, Path]):
        """Create an ExtendedAudioFile

        Args:
            path: Path to audio file

        Note:
            File is not opened automatically. Call open() or use as context manager.
        """
        super().__init__()
        self._path = str(path)
        self._is_open = False
        self._file_format: Optional[AudioFormat] = None
        self._client_format: Optional[AudioFormat] = None

    def open(self) -> "ExtendedAudioFile":
        """Open the audio file for reading

        Returns:
            Self for method chaining

        Raises:
            AudioFileError: If opening fails
        """
        self._ensure_not_disposed()
        if not self._is_open:
            try:
                file_id = capi.extended_audio_file_open_url(self._path)
                self._set_object_id(file_id)
                self._is_open = True
            except Exception as e:
                raise AudioFileError(f"Failed to open file {self._path}: {e}")
        return self

    @classmethod
    def create(
        cls, path: Union[str, Path], file_type: int, format: AudioFormat
    ) -> "ExtendedAudioFile":
        """Create a new audio file for writing

        Args:
            path: Path for new file
            file_type: Audio file type (e.g., kAudioFileWAVEType)
            format: Audio format for the file

        Returns:
            Opened ExtendedAudioFile instance

        Raises:
            AudioFileError: If creation fails
        """
        file = cls(path)
        try:
            file_id = capi.extended_audio_file_create_with_url(
                str(path), file_type, format.to_dict()
            )
            file._set_object_id(file_id)
            file._is_open = True
            file._file_format = format
            return file
        except Exception as e:
            raise AudioFileError(f"Failed to create file {path}: {e}")

    def close(self) -> None:
        """Close the audio file"""
        if self._is_open:
            try:
                capi.extended_audio_file_dispose(self.object_id)
            except Exception as e:
                raise AudioFileError(f"Failed to close file: {e}")
            finally:
                self._is_open = False
                self.dispose()

    def __enter__(self) -> "ExtendedAudioFile":
        if not self._is_open:
            self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @property
    def file_format(self) -> AudioFormat:
        """Get the file's native audio format

        Returns:
            File's audio format

        Raises:
            AudioFileError: If getting format fails
        """
        self._ensure_not_disposed()
        if not self._is_open:
            self.open()

        if self._file_format is None:
            try:
                format_data = capi.extended_audio_file_get_property(
                    self.object_id,
                    capi.get_extended_audio_file_property_file_data_format(),
                )
                # Parse AudioStreamBasicDescription (40 bytes)
                if len(format_data) >= 40:
                    asbd = struct.unpack("<dLLLLLLLL", format_data[:40])
                    (
                        sample_rate,
                        format_id_int,
                        format_flags,
                        bytes_per_packet,
                        frames_per_packet,
                        bytes_per_frame,
                        channels_per_frame,
                        bits_per_channel,
                        reserved,
                    ) = asbd

                    format_id = capi.int_to_fourchar(format_id_int)

                    self._file_format = AudioFormat(
                        sample_rate=sample_rate,
                        format_id=format_id,
                        format_flags=format_flags,
                        bytes_per_packet=bytes_per_packet,
                        frames_per_packet=frames_per_packet,
                        bytes_per_frame=bytes_per_frame,
                        channels_per_frame=channels_per_frame,
                        bits_per_channel=bits_per_channel,
                    )
                else:
                    raise AudioFileError(
                        f"Invalid format data size: {len(format_data)} bytes"
                    )
            except Exception as e:
                raise AudioFileError(f"Failed to get file format: {e}")

        return self._file_format

    @property
    def client_format(self) -> Optional[AudioFormat]:
        """Get the client audio format (for automatic conversion)

        Returns:
            Client audio format or None if not set
        """
        return self._client_format

    @client_format.setter
    def client_format(self, format: AudioFormat) -> None:
        """Set the client audio format for automatic conversion

        Args:
            format: Desired audio format for read/write operations

        Raises:
            AudioFileError: If setting format fails
        """
        self._ensure_not_disposed()
        if not self._is_open:
            self.open()

        try:
            format_bytes = struct.pack(
                "<dLLLLLLLL",
                format.sample_rate,
                capi.fourchar_to_int(format.format_id),
                format.format_flags,
                format.bytes_per_packet,
                format.frames_per_packet,
                format.bytes_per_frame,
                format.channels_per_frame,
                format.bits_per_channel,
                0,  # reserved
            )
            capi.extended_audio_file_set_property(
                self.object_id,
                capi.get_extended_audio_file_property_client_data_format(),
                format_bytes,
            )
            self._client_format = format
        except Exception as e:
            raise AudioFileError(f"Failed to set client format: {e}")

    def read(self, num_frames: int) -> Tuple[bytes, int]:
        """Read audio frames from the file

        Automatically converts to client format if set.

        Args:
            num_frames: Number of frames to read

        Returns:
            Tuple of (audio_data_bytes, frames_read)

        Raises:
            AudioFileError: If reading fails
        """
        self._ensure_not_disposed()
        if not self._is_open:
            self.open()

        try:
            return capi.extended_audio_file_read(self.object_id, num_frames)
        except Exception as e:
            raise AudioFileError(f"Failed to read frames: {e}")

    def write(self, num_frames: int, audio_data: bytes) -> None:
        """Write audio frames to the file

        Automatically converts from client format if set.

        Args:
            num_frames: Number of frames to write
            audio_data: Audio data bytes

        Raises:
            AudioFileError: If writing fails
        """
        self._ensure_not_disposed()
        if not self._is_open:
            raise AudioFileError("File not open")

        try:
            capi.extended_audio_file_write(self.object_id, num_frames, audio_data)
        except Exception as e:
            raise AudioFileError(f"Failed to write frames: {e}")

    def __repr__(self) -> str:
        status = "open" if self._is_open else "closed"
        return f"ExtendedAudioFile({self._path}, {status})"

    def dispose(self) -> None:
        """Dispose of the extended audio file"""
        if not self.is_disposed:
            if self._is_open:
                try:
                    capi.extended_audio_file_dispose(self.object_id)
                except:
                    pass  # Best effort cleanup
                finally:
                    self._is_open = False
            super().dispose()


# ============================================================================
# Audio Queue Framework
# ============================================================================


class AudioBuffer(capi.CoreAudioObject):
    """Audio buffer for queue operations"""

    def __init__(self, queue_id: int, buffer_size: int):
        super().__init__()
        self._queue_id = queue_id
        self._buffer_size = buffer_size

    @property
    def buffer_size(self) -> int:
        return self._buffer_size


class AudioQueue(capi.CoreAudioObject):
    """Audio queue for buffered playback and recording"""

    def __init__(self, audio_format: AudioFormat):
        super().__init__()
        self._format = audio_format
        self._buffers: List[AudioBuffer] = []

    @classmethod
    def new_output(cls, audio_format: AudioFormat) -> "AudioQueue":
        """Create a new output audio queue"""
        queue = cls(audio_format)
        try:
            queue_id = capi.audio_queue_new_output(audio_format.to_dict())
            queue._set_object_id(queue_id)
        except Exception as e:
            raise AudioQueueError(f"Failed to create output queue: {e}")
        return queue

    def allocate_buffer(self, buffer_size: int) -> AudioBuffer:
        """Allocate an audio buffer"""
        self._ensure_not_disposed()
        try:
            buffer_id = capi.audio_queue_allocate_buffer(self.object_id, buffer_size)
            buffer = AudioBuffer(self.object_id, buffer_size)
            buffer._set_object_id(buffer_id)
            self._buffers.append(buffer)
            return buffer
        except Exception as e:
            raise AudioQueueError(f"Failed to allocate buffer: {e}")

    def enqueue_buffer(self, buffer: AudioBuffer) -> None:
        """Enqueue an audio buffer"""
        self._ensure_not_disposed()
        try:
            capi.audio_queue_enqueue_buffer(self.object_id, buffer.object_id)
        except Exception as e:
            raise AudioQueueError(f"Failed to enqueue buffer: {e}")

    def start(self) -> None:
        """Start the audio queue"""
        self._ensure_not_disposed()
        try:
            capi.audio_queue_start(self.object_id)
        except Exception as e:
            raise AudioQueueError(f"Failed to start queue: {e}")

    def stop(self, immediate: bool = True) -> None:
        """Stop the audio queue"""
        self._ensure_not_disposed()
        try:
            capi.audio_queue_stop(self.object_id, immediate)
        except Exception as e:
            raise AudioQueueError(f"Failed to stop queue: {e}")

    def dispose(self, immediate: bool = True) -> None:
        """Dispose of the audio queue"""
        if not self.is_disposed:
            try:
                capi.audio_queue_dispose(self.object_id, immediate)
            except Exception as e:
                raise AudioQueueError(f"Failed to dispose queue: {e}")
            finally:
                # Clear buffer references and call base dispose
                self._buffers.clear()
                super().dispose()


# ============================================================================
# Audio Component & AudioUnit Framework
# ============================================================================


class AudioComponentDescription:
    """Pythonic representation of AudioComponent description"""

    def __init__(
        self,
        type: str,
        subtype: str,
        manufacturer: str,
        flags: int = 0,
        flags_mask: int = 0,
    ):
        self.type = type
        self.subtype = subtype
        self.manufacturer = manufacturer
        self.flags = flags
        self.flags_mask = flags_mask

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for functional API"""
        # Convert fourcc strings to integers
        from . import capi

        type_int = (
            capi.fourchar_to_int(self.type) if isinstance(self.type, str) else self.type
        )
        subtype_int = (
            capi.fourchar_to_int(self.subtype)
            if isinstance(self.subtype, str)
            else self.subtype
        )
        manufacturer_int = (
            capi.fourchar_to_int(self.manufacturer)
            if isinstance(self.manufacturer, str)
            else self.manufacturer
        )

        return {
            "type": type_int,
            "subtype": subtype_int,
            "manufacturer": manufacturer_int,
            "flags": self.flags,
            "flags_mask": self.flags_mask,
        }


class AudioComponent(capi.CoreAudioObject):
    """Audio component wrapper"""

    def __init__(self, description: AudioComponentDescription):
        super().__init__()
        self._description = description

    @classmethod
    def find_next(
        cls, description: AudioComponentDescription
    ) -> Optional["AudioComponent"]:
        """Find the next matching audio component"""
        try:
            result = capi.audio_component_find_next(description.to_dict())
            if result is None or result == 0:
                return None
            component = cls(description)
            # Set the object_id using the Cython method
            component._set_object_id(result)
            return component
        except Exception:
            # If lookup fails, component doesn't exist
            return None

    def create_instance(self) -> "AudioUnit":
        """Create an AudioUnit instance from this component"""
        self._ensure_not_disposed()
        try:
            unit_id = capi.audio_component_instance_new(self.object_id)
            unit = AudioUnit(self._description)
            unit._set_object_id(unit_id)
            return unit
        except Exception as e:
            raise AudioUnitError(f"Failed to create instance: {e}")


class AudioUnit(capi.CoreAudioObject):
    """Audio unit for real-time audio processing"""

    def __init__(self, description: AudioComponentDescription):
        super().__init__()
        self._description = description
        self._is_initialized = False

    @classmethod
    def default_output(cls) -> "AudioUnit":
        """Create a default output AudioUnit"""
        desc = AudioComponentDescription(
            type="auou",  # kAudioUnitType_Output
            subtype="def ",  # kAudioUnitSubType_DefaultOutput
            manufacturer="appl",  # kAudioUnitManufacturer_Apple
        )
        component = AudioComponent.find_next(desc)
        if component is None:
            raise AudioUnitError("Default output AudioUnit not found")
        return component.create_instance()

    def initialize(self) -> None:
        """Initialize the AudioUnit"""
        self._ensure_not_disposed()
        if not self._is_initialized:
            try:
                capi.audio_unit_initialize(self.object_id)
                self._is_initialized = True
            except Exception as e:
                raise AudioUnitError(f"Failed to initialize: {e}")

    def uninitialize(self) -> None:
        """Uninitialize the AudioUnit"""
        if self._is_initialized:
            try:
                capi.audio_unit_uninitialize(self.object_id)
            except Exception as e:
                raise AudioUnitError(f"Failed to uninitialize: {e}")
            finally:
                self._is_initialized = False

    def start(self) -> None:
        """Start the AudioUnit output"""
        self._ensure_not_disposed()
        if not self._is_initialized:
            raise AudioUnitError("AudioUnit not initialized")
        try:
            capi.audio_output_unit_start(self.object_id)
        except Exception as e:
            raise AudioUnitError(f"Failed to start: {e}")

    def stop(self) -> None:
        """Stop the AudioUnit output"""
        self._ensure_not_disposed()
        try:
            capi.audio_output_unit_stop(self.object_id)
        except Exception as e:
            raise AudioUnitError(f"Failed to stop: {e}")

    def get_property(self, property_id: int, scope: int, element: int) -> bytes:
        """Get a property from the AudioUnit"""
        self._ensure_not_disposed()
        try:
            return capi.audio_unit_get_property(
                self.object_id, property_id, scope, element
            )
        except Exception as e:
            raise AudioUnitError(f"Failed to get property: {e}")

    def set_property(
        self, property_id: int, scope: int, element: int, data: bytes
    ) -> None:
        """Set a property on the AudioUnit"""
        self._ensure_not_disposed()
        try:
            capi.audio_unit_set_property(
                self.object_id, property_id, scope, element, data
            )
        except Exception as e:
            raise AudioUnitError(f"Failed to set property: {e}")

    # ========================================================================
    # Advanced AudioUnit Features
    # ========================================================================

    def get_stream_format(self, scope: str = "output", element: int = 0) -> AudioFormat:
        """Get the stream format for a specific scope and element

        Args:
            scope: 'input', 'output', or 'global' (default: 'output')
            element: Element index (default: 0)

        Returns:
            AudioFormat object with the current stream format
        """
        self._ensure_not_disposed()

        # Map scope name to constant
        scope_map = {
            "input": capi.get_audio_unit_scope_input(),
            "output": capi.get_audio_unit_scope_output(),
            "global": capi.get_audio_unit_scope_global(),
        }
        scope_val = scope_map.get(scope.lower())
        if scope_val is None:
            raise AudioUnitError(f"Invalid scope: {scope}")

        try:
            import struct

            asbd_data = self.get_property(
                capi.get_audio_unit_property_stream_format(), scope_val, element
            )

            if len(asbd_data) >= 40:
                asbd = struct.unpack("<dLLLLLLLL", asbd_data[:40])
                (
                    sample_rate,
                    format_id_int,
                    format_flags,
                    bytes_per_packet,
                    frames_per_packet,
                    bytes_per_frame,
                    channels_per_frame,
                    bits_per_channel,
                    reserved,
                ) = asbd

                format_id = capi.int_to_fourchar(format_id_int)

                return AudioFormat(
                    sample_rate=sample_rate,
                    format_id=format_id,
                    format_flags=format_flags,
                    bytes_per_packet=bytes_per_packet,
                    frames_per_packet=frames_per_packet,
                    bytes_per_frame=bytes_per_frame,
                    channels_per_frame=channels_per_frame,
                    bits_per_channel=bits_per_channel,
                )
            else:
                raise AudioUnitError(f"Invalid ASBD data size: {len(asbd_data)}")
        except Exception as e:
            raise AudioUnitError(f"Failed to get stream format: {e}")

    def set_stream_format(
        self, format: AudioFormat, scope: str = "output", element: int = 0
    ) -> None:
        """Set the stream format for a specific scope and element

        Args:
            format: AudioFormat object with desired format
            scope: 'input', 'output', or 'global' (default: 'output')
            element: Element index (default: 0)
        """
        self._ensure_not_disposed()

        # Map scope name to constant
        scope_map = {
            "input": capi.get_audio_unit_scope_input(),
            "output": capi.get_audio_unit_scope_output(),
            "global": capi.get_audio_unit_scope_global(),
        }
        scope_val = scope_map.get(scope.lower())
        if scope_val is None:
            raise AudioUnitError(f"Invalid scope: {scope}")

        try:
            import struct

            # Convert format_id to integer
            format_id_int = (
                capi.fourchar_to_int(format.format_id)
                if isinstance(format.format_id, str)
                else format.format_id
            )

            # Pack AudioStreamBasicDescription
            asbd_data = struct.pack(
                "<dLLLLLLLL",
                format.sample_rate,
                format_id_int,
                format.format_flags,
                format.bytes_per_packet,
                format.frames_per_packet,
                format.bytes_per_frame,
                format.channels_per_frame,
                format.bits_per_channel,
                0,  # reserved
            )

            self.set_property(
                capi.get_audio_unit_property_stream_format(),
                scope_val,
                element,
                asbd_data,
            )
        except Exception as e:
            raise AudioUnitError(f"Failed to set stream format: {e}")

    @property
    def sample_rate(self) -> float:
        """Get the sample rate (kAudioUnitProperty_SampleRate on global scope)"""
        self._ensure_not_disposed()
        try:
            import struct

            data = self.get_property(
                2, capi.get_audio_unit_scope_global(), 0
            )  # kAudioUnitProperty_SampleRate = 2
            if len(data) >= 8:
                return struct.unpack("<d", data[:8])[0]
            return 0.0
        except Exception:
            # Fallback to stream format sample rate
            try:
                return self.get_stream_format("output", 0).sample_rate
            except Exception:
                return 0.0

    @sample_rate.setter
    def sample_rate(self, rate: float) -> None:
        """Set the sample rate (kAudioUnitProperty_SampleRate on global scope)"""
        self._ensure_not_disposed()
        try:
            import struct

            data = struct.pack("<d", rate)
            self.set_property(
                2, capi.get_audio_unit_scope_global(), 0, data
            )  # kAudioUnitProperty_SampleRate = 2
        except Exception as e:
            raise AudioUnitError(f"Failed to set sample rate: {e}")

    @property
    def latency(self) -> float:
        """Get the latency in seconds (kAudioUnitProperty_Latency)"""
        self._ensure_not_disposed()
        try:
            import struct

            data = self.get_property(
                12, capi.get_audio_unit_scope_global(), 0
            )  # kAudioUnitProperty_Latency = 12
            if len(data) >= 8:
                return struct.unpack("<d", data[:8])[0]
            return 0.0
        except Exception:
            return 0.0

    @property
    def cpu_load(self) -> float:
        """Get the CPU load as a fraction (0.0 to 1.0) (kAudioUnitProperty_CPULoad)"""
        self._ensure_not_disposed()
        try:
            import struct

            data = self.get_property(
                6, capi.get_audio_unit_scope_global(), 0
            )  # kAudioUnitProperty_CPULoad = 6
            if len(data) >= 4:
                return struct.unpack("<f", data[:4])[0]
            return 0.0
        except Exception:
            return 0.0

    @property
    def max_frames_per_slice(self) -> int:
        """Get the maximum frames per slice (kAudioUnitProperty_MaximumFramesPerSlice)"""
        self._ensure_not_disposed()
        try:
            import struct

            data = self.get_property(
                14, capi.get_audio_unit_scope_global(), 0
            )  # kAudioUnitProperty_MaximumFramesPerSlice = 14
            if len(data) >= 4:
                return struct.unpack("<L", data[:4])[0]
            return 0
        except Exception:
            return 0

    @max_frames_per_slice.setter
    def max_frames_per_slice(self, frames: int) -> None:
        """Set the maximum frames per slice (kAudioUnitProperty_MaximumFramesPerSlice)"""
        self._ensure_not_disposed()
        try:
            import struct

            data = struct.pack("<L", frames)
            self.set_property(
                14, capi.get_audio_unit_scope_global(), 0, data
            )  # kAudioUnitProperty_MaximumFramesPerSlice = 14
        except Exception as e:
            raise AudioUnitError(f"Failed to set max frames per slice: {e}")

    def get_parameter_list(self, scope: str = "global") -> List[int]:
        """Get list of available parameter IDs (kAudioUnitProperty_ParameterList)

        Args:
            scope: 'input', 'output', or 'global' (default: 'global')

        Returns:
            List of parameter IDs
        """
        self._ensure_not_disposed()

        scope_map = {
            "input": capi.get_audio_unit_scope_input(),
            "output": capi.get_audio_unit_scope_output(),
            "global": capi.get_audio_unit_scope_global(),
        }
        scope_val = scope_map.get(scope.lower())
        if scope_val is None:
            raise AudioUnitError(f"Invalid scope: {scope}")

        try:
            import struct

            data = self.get_property(
                3, scope_val, 0
            )  # kAudioUnitProperty_ParameterList = 3
            # Data is an array of UInt32 parameter IDs
            param_count = len(data) // 4
            if param_count > 0:
                return list(struct.unpack(f"<{param_count}L", data[: param_count * 4]))
            return []
        except Exception:
            return []

    def render(self, num_frames: int, timestamp: Optional[int] = None) -> bytes:
        """Render audio frames (for offline processing)

        Args:
            num_frames: Number of frames to render
            timestamp: Optional timestamp (default: None uses current time)

        Returns:
            Rendered audio data as bytes

        Note: This is a simplified render method for offline processing.
        For real-time audio, use render callbacks with the audio player infrastructure.
        """
        # This would require implementing AudioUnitRender which needs more infrastructure
        raise NotImplementedError(
            "Direct rendering not yet implemented. "
            "Use the audio player infrastructure with render callbacks for real-time audio."
        )

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.uninitialize()
        self.dispose()

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def dispose(self) -> None:
        """Dispose of the AudioUnit"""
        if not self.is_disposed:
            if self._is_initialized:
                try:
                    capi.audio_unit_uninitialize(self.object_id)
                except:
                    pass  # Best effort cleanup
                finally:
                    self._is_initialized = False

            if self.object_id != 0:
                try:
                    capi.audio_component_instance_dispose(self.object_id)
                except:
                    pass  # Best effort cleanup

            super().dispose()


# ============================================================================
# MIDI Framework
# ============================================================================


class MIDIPort(capi.CoreAudioObject):
    """Base class for MIDI ports"""

    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self._client: Optional["MIDIClient"] = None  # Reference to parent MIDIClient

    @property
    def name(self) -> str:
        return self._name

    def dispose(self) -> None:
        """Dispose of the MIDI port"""
        if not self.is_disposed:
            try:
                capi.midi_port_dispose(self.object_id)
            except Exception:
                # Best effort disposal - some MIDI operations may fail in test environments
                pass
            finally:
                # Remove from client's port list if we have a client reference
                if self._client and hasattr(self._client, "_ports"):
                    try:
                        self._client._ports.remove(self)
                    except ValueError:
                        pass  # Already removed
                super().dispose()


class MIDIInputPort(MIDIPort):
    """MIDI input port for receiving MIDI data"""

    def connect_source(self, source) -> None:
        """Connect to a MIDI source"""
        self._ensure_not_disposed()
        try:
            capi.midi_port_connect_source(self.object_id, source.object_id)
        except Exception as e:
            raise MIDIError(f"Failed to connect source: {e}")

    def disconnect_source(self, source) -> None:
        """Disconnect from a MIDI source"""
        self._ensure_not_disposed()
        try:
            capi.midi_port_disconnect_source(self.object_id, source.object_id)
        except Exception as e:
            raise MIDIError(f"Failed to disconnect source: {e}")


class MIDIOutputPort(MIDIPort):
    """MIDI output port for sending MIDI data"""

    def send_data(self, destination, data: bytes, timestamp: int = 0) -> None:
        """Send MIDI data to a destination"""
        self._ensure_not_disposed()
        try:
            capi.midi_send(self.object_id, destination.object_id, data, timestamp)
        except Exception as e:
            raise MIDIError(f"Failed to send data: {e}")


class MIDIClient(capi.CoreAudioObject):
    """MIDI client for managing MIDI operations"""

    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self._ports: List[MIDIPort] = []
        try:
            client_id = capi.midi_client_create(name)
            self._set_object_id(client_id)
        except Exception as e:
            raise MIDIError(f"Failed to create MIDI client: {e}")

    @property
    def name(self) -> str:
        return self._name

    def create_input_port(self, name: str) -> MIDIInputPort:
        """Create a MIDI input port"""
        self._ensure_not_disposed()
        try:
            port_id = capi.midi_input_port_create(self.object_id, name)
            port = MIDIInputPort(name)
            port._set_object_id(port_id)
            port._client = self
            self._ports.append(port)
            return port
        except Exception as e:
            raise MIDIError(f"Failed to create input port: {e}")

    def create_output_port(self, name: str) -> MIDIOutputPort:
        """Create a MIDI output port"""
        self._ensure_not_disposed()
        try:
            port_id = capi.midi_output_port_create(self.object_id, name)
            port = MIDIOutputPort(name)
            port._set_object_id(port_id)
            port._client = self
            self._ports.append(port)
            return port
        except Exception as e:
            raise MIDIError(f"Failed to create output port: {e}")

    def dispose(self) -> None:
        """Dispose of the MIDI client and all its ports"""
        if not self.is_disposed:
            # Dispose all ports first
            for port in self._ports[
                :
            ]:  # Copy list to avoid modification during iteration
                if not port.is_disposed:
                    try:
                        port.dispose()
                    except:
                        pass  # Best effort cleanup

            try:
                capi.midi_client_dispose(self.object_id)
            except Exception:
                # Best effort disposal - some MIDI operations may fail in test environments
                pass
            finally:
                # Clear port references and call base dispose
                self._ports.clear()
                super().dispose()


# ============================================================================
# Audio Device & Hardware Abstraction
# ============================================================================


class AudioDevice(capi.CoreAudioObject):
    """Represents a hardware audio device with property access

    Provides Pythonic access to audio hardware devices including inputs,
    outputs, and their properties like name, sample rate, channels, etc.
    """

    def __init__(self, device_id: int):
        """Initialize AudioDevice with a device ID

        Args:
            device_id: The AudioObjectID for this device
        """
        super().__init__()
        self._set_object_id(device_id)

    def _get_property_string(
        self, property_id: int, scope: Optional[int] = None, element: int = 0
    ) -> str:
        """Get a string property from the device"""
        if scope is None:
            scope = capi.get_audio_object_property_scope_global()

        try:
            data = capi.audio_object_get_property_string(
                self.object_id, property_id, scope, element
            )
            if data:
                # Decode UTF-8 string from CoreFoundation
                # Remove any null terminators
                return data.decode("utf-8", errors="ignore").strip("\x00")
            return ""
        except Exception:
            return ""

    def _get_property_uint32(
        self, property_id: int, scope: Optional[int] = None, element: int = 0
    ) -> int:
        """Get a UInt32 property from the device"""
        if scope is None:
            scope = capi.get_audio_object_property_scope_global()

        try:
            data = capi.audio_object_get_property_data(
                self.object_id, property_id, scope, element
            )
            if len(data) >= 4:
                return struct.unpack("<L", data[:4])[0]
            return 0
        except Exception:
            return 0

    def _get_property_float64(
        self, property_id: int, scope: Optional[int] = None, element: int = 0
    ) -> float:
        """Get a Float64 property from the device"""
        if scope is None:
            scope = capi.get_audio_object_property_scope_global()

        try:
            data = capi.audio_object_get_property_data(
                self.object_id, property_id, scope, element
            )
            if len(data) >= 8:
                return struct.unpack("<d", data[:8])[0]
            return 0.0
        except Exception:
            return 0.0

    @property
    def name(self) -> str:
        """Get the device name"""
        return self._get_property_string(capi.get_audio_object_property_name())

    @property
    def manufacturer(self) -> str:
        """Get the device manufacturer"""
        return self._get_property_string(capi.get_audio_object_property_manufacturer())

    @property
    def uid(self) -> str:
        """Get the device UID (unique identifier)"""
        return self._get_property_string(capi.get_audio_device_property_device_uid())

    @property
    def model_uid(self) -> str:
        """Get the device model UID"""
        return self._get_property_string(capi.get_audio_device_property_model_uid())

    @property
    def transport_type(self) -> int:
        """Get the transport type (USB, PCI, etc.)"""
        return self._get_property_uint32(
            capi.get_audio_device_property_transport_type()
        )

    @property
    def sample_rate(self) -> float:
        """Get the current nominal sample rate"""
        return self._get_property_float64(
            capi.get_audio_device_property_nominal_sample_rate()
        )

    @sample_rate.setter
    def sample_rate(self, rate: float) -> None:
        """Set the nominal sample rate (not all devices support this)"""
        # This would require implementing AudioObjectSetPropertyData
        raise NotImplementedError("Setting sample rate not yet implemented")

    @property
    def is_alive(self) -> bool:
        """Check if the device is alive/connected"""
        value = self._get_property_uint32(
            capi.get_audio_device_property_device_is_alive()
        )
        return bool(value)

    @property
    def is_hidden(self) -> bool:
        """Check if the device is hidden"""
        value = self._get_property_uint32(capi.get_audio_device_property_is_hidden())
        return bool(value)

    def get_stream_configuration(self, scope: str = "output") -> Dict[str, Any]:
        """Get stream configuration (channel layout)

        Args:
            scope: 'input' or 'output' (default: 'output')

        Returns:
            Dictionary with stream configuration information
        """
        scope_map = {
            "input": capi.get_audio_object_property_scope_input(),
            "output": capi.get_audio_object_property_scope_output(),
        }
        scope_val = scope_map.get(scope.lower())
        if scope_val is None:
            raise AudioDeviceError(f"Invalid scope: {scope}")

        try:
            data = capi.audio_object_get_property_data(
                self.object_id,
                capi.get_audio_device_property_stream_configuration(),
                scope_val,
                0,
            )
            # AudioBufferList structure - would need detailed parsing
            # For now, return basic info
            return {"raw_data_length": len(data)}
        except Exception as e:
            raise AudioDeviceError(f"Failed to get stream configuration: {e}")

    def __repr__(self) -> str:
        name = self.name or "Unknown"
        return f"AudioDevice(id={self.object_id}, name='{name}')"

    def __str__(self) -> str:
        return f"{self.name} ({self.manufacturer})"


class AudioDeviceManager:
    """Manager for discovering and accessing audio devices

    Provides static methods for device discovery and retrieval.
    """

    @staticmethod
    def get_devices() -> List[AudioDevice]:
        """Get all available audio devices

        Returns:
            List of AudioDevice objects
        """
        device_ids = capi.audio_hardware_get_devices()
        return [AudioDevice(device_id) for device_id in device_ids]

    @staticmethod
    def get_default_output_device() -> Optional[AudioDevice]:
        """Get the default output device

        Returns:
            AudioDevice object or None if no default
        """
        device_id = capi.audio_hardware_get_default_output_device()
        if device_id == 0:
            return None
        return AudioDevice(device_id)

    @staticmethod
    def get_default_input_device() -> Optional[AudioDevice]:
        """Get the default input device

        Returns:
            AudioDevice object or None if no default
        """
        device_id = capi.audio_hardware_get_default_input_device()
        if device_id == 0:
            return None
        return AudioDevice(device_id)

    @staticmethod
    def get_output_devices() -> List[AudioDevice]:
        """Get all output devices

        Returns:
            List of AudioDevice objects that have output capability
        """
        # For now, return all devices - would need to filter by checking
        # stream configuration for output scope
        return AudioDeviceManager.get_devices()

    @staticmethod
    def get_input_devices() -> List[AudioDevice]:
        """Get all input devices

        Returns:
            List of AudioDevice objects that have input capability
        """
        # For now, return all devices - would need to filter by checking
        # stream configuration for input scope
        return AudioDeviceManager.get_devices()

    @staticmethod
    def find_device_by_name(name: str) -> Optional[AudioDevice]:
        """Find a device by name

        Args:
            name: Device name to search for (case-insensitive)

        Returns:
            AudioDevice object or None if not found
        """
        for device in AudioDeviceManager.get_devices():
            if device.name.lower() == name.lower():
                return device
        return None

    @staticmethod
    def find_device_by_uid(uid: str) -> Optional[AudioDevice]:
        """Find a device by UID

        Args:
            uid: Device UID to search for

        Returns:
            AudioDevice object or None if not found
        """
        for device in AudioDeviceManager.get_devices():
            try:
                device_uid = device.uid
                if device_uid and device_uid == uid:
                    return device
            except Exception:
                # Some devices may not have UID property accessible
                continue
        return None


# ============================================================================
# AUGraph Framework
# ============================================================================


class AUGraphError(CoreAudioError):
    """Exception for AUGraph operations"""

    pass


class AUGraph(capi.CoreAudioObject):
    """Audio Unit Graph for managing and connecting multiple AudioUnits

    AUGraph provides a high-level API for creating and managing graphs of
    AudioUnits, including connections between nodes and overall graph lifecycle.

    Note: AUGraph is deprecated by Apple in favor of AVAudioEngine, but remains
    fully functional and useful for advanced audio processing scenarios.
    """

    def __init__(self):
        """Create a new AUGraph

        Raises:
            AUGraphError: If graph creation fails
        """
        super().__init__()
        try:
            graph_id = capi.au_graph_new()
            self._set_object_id(graph_id)
        except Exception as e:
            raise AUGraphError(f"Failed to create AUGraph: {e}")

    def open(self) -> "AUGraph":
        """Open the graph (opens AudioUnits but doesn't initialize them)

        Returns:
            Self for method chaining

        Raises:
            AUGraphError: If open fails
        """
        self._ensure_not_disposed()
        try:
            capi.au_graph_open(self.object_id)
            return self
        except Exception as e:
            raise AUGraphError(f"Failed to open graph: {e}")

    def close(self) -> None:
        """Close the graph (closes all AudioUnits)

        Raises:
            AUGraphError: If close fails
        """
        self._ensure_not_disposed()
        try:
            capi.au_graph_close(self.object_id)
        except Exception as e:
            raise AUGraphError(f"Failed to close graph: {e}")

    def initialize(self) -> "AUGraph":
        """Initialize the graph (prepares all AudioUnits for rendering)

        Returns:
            Self for method chaining

        Raises:
            AUGraphError: If initialization fails
        """
        self._ensure_not_disposed()
        try:
            capi.au_graph_initialize(self.object_id)
            return self
        except Exception as e:
            raise AUGraphError(f"Failed to initialize graph: {e}")

    def uninitialize(self) -> None:
        """Uninitialize the graph

        Raises:
            AUGraphError: If uninitialization fails
        """
        self._ensure_not_disposed()
        try:
            capi.au_graph_uninitialize(self.object_id)
        except Exception as e:
            raise AUGraphError(f"Failed to uninitialize graph: {e}")

    def start(self) -> None:
        """Start the graph (begins audio rendering)

        Raises:
            AUGraphError: If start fails
        """
        self._ensure_not_disposed()
        try:
            capi.au_graph_start(self.object_id)
        except Exception as e:
            raise AUGraphError(f"Failed to start graph: {e}")

    def stop(self) -> None:
        """Stop the graph (stops audio rendering)

        Raises:
            AUGraphError: If stop fails
        """
        self._ensure_not_disposed()
        try:
            capi.au_graph_stop(self.object_id)
        except Exception as e:
            raise AUGraphError(f"Failed to stop graph: {e}")

    @property
    def is_open(self) -> bool:
        """Check if the graph is open"""
        self._ensure_not_disposed()
        return capi.au_graph_is_open(self.object_id)

    @property
    def is_initialized(self) -> bool:
        """Check if the graph is initialized"""
        self._ensure_not_disposed()
        return capi.au_graph_is_initialized(self.object_id)

    @property
    def is_running(self) -> bool:
        """Check if the graph is running"""
        self._ensure_not_disposed()
        return capi.au_graph_is_running(self.object_id)

    def add_node(self, description: AudioComponentDescription) -> int:
        """Add a node to the graph

        Args:
            description: AudioComponentDescription for the node

        Returns:
            Node ID

        Raises:
            AUGraphError: If adding node fails
        """
        self._ensure_not_disposed()
        try:
            desc_dict = {
                "type": capi.fourchar_to_int(description.type)
                if isinstance(description.type, str)
                else description.type,
                "subtype": capi.fourchar_to_int(description.subtype)
                if isinstance(description.subtype, str)
                else description.subtype,
                "manufacturer": capi.fourchar_to_int(description.manufacturer)
                if isinstance(description.manufacturer, str)
                else description.manufacturer,
                "flags": description.flags,
                "flags_mask": description.flags_mask,
            }
            return capi.au_graph_add_node(self.object_id, desc_dict)
        except Exception as e:
            raise AUGraphError(f"Failed to add node: {e}")

    def remove_node(self, node_id: int) -> None:
        """Remove a node from the graph

        Args:
            node_id: Node ID to remove

        Raises:
            AUGraphError: If removing node fails
        """
        self._ensure_not_disposed()
        try:
            capi.au_graph_remove_node(self.object_id, node_id)
        except Exception as e:
            raise AUGraphError(f"Failed to remove node: {e}")

    @property
    def node_count(self) -> int:
        """Get the number of nodes in the graph"""
        self._ensure_not_disposed()
        return capi.au_graph_get_node_count(self.object_id)

    def get_node_at_index(self, index: int) -> int:
        """Get node ID at the specified index

        Args:
            index: Node index

        Returns:
            Node ID

        Raises:
            AUGraphError: If index is invalid
        """
        self._ensure_not_disposed()
        try:
            return capi.au_graph_get_ind_node(self.object_id, index)
        except Exception as e:
            raise AUGraphError(f"Failed to get node at index {index}: {e}")

    def get_node_info(self, node_id: int) -> Tuple[AudioComponentDescription, int]:
        """Get information about a node

        Args:
            node_id: Node ID

        Returns:
            Tuple of (AudioComponentDescription, AudioUnit ID)

        Raises:
            AUGraphError: If getting node info fails
        """
        self._ensure_not_disposed()
        try:
            desc_dict, audio_unit_id = capi.au_graph_node_info(self.object_id, node_id)

            # Convert back to AudioComponentDescription
            desc = AudioComponentDescription(
                type=capi.int_to_fourchar(desc_dict["type"]),
                subtype=capi.int_to_fourchar(desc_dict["subtype"]),
                manufacturer=capi.int_to_fourchar(desc_dict["manufacturer"]),
                flags=desc_dict["flags"],
                flags_mask=desc_dict["flags_mask"],
            )

            return (desc, audio_unit_id)
        except Exception as e:
            raise AUGraphError(f"Failed to get node info: {e}")

    def connect(
        self, source_node: int, source_output: int, dest_node: int, dest_input: int
    ) -> None:
        """Connect two nodes in the graph

        Args:
            source_node: Source node ID
            source_output: Source output bus number
            dest_node: Destination node ID
            dest_input: Destination input bus number

        Raises:
            AUGraphError: If connection fails
        """
        self._ensure_not_disposed()
        try:
            capi.au_graph_connect_node_input(
                self.object_id, source_node, source_output, dest_node, dest_input
            )
        except Exception as e:
            raise AUGraphError(f"Failed to connect nodes: {e}")

    def disconnect(self, dest_node: int, dest_input: int) -> None:
        """Disconnect a node's input

        Args:
            dest_node: Destination node ID
            dest_input: Destination input bus number

        Raises:
            AUGraphError: If disconnection fails
        """
        self._ensure_not_disposed()
        try:
            capi.au_graph_disconnect_node_input(self.object_id, dest_node, dest_input)
        except Exception as e:
            raise AUGraphError(f"Failed to disconnect node: {e}")

    def clear_connections(self) -> None:
        """Clear all connections in the graph

        Raises:
            AUGraphError: If clearing connections fails
        """
        self._ensure_not_disposed()
        try:
            capi.au_graph_clear_connections(self.object_id)
        except Exception as e:
            raise AUGraphError(f"Failed to clear connections: {e}")

    def update(self) -> bool:
        """Update the graph after making changes

        Returns:
            True if update completed immediately, False if pending

        Raises:
            AUGraphError: If update fails
        """
        self._ensure_not_disposed()
        try:
            return capi.au_graph_update(self.object_id)
        except Exception as e:
            raise AUGraphError(f"Failed to update graph: {e}")

    @property
    def cpu_load(self) -> float:
        """Get current CPU load (0.0-1.0)"""
        self._ensure_not_disposed()
        return capi.au_graph_get_cpu_load(self.object_id)

    @property
    def max_cpu_load(self) -> float:
        """Get maximum CPU load since last query (0.0-1.0)"""
        self._ensure_not_disposed()
        return capi.au_graph_get_max_cpu_load(self.object_id)

    def dispose(self) -> None:
        """Dispose of the graph"""
        if not self.is_disposed:
            try:
                capi.au_graph_dispose(self.object_id)
            except:
                pass  # Best effort cleanup
            finally:
                super().dispose()

    def __enter__(self) -> "AUGraph":
        """Enter context manager"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and dispose"""
        self.dispose()

    def __repr__(self) -> str:
        status = []
        if not self.is_disposed:
            try:
                if self.is_open:
                    status.append("open")
                if self.is_initialized:
                    status.append("initialized")
                if self.is_running:
                    status.append("running")
            except:
                pass
        else:
            status.append("disposed")

        status_str = ", ".join(status) if status else "closed"
        return f"AUGraph({status_str}, nodes={self.node_count if not self.is_disposed else 0})"


# ============================================================================
# CoreAudioClock - Audio/MIDI Synchronization and Timing
# ============================================================================

class ClockTimeFormat:
    """Time format constants for CoreAudioClock"""
    HOST_TIME = capi.get_ca_clock_time_format_host_time()
    SAMPLES = capi.get_ca_clock_time_format_samples()
    BEATS = capi.get_ca_clock_time_format_beats()
    SECONDS = capi.get_ca_clock_time_format_seconds()
    SMPTE_TIME = capi.get_ca_clock_time_format_smpte_time()


class AudioClock(capi.CoreAudioObject):
    """High-level CoreAudioClock for audio/MIDI synchronization and timing

    AudioClock provides synchronization services for audio and MIDI applications,
    supporting multiple time formats and playback control.

    Supported time formats:

    - Host time (mach_absolute_time)
    - Audio samples
    - Musical beats
    - Seconds
    - SMPTE timecode

    Example::

        # Create and control a clock
        with AudioClock() as clock:
            clock.play_rate = 1.0
            clock.start()

            # Get current time in different formats
            seconds = clock.get_time_seconds()
            beats = clock.get_time_beats()

            print(f"Position: {seconds:.2f}s ({beats:.2f} beats)")

            clock.stop()

    Example with tempo and speed control::

        clock = AudioClock()
        clock.play_rate = 0.5  # Half speed
        clock.start()
        # ... use clock for synchronization
        clock.stop()
        clock.dispose()
    """
    
    def __init__(self):
        """Initialize a new CoreAudioClock"""
        super().__init__()
        self._is_created = False
        self._is_running = False
    
    def create(self) -> "AudioClock":
        """Create the underlying clock object
        
        Returns:
            Self for method chaining
            
        Raises:
            RuntimeError: If clock creation fails
        """
        if not self._is_created:
            try:
                clock_id = capi.ca_clock_new()
                self._set_object_id(clock_id)
                self._is_created = True
            except Exception as e:
                raise RuntimeError(f"Failed to create clock: {e}")
        return self
    
    def start(self) -> None:
        """Start the clock advancing on its timeline
        
        Raises:
            RuntimeError: If clock is not created or start fails
        """
        self._ensure_not_disposed()
        if not self._is_created:
            self.create()
        
        try:
            capi.ca_clock_start(self.object_id)
            self._is_running = True
        except Exception as e:
            raise RuntimeError(f"Failed to start clock: {e}")
    
    def stop(self) -> None:
        """Stop the clock
        
        Raises:
            RuntimeError: If stop fails
        """
        if self._is_running:
            try:
                capi.ca_clock_stop(self.object_id)
                self._is_running = False
            except Exception as e:
                raise RuntimeError(f"Failed to stop clock: {e}")
    
    @property
    def is_running(self) -> bool:
        """Check if the clock is currently running"""
        return self._is_running
    
    @property
    def play_rate(self) -> float:
        """Get or set the playback rate (1.0 = normal speed)"""
        self._ensure_not_disposed()
        if not self._is_created:
            self.create()
        
        try:
            return capi.ca_clock_get_play_rate(self.object_id)
        except Exception as e:
            raise RuntimeError(f"Failed to get play rate: {e}")
    
    @play_rate.setter
    def play_rate(self, rate: float) -> None:
        """Set the playback rate"""
        self._ensure_not_disposed()
        if not self._is_created:
            self.create()
        
        try:
            capi.ca_clock_set_play_rate(self.object_id, rate)
        except Exception as e:
            raise RuntimeError(f"Failed to set play rate: {e}")
    
    def get_current_time(self, time_format: int) -> Dict[str, Any]:
        """Get current time in specified format
        
        Args:
            time_format: Time format constant from ClockTimeFormat
            
        Returns:
            Dictionary with 'format' and 'value' keys
            
        Raises:
            RuntimeError: If getting time fails
        """
        self._ensure_not_disposed()
        if not self._is_created:
            self.create()
        
        try:
            return capi.ca_clock_get_current_time(self.object_id, time_format)
        except Exception as e:
            raise RuntimeError(f"Failed to get current time: {e}")
    
    def get_time_seconds(self) -> float:
        """Get current time in seconds
        
        Returns:
            Current time in seconds
        """
        time_info = self.get_current_time(ClockTimeFormat.SECONDS)
        return float(time_info.get("value", 0.0))
    
    def get_time_beats(self) -> float:
        """Get current time in musical beats
        
        Returns:
            Current time in beats
        """
        time_info = self.get_current_time(ClockTimeFormat.BEATS)
        return float(time_info.get("value", 0.0))
    
    def get_time_samples(self) -> float:
        """Get current time in audio samples
        
        Returns:
            Current time in samples
        """
        time_info = self.get_current_time(ClockTimeFormat.SAMPLES)
        return float(time_info.get("value", 0.0))
    
    def get_time_host(self) -> int:
        """Get current time as host time
        
        Returns:
            Current host time (mach_absolute_time)
        """
        time_info = self.get_current_time(ClockTimeFormat.HOST_TIME)
        return int(time_info.get("value", 0))
    
    def get_smpte_time(self) -> Dict[str, int]:
        """Get current time as SMPTE timecode
        
        Returns:
            Dictionary with SMPTE time components:
            - hours, minutes, seconds, frames
            - subframes, subframe_divisor
            - type, flags
        """
        time_info = self.get_current_time(ClockTimeFormat.SMPTE_TIME)
        value = time_info.get("value", {})
        if isinstance(value, dict):
            return value
        return {}
    
    def dispose(self) -> None:
        """Dispose the clock and free resources"""
        if not self.is_disposed and self._is_created:
            try:
                if self._is_running:
                    self.stop()
                capi.ca_clock_dispose(self.object_id)
            except:
                pass  # Best effort cleanup
            finally:
                self._is_created = False
                self._is_running = False
                super().dispose()
    
    def __enter__(self) -> "AudioClock":
        """Enter context manager"""
        self.create()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and dispose"""
        self.dispose()
    
    def __repr__(self) -> str:
        status = []
        if not self.is_disposed:
            if self._is_created:
                status.append("created")
            if self._is_running:
                status.append("running")
                try:
                    rate = self.play_rate
                    status.append(f"rate={rate:.2f}")
                except:
                    pass
        else:
            status.append("disposed")
        
        status_str = ", ".join(status) if status else "not created"
        return f"AudioClock({status_str})"
