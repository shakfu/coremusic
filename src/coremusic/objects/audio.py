"""Audio file and format classes for coremusic.

This module provides classes for working with audio files and formats:
- AudioFormat: Represents audio stream format (sample rate, channels, etc.)
- AudioFile: Read audio files
- AudioFileStream: Parse streaming audio data
- AudioConverter: Convert between audio formats
- ExtendedAudioFile: High-level audio file I/O with automatic conversion
- AudioBuffer: Audio buffer for queue operations
- AudioQueue: Audio queue for buffered playback/recording
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from .. import capi
from .exceptions import AudioConverterError, AudioFileError, AudioQueueError

# Check if NumPy is available
try:
    import numpy as np
    from numpy.typing import NDArray

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore
    if TYPE_CHECKING:
        from numpy.typing import NDArray

__all__ = [
    "AudioFormat",
    "AudioFile",
    "AudioFileStream",
    "AudioConverter",
    "ExtendedAudioFile",
    "AudioBuffer",
    "AudioQueue",
]


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
        """Read audio packets from the file.

        Args:
            start_packet: Starting packet index (must be non-negative)
            packet_count: Number of packets to read (must be positive)

        Returns:
            Tuple of (audio_data_bytes, packets_read)

        Raises:
            ValueError: If start_packet < 0 or packet_count <= 0
            AudioFileError: If reading fails

        Example::

            import coremusic as cm

            # Read audio data in chunks
            with cm.AudioFile("audio.wav") as audio:
                chunk_size = 4096
                offset = 0

                while True:
                    data, packets_read = audio.read_packets(offset, chunk_size)
                    if packets_read == 0:
                        break
                    # Process data...
                    offset += packets_read
        """
        if start_packet < 0:
            raise ValueError(f"start_packet must be non-negative, got {start_packet}")
        if packet_count <= 0:
            raise ValueError(f"packet_count must be positive, got {packet_count}")

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

        if start_packet < 0:
            raise ValueError(f"start_packet must be non-negative, got {start_packet}")
        if packet_count is not None and packet_count <= 0:
            raise ValueError(f"packet_count must be positive, got {packet_count}")

        self._ensure_not_disposed()
        if not self._is_open:
            self.open()

        try:
            # Get format information
            format = self.format

            # If packet_count not specified, read all remaining packets
            if packet_count is None:
                # Get total packet count from file
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
                except Exception:
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
        """Parse audio data bytes from a streaming source

        Used for parsing audio file data incrementally, such as when
        reading from a network stream or progressive download.

        Args:
            data: Raw audio file data bytes (e.g., WAV, MP3, AAC chunks)

        Raises:
            AudioFileError: If parsing fails

        Example::

            stream = AudioFileStream()
            with open("audio.mp3", "rb") as f:
                while chunk := f.read(4096):
                    stream.parse_bytes(chunk)
                    if stream.ready_to_produce_packets:
                        # Process parsed packets
                        pass
        """
        self._ensure_not_disposed()
        if not self._is_open:
            self.open()
        try:
            capi.audio_file_stream_parse_bytes(self.object_id, data)
        except Exception as e:
            raise AudioFileError(f"Failed to parse bytes: {e}")

    def seek(self, packet_offset: int) -> None:
        """Seek to packet offset.

        Args:
            packet_offset: Packet offset to seek to (must be non-negative)

        Raises:
            ValueError: If packet_offset < 0
            AudioFileError: If stream not open or seek fails
        """
        if packet_offset < 0:
            raise ValueError(f"packet_offset must be non-negative, got {packet_offset}")

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

    def __repr__(self) -> str:
        status = "open" if self._is_open else "closed"
        return f"AudioFileStream({status})"

    def dispose(self) -> None:
        """Dispose of the audio file stream"""
        if not self.is_disposed:
            if self._is_open:
                try:
                    capi.audio_file_stream_close(self.object_id)
                except Exception:
                    pass  # Best effort cleanup
                finally:
                    self._is_open = False
            super().dispose()


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
            audio_data: Input audio data in source format (raw PCM samples)

        Returns:
            Converted audio data in destination format

        Raises:
            AudioConverterError: If conversion fails

        Example::

            # Convert stereo to mono (simple format conversion)
            source = AudioFormat(44100.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)
            dest = AudioFormat(44100.0, 'lpcm', channels_per_frame=1, bits_per_channel=16)

            with AudioConverter(source, dest) as converter:
                stereo_data = b'\\x00\\x00\\xff\\xff' * 1024  # Raw 16-bit stereo samples
                mono_data = converter.convert(stereo_data)
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
        if input_packet_count <= 0:
            raise ValueError(f"input_packet_count must be positive, got {input_packet_count}")
        if output_packet_count is not None and output_packet_count <= 0:
            raise ValueError(f"output_packet_count must be positive, got {output_packet_count}")
        if not isinstance(input_data, (bytes, bytearray)):
            raise TypeError(f"input_data must be bytes or bytearray, got {type(input_data).__name__}")
        if len(input_data) == 0:
            raise ValueError("input_data cannot be empty")

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
            property_id: Property ID (from capi.get_audio_converter_property_*())
            data: Property data as bytes (use struct.pack for binary encoding)

        Raises:
            AudioConverterError: If setting property fails

        Example::

            import struct
            import coremusic as cm

            converter = AudioConverter(source_fmt, dest_fmt)

            # Set bitrate to 128 kbps (requires UInt32)
            bitrate_prop = cm.capi.get_audio_converter_property_bit_rate()
            converter.set_property(bitrate_prop, struct.pack('<I', 128000))

            # Set quality (requires UInt32, 0=lowest, 127=highest)
            quality_prop = cm.capi.get_audio_converter_property_quality()
            converter.set_property(quality_prop, struct.pack('<I', 127))
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
            except Exception:
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
            num_frames: Number of frames to read (must be positive)

        Returns:
            Tuple of (audio_data_bytes, frames_read)

        Raises:
            ValueError: If num_frames <= 0
            AudioFileError: If reading fails
        """
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")

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
            audio_data: Audio data bytes (raw PCM samples)

        Raises:
            AudioFileError: If writing fails

        Example::

            import struct
            import coremusic as cm

            # Create output file for stereo 16-bit PCM
            file_format = cm.AudioFormat(
                sample_rate=44100.0,
                format_id='lpcm',
                channels_per_frame=2,
                bits_per_channel=16
            )

            with cm.ExtendedAudioFile.create("output.wav", 'WAVE', file_format) as out_file:
                # Generate 1 second of 440Hz sine wave (stereo)
                samples = []
                for i in range(44100):
                    value = int(32767 * 0.5 * (i % 100) / 100)  # Simple sawtooth
                    samples.extend([value, value])  # Stereo

                audio_data = struct.pack(f'<{len(samples)}h', *samples)
                out_file.write(num_frames=44100, audio_data=audio_data)
        """
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        if not isinstance(audio_data, (bytes, bytearray)):
            raise TypeError(f"audio_data must be bytes or bytearray, got {type(audio_data).__name__}")
        if len(audio_data) == 0:
            raise ValueError("audio_data cannot be empty")

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
                except Exception:
                    pass  # Best effort cleanup
                finally:
                    self._is_open = False
            super().dispose()


class AudioBuffer(capi.CoreAudioObject):
    """Audio buffer for queue operations"""

    def __init__(self, queue_id: int, buffer_size: int):
        super().__init__()
        self._queue_id = queue_id
        self._buffer_size = buffer_size

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    def __repr__(self) -> str:
        return f"AudioBuffer(size={self._buffer_size})"


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

    def __repr__(self) -> str:
        if self.is_disposed:
            return "AudioQueue(disposed)"
        return f"AudioQueue({self._format.sample_rate}Hz, {self._format.channels_per_frame}ch, buffers={len(self._buffers)})"

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
