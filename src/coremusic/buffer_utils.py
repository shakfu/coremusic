"""Buffer management utilities for CoreAudio.

This module provides utilities for working with audio buffers, including:
- AudioStreamBasicDescription dataclass for type-safe format handling
- Buffer packing/unpacking utilities
- Format conversion helpers
"""

from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class AudioStreamBasicDescription:
    """Type-safe representation of CoreAudio AudioStreamBasicDescription.

    This dataclass provides a Pythonic interface to AudioStreamBasicDescription,
    with validation and helpful conversion methods.

    Attributes:
        sample_rate: Sample rate in Hz (e.g., 44100.0, 48000.0)
        format_id: Audio format identifier (FourCC code as string or int)
        format_flags: Format-specific flags
        bytes_per_packet: Number of bytes in a packet
        frames_per_packet: Number of frames in a packet
        bytes_per_frame: Number of bytes in a frame
        channels_per_frame: Number of channels per frame (1=mono, 2=stereo)
        bits_per_channel: Number of bits per channel

    Example::

        # Create a standard 44.1kHz stereo float32 format
        format = AudioStreamBasicDescription(
            sample_rate=44100.0,
            format_id='lpcm',
            format_flags=0x29,  # float, packed, non-interleaved
            bytes_per_packet=8,
            frames_per_packet=1,
            bytes_per_frame=8,
            channels_per_frame=2,
            bits_per_channel=32
        )

        # Check properties
        print(format.is_pcm)  # True
        print(format.is_float)  # True
        print(format.is_interleaved)  # False

        # Convert to dictionary for capi
        asbd_dict = format.to_dict()
    """

    sample_rate: float
    format_id: Union[str, int]
    format_flags: int
    bytes_per_packet: int
    frames_per_packet: int
    bytes_per_frame: int
    channels_per_frame: int
    bits_per_channel: int

    def __post_init__(self):
        """Validate format parameters."""
        if self.sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {self.sample_rate}")

        if self.channels_per_frame < 1:
            raise ValueError(f"Invalid channel count: {self.channels_per_frame}")

        if self.bits_per_channel < 1:
            raise ValueError(f"Invalid bits per channel: {self.bits_per_channel}")

        # Convert format_id to int if it's a string
        if isinstance(self.format_id, str):
            if len(self.format_id) == 4:
                # Convert FourCC string to int
                self.format_id = fourcc_to_int(self.format_id)
            else:
                raise ValueError(f"Invalid format_id string: {self.format_id}")

    @property
    def is_pcm(self) -> bool:
        """Check if format is Linear PCM."""
        lpcm = fourcc_to_int('lpcm')
        return self.format_id == lpcm

    @property
    def is_float(self) -> bool:
        """Check if format uses floating point samples."""
        return bool(self.format_flags & 0x01)  # kAudioFormatFlagIsFloat

    @property
    def is_signed_integer(self) -> bool:
        """Check if format uses signed integer samples."""
        return bool(self.format_flags & 0x04)  # kAudioFormatFlagIsSignedInteger

    @property
    def is_packed(self) -> bool:
        """Check if format is packed (no unused bits)."""
        return bool(self.format_flags & 0x08)  # kAudioFormatFlagIsPacked

    @property
    def is_interleaved(self) -> bool:
        """Check if format is interleaved (channels mixed in each frame)."""
        return not bool(self.format_flags & 0x20)  # NOT kAudioFormatFlagIsNonInterleaved

    @property
    def is_non_interleaved(self) -> bool:
        """Check if format is non-interleaved (separate channel buffers)."""
        return bool(self.format_flags & 0x20)  # kAudioFormatFlagIsNonInterleaved

    @property
    def is_big_endian(self) -> bool:
        """Check if format is big endian."""
        return bool(self.format_flags & 0x02)  # kAudioFormatFlagIsBigEndian

    def duration_for_frames(self, num_frames: int) -> float:
        """Calculate duration in seconds for a given number of frames.

        Args:
            num_frames: Number of audio frames

        Returns:
            Duration in seconds
        """
        return num_frames / self.sample_rate

    def bytes_for_frames(self, num_frames: int) -> int:
        """Calculate number of bytes needed for given number of frames.

        Args:
            num_frames: Number of audio frames

        Returns:
            Number of bytes required
        """
        return num_frames * self.bytes_per_frame

    def frames_for_bytes(self, num_bytes: int) -> int:
        """Calculate number of frames from number of bytes.

        Args:
            num_bytes: Number of bytes

        Returns:
            Number of audio frames
        """
        if self.bytes_per_frame == 0:
            return 0
        return num_bytes // self.bytes_per_frame

    def to_dict(self) -> dict:
        """Convert to dictionary suitable for capi functions.

        Returns:
            Dictionary with all format fields
        """
        return {
            'sample_rate': self.sample_rate,
            'format_id': self.format_id,
            'format_flags': self.format_flags,
            'bytes_per_packet': self.bytes_per_packet,
            'frames_per_packet': self.frames_per_packet,
            'bytes_per_frame': self.bytes_per_frame,
            'channels_per_frame': self.channels_per_frame,
            'bits_per_channel': self.bits_per_channel,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AudioStreamBasicDescription':
        """Create from dictionary.

        Args:
            data: Dictionary with format fields

        Returns:
            New AudioStreamBasicDescription instance
        """
        return cls(
            sample_rate=data['sample_rate'],
            format_id=data.get('format_id', 'lpcm'),
            format_flags=data.get('format_flags', 0),
            bytes_per_packet=data.get('bytes_per_packet', 0),
            frames_per_packet=data.get('frames_per_packet', 1),
            bytes_per_frame=data.get('bytes_per_frame', 0),
            channels_per_frame=data['channels_per_frame'],
            bits_per_channel=data.get('bits_per_channel', 16),
        )

    @classmethod
    def pcm_float32_stereo(cls, sample_rate: float = 44100.0) -> 'AudioStreamBasicDescription':
        """Create standard stereo float32 PCM format.

        Args:
            sample_rate: Sample rate in Hz (default: 44100.0)

        Returns:
            AudioStreamBasicDescription for stereo float32
        """
        return cls(
            sample_rate=sample_rate,
            format_id='lpcm',
            format_flags=0x09,  # float | packed
            bytes_per_packet=8,
            frames_per_packet=1,
            bytes_per_frame=8,
            channels_per_frame=2,
            bits_per_channel=32
        )

    @classmethod
    def pcm_int16_stereo(cls, sample_rate: float = 44100.0) -> 'AudioStreamBasicDescription':
        """Create standard stereo int16 PCM format.

        Args:
            sample_rate: Sample rate in Hz (default: 44100.0)

        Returns:
            AudioStreamBasicDescription for stereo int16
        """
        return cls(
            sample_rate=sample_rate,
            format_id='lpcm',
            format_flags=0x0C,  # signed int | packed
            bytes_per_packet=4,
            frames_per_packet=1,
            bytes_per_frame=4,
            channels_per_frame=2,
            bits_per_channel=16
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        format_name = int_to_fourcc(self.format_id) if isinstance(self.format_id, int) else self.format_id
        type_str = "float" if self.is_float else "int"
        layout = "non-interleaved" if self.is_non_interleaved else "interleaved"

        return (
            f"AudioStreamBasicDescription("
            f"{self.sample_rate}Hz, "
            f"{self.channels_per_frame}ch, "
            f"{self.bits_per_channel}-bit {type_str}, "
            f"{format_name}, "
            f"{layout})"
        )


def fourcc_to_int(fourcc: str) -> int:
    """Convert FourCC string to integer.

    Args:
        fourcc: Four-character code string (e.g., 'lpcm')

    Returns:
        Integer representation of FourCC

    Raises:
        ValueError: If fourcc is not exactly 4 characters
    """
    if len(fourcc) != 4:
        raise ValueError(f"FourCC must be exactly 4 characters, got {len(fourcc)}")

    return (
        (ord(fourcc[0]) << 24) |
        (ord(fourcc[1]) << 16) |
        (ord(fourcc[2]) << 8) |
        ord(fourcc[3])
    )


def int_to_fourcc(value: int) -> str:
    """Convert integer to FourCC string.

    Args:
        value: Integer representation of FourCC

    Returns:
        Four-character code string
    """
    return bytes([
        (value >> 24) & 0xFF,
        (value >> 16) & 0xFF,
        (value >> 8) & 0xFF,
        value & 0xFF
    ]).decode('ascii', errors='replace')


def pack_audio_buffer(
    data: bytes,
    format: AudioStreamBasicDescription,
    num_frames: int
) -> Tuple[bytes, int]:
    """Pack audio data into buffer suitable for CoreAudio.

    Args:
        data: Raw audio data
        format: Audio format description
        num_frames: Number of frames to pack

    Returns:
        Tuple of (packed_data, actual_frames)

    Example::

        format = AudioStreamBasicDescription.pcm_float32_stereo()
        packed, count = pack_audio_buffer(raw_data, format, 1024)
    """
    expected_bytes = format.bytes_for_frames(num_frames)
    original_len = len(data)

    if len(data) < expected_bytes:
        # Calculate actual frames from original data before padding
        actual_frames = format.frames_for_bytes(original_len)
        # Pad with zeros to requested size
        data = data + b'\x00' * (expected_bytes - len(data))
    elif len(data) > expected_bytes:
        # Truncate
        data = data[:expected_bytes]
        actual_frames = num_frames
    else:
        actual_frames = num_frames

    return (data, actual_frames)


def unpack_audio_buffer(
    data: bytes,
    format: AudioStreamBasicDescription
) -> Tuple[bytes, int]:
    """Unpack CoreAudio buffer data.

    Args:
        data: Packed audio buffer data
        format: Audio format description

    Returns:
        Tuple of (unpacked_data, num_frames)

    Example::

        format = AudioStreamBasicDescription.pcm_float32_stereo()
        unpacked, frames = unpack_audio_buffer(buffer_data, format)
    """
    num_frames = format.frames_for_bytes(len(data))
    return (data, num_frames)


def convert_buffer_format(
    data: bytes,
    from_format: AudioStreamBasicDescription,
    to_format: AudioStreamBasicDescription
) -> bytes:
    """Convert audio buffer between formats (simplified).

    This is a simple conversion for common cases. For complex conversions,
    use AudioConverter or ExtendedAudioFile.

    Args:
        data: Source audio data
        from_format: Source format
        to_format: Target format

    Returns:
        Converted audio data

    Raises:
        ValueError: If conversion is not supported

    Note:
        Currently supports:
        - Float32 <-> Int16 conversion
        - Mono <-> Stereo conversion (simple duplication/averaging)
    """
    import numpy as np

    # Get source samples
    if from_format.is_float:
        samples = np.frombuffer(data, dtype=np.float32)
    elif from_format.is_signed_integer and from_format.bits_per_channel == 16:
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        raise ValueError(f"Unsupported source format: {from_format}")

    # Reshape for channels
    num_frames = len(samples) // from_format.channels_per_frame
    if from_format.channels_per_frame > 1:
        samples = samples.reshape(num_frames, from_format.channels_per_frame)

    # Convert channels
    if from_format.channels_per_frame != to_format.channels_per_frame:
        if from_format.channels_per_frame == 1 and to_format.channels_per_frame == 2:
            # Mono to stereo: duplicate
            samples = np.column_stack([samples, samples])
        elif from_format.channels_per_frame == 2 and to_format.channels_per_frame == 1:
            # Stereo to mono: average
            samples = samples.mean(axis=1)
        else:
            raise ValueError(
                f"Unsupported channel conversion: "
                f"{from_format.channels_per_frame} -> {to_format.channels_per_frame}"
            )

    # Flatten
    samples = samples.flatten() if samples.ndim > 1 else samples

    # Convert to target format
    if to_format.is_float:
        result = samples.astype(np.float32).tobytes()
    elif to_format.is_signed_integer and to_format.bits_per_channel == 16:
        result = (samples * 32767.0).astype(np.int16).tobytes()
    else:
        raise ValueError(f"Unsupported target format: {to_format}")

    return result


def calculate_buffer_size(
    duration_seconds: float,
    format: AudioStreamBasicDescription
) -> int:
    """Calculate buffer size in bytes for given duration.

    Args:
        duration_seconds: Desired buffer duration in seconds
        format: Audio format description

    Returns:
        Buffer size in bytes

    Example::

        format = AudioStreamBasicDescription.pcm_float32_stereo(44100.0)
        # Calculate buffer for 0.1 seconds (100ms)
        size = calculate_buffer_size(0.1, format)  # Returns 35280 bytes
    """
    num_frames = int(duration_seconds * format.sample_rate)
    return format.bytes_for_frames(num_frames)


def optimal_buffer_size(
    format: AudioStreamBasicDescription,
    latency_ms: float = 10.0
) -> Tuple[int, int]:
    """Calculate optimal buffer size for given latency target.

    Args:
        format: Audio format description
        latency_ms: Target latency in milliseconds (default: 10ms)

    Returns:
        Tuple of (buffer_size_bytes, buffer_size_frames)

    Example::

        format = AudioStreamBasicDescription.pcm_float32_stereo(44100.0)
        bytes, frames = optimal_buffer_size(format, latency_ms=10.0)
        print(f"Buffer: {frames} frames ({bytes} bytes)")
    """
    duration_seconds = latency_ms / 1000.0
    num_frames = int(duration_seconds * format.sample_rate)

    # Round to power of 2 for efficiency
    import math
    num_frames = 2 ** math.ceil(math.log2(num_frames))

    num_bytes = format.bytes_for_frames(num_frames)

    return (num_bytes, num_frames)
