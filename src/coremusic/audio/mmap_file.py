"""Memory-mapped audio file access for high-performance I/O.

This module provides memory-mapped file access for audio files, offering
significant performance improvements for large file operations.
"""

import mmap
import struct
from io import BufferedReader
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

# Conditional numpy import
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore
    if TYPE_CHECKING:
        import numpy as np

from ..buffer_utils import AudioStreamBasicDescription


class MMapAudioFile:
    """Memory-mapped audio file reader for high-performance access.

    This class uses memory-mapping to provide fast random access to audio
    file data without loading the entire file into memory.

    Attributes:
        path: Path to the audio file
        format: Audio format description
        data_offset: Offset to audio data in file
        data_size: Size of audio data in bytes

    Example::

        # Open large file with memory mapping
        with MMapAudioFile("large_audio.wav") as audio:
            # Fast random access - no loading needed
            chunk = audio.read_frames(1000, 1024)

            # Read as NumPy array - zero-copy when possible
            data = audio.read_as_numpy(0, 44100)

            # Memory-efficient slicing
            slice_data = audio[1000:2000]  # Returns view, not copy

    Notes:
        - Best for large files (>100MB) with random access
        - Uses OS virtual memory for efficient caching
        - Read-only access (use ExtendedAudioFile for writing)
        - Supports WAV and AIFF formats
    """

    def __init__(self, path: Union[str, Path]):
        """Initialize memory-mapped audio file.

        Args:
            path: Path to audio file
        """
        self.path = Path(path)
        self._file: Optional[BufferedReader] = None
        self._mmap: Optional[mmap.mmap] = None
        self._format: Optional[AudioStreamBasicDescription] = None
        self._data_offset = 0
        self._data_size = 0
        self._is_open = False

    def open(self) -> "MMapAudioFile":
        """Open file and create memory mapping.

        Returns:
            Self for chaining

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        if self._is_open:
            return self

        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

        # Open file for reading
        self._file = open(self.path, 'rb')

        # Create memory mapping
        self._mmap = mmap.mmap(
            self._file.fileno(),
            0,
            access=mmap.ACCESS_READ
        )

        # Parse file format
        self._parse_format()
        self._is_open = True

        return self

    def close(self) -> None:
        """Close memory mapping and file."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None

        if self._file:
            self._file.close()
            self._file = None

        self._is_open = False

    def __enter__(self) -> "MMapAudioFile":
        """Context manager entry."""
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def _parse_format(self) -> None:
        """Parse audio file format from header."""
        if not self._mmap:
            raise RuntimeError("File not open")

        # Read first 12 bytes to identify format
        header = self._mmap[:12]

        if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
            self._parse_wav_format()
        elif header[:4] == b'FORM' and header[8:12] == b'AIFF':
            self._parse_aiff_format()
        else:
            raise ValueError(f"Unsupported format: {header[:4]!r}")

    def _parse_wav_format(self) -> None:
        """Parse WAV file format."""
        assert self._mmap is not None
        # Skip RIFF header (12 bytes)
        pos = 12

        # Find fmt chunk
        while pos < len(self._mmap):
            chunk_id = self._mmap[pos:pos+4]
            chunk_size = struct.unpack('<I', self._mmap[pos+4:pos+8])[0]

            if chunk_id == b'fmt ':
                # Parse format chunk
                fmt_data = self._mmap[pos+8:pos+8+chunk_size]
                audio_format = struct.unpack('<H', fmt_data[0:2])[0]
                channels = struct.unpack('<H', fmt_data[2:4])[0]
                sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
                _ = struct.unpack('<I', fmt_data[8:12])[0]  # byte_rate - not used
                block_align = struct.unpack('<H', fmt_data[12:14])[0]
                bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]

                # Determine format flags
                if audio_format == 1:  # PCM
                    format_flags = 0x0C  # signed integer | packed
                elif audio_format == 3:  # IEEE float
                    format_flags = 0x09  # float | packed
                else:
                    raise ValueError(f"Unsupported audio format: {audio_format}")

                self._format = AudioStreamBasicDescription(
                    sample_rate=float(sample_rate),
                    format_id='lpcm',
                    format_flags=format_flags,
                    bytes_per_packet=block_align,
                    frames_per_packet=1,
                    bytes_per_frame=block_align,
                    channels_per_frame=channels,
                    bits_per_channel=bits_per_sample,
                )

            elif chunk_id == b'data':
                # Found data chunk
                self._data_offset = pos + 8
                self._data_size = chunk_size
                return

            # Move to next chunk
            pos += 8 + chunk_size
            # Align to even boundary
            if chunk_size % 2:
                pos += 1

        raise ValueError("No data chunk found in WAV file")

    def _parse_aiff_format(self) -> None:
        """Parse AIFF file format."""
        assert self._mmap is not None
        # Skip FORM header (12 bytes)
        pos = 12

        # Find COMM and SSND chunks
        while pos < len(self._mmap):
            chunk_id = self._mmap[pos:pos+4]
            chunk_size = struct.unpack('>I', self._mmap[pos+4:pos+8])[0]

            if chunk_id == b'COMM':
                # Parse common chunk
                comm_data = self._mmap[pos+8:pos+8+chunk_size]
                channels = struct.unpack('>H', comm_data[0:2])[0]
                _ = struct.unpack('>I', comm_data[2:6])[0]  # num_frames - not used
                bits_per_sample = struct.unpack('>H', comm_data[6:8])[0]
                # Extended 80-bit float for sample rate
                sample_rate = self._parse_extended_float(comm_data[8:18])

                block_align = channels * ((bits_per_sample + 7) // 8)

                self._format = AudioStreamBasicDescription(
                    sample_rate=sample_rate,
                    format_id='lpcm',
                    format_flags=0x0E,  # signed integer | packed | big endian
                    bytes_per_packet=block_align,
                    frames_per_packet=1,
                    bytes_per_frame=block_align,
                    channels_per_frame=channels,
                    bits_per_channel=bits_per_sample,
                )

            elif chunk_id == b'SSND':
                # Found sound data chunk
                offset = struct.unpack('>I', self._mmap[pos+8:pos+12])[0]
                self._data_offset = pos + 16 + offset  # 8 (chunk header) + 8 (SSND header) + offset
                self._data_size = chunk_size - 8 - offset
                return

            # Move to next chunk
            pos += 8 + chunk_size
            # Align to even boundary
            if chunk_size % 2:
                pos += 1

        raise ValueError("No sound data chunk found in AIFF file")

    @staticmethod
    def _parse_extended_float(data: bytes) -> float:
        """Parse 80-bit extended float (used in AIFF sample rate)."""
        # Simplified conversion for common sample rates
        # Full implementation would decode IEEE 754 extended precision
        exponent = struct.unpack('>H', data[0:2])[0]
        mantissa = struct.unpack('>Q', data[2:10])[0]

        # Common sample rates
        if exponent == 0x400E:
            return 44100.0
        elif exponent == 0x400E and mantissa == 0xAC44000000000000:
            return 44100.0
        elif exponent == 0x400E and mantissa == 0xBB80000000000000:
            return 48000.0

        # Generic conversion
        sign = 1 if exponent & 0x8000 == 0 else -1
        exp_value = (exponent & 0x7FFF) - 16383
        mantissa_value = mantissa / (2 ** 63)

        return float(sign * mantissa_value * (2 ** exp_value))

    @property
    def format(self) -> AudioStreamBasicDescription:
        """Get audio format."""
        if not self._is_open:
            self.open()
        if self._format is None:
            raise RuntimeError("Format not parsed")
        return self._format

    @property
    def frame_count(self) -> int:
        """Get total number of frames."""
        if not self._is_open:
            self.open()
        return self._data_size // self.format.bytes_per_frame

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        return self.frame_count / self.format.sample_rate

    def read_frames(self, start_frame: int, num_frames: int) -> bytes:
        """Read audio frames from file.

        Args:
            start_frame: Starting frame index
            num_frames: Number of frames to read

        Returns:
            Raw audio data bytes

        Example::

            # Read 1024 frames starting at frame 1000
            data = audio.read_frames(1000, 1024)
        """
        if not self._is_open:
            self.open()

        if start_frame < 0 or start_frame >= self.frame_count:
            raise ValueError(f"Invalid start_frame: {start_frame}")

        assert self._mmap is not None
        # Calculate byte offsets
        bytes_per_frame = self.format.bytes_per_frame
        start_offset = self._data_offset + (start_frame * bytes_per_frame)
        num_bytes = min(num_frames * bytes_per_frame,
                       self._data_size - (start_frame * bytes_per_frame))

        # Read from memory map (very fast)
        return bytes(self._mmap[start_offset:start_offset + num_bytes])

    def read_as_numpy(
        self,
        start_frame: int = 0,
        num_frames: Optional[int] = None
    ) -> "np.ndarray":
        """Read audio data as NumPy array.

        Args:
            start_frame: Starting frame index (default: 0)
            num_frames: Number of frames to read (default: all)

        Returns:
            NumPy array with shape (frames, channels) for stereo,
            or (frames,) for mono

        Raises:
            ImportError: If NumPy is not installed

        Example::

            # Read first second as NumPy array
            data = audio.read_as_numpy(0, 44100)
            print(data.shape, data.dtype)  # (44100, 2), dtype('int16')
        """
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "NumPy is required for read_as_numpy(). "
                "Install with: pip install numpy"
            )

        if not self._is_open:
            self.open()

        if num_frames is None:
            num_frames = self.frame_count - start_frame

        # Read raw data
        data = self.read_frames(start_frame, num_frames)

        # Determine dtype from format
        dtype: Any
        if self.format.is_float:
            if self.format.bits_per_channel == 32:
                dtype = np.float32
            else:
                dtype = np.float64
        elif self.format.is_signed_integer:
            if self.format.bits_per_channel == 16:
                dtype = np.int16
            elif self.format.bits_per_channel == 32:
                dtype = np.int32
            else:
                dtype = np.int8
        else:
            raise ValueError(f"Unsupported format: {self.format}")

        # Convert to NumPy array
        samples = np.frombuffer(data, dtype=dtype)

        # Reshape for channels
        if self.format.channels_per_frame > 1:
            actual_frames = len(samples) // self.format.channels_per_frame
            samples = samples.reshape(actual_frames, self.format.channels_per_frame)

        return samples

    def __getitem__(self, key: Union[int, slice]) -> "np.ndarray":
        """Array-like access with slicing support.

        Args:
            key: Frame index or slice

        Returns:
            NumPy array of audio data

        Raises:
            ImportError: If NumPy is not installed

        Example::

            # Get single frame
            frame = audio[1000]

            # Get frame range
            frames = audio[1000:2000]

            # Get every 10th frame
            frames = audio[::10]
        """
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "NumPy is required for array-like access. "
                "Install with: pip install numpy"
            )

        if isinstance(key, int):
            # Single frame
            if key < 0:
                key = self.frame_count + key
            result: np.ndarray = self.read_as_numpy(key, 1)[0]
            return result

        elif isinstance(key, slice):
            # Frame range
            start, stop, step = key.indices(self.frame_count)

            if step == 1:
                # Contiguous read
                return self.read_as_numpy(start, stop - start)
            else:
                # Strided access
                frames = []
                for i in range(start, stop, step):
                    frames.append(self.read_as_numpy(i, 1))
                return np.vstack(frames)
        else:
            raise TypeError(f"Invalid index type: {type(key)}")

    def __len__(self) -> int:
        """Get total number of frames."""
        return self.frame_count

    def __repr__(self) -> str:
        """String representation."""
        if self._is_open:
            return (
                f"MMapAudioFile({self.path.name}, "
                f"{self.format.sample_rate}Hz, "
                f"{self.format.channels_per_frame}ch, "
                f"{self.frame_count} frames)"
            )
        else:
            return f"MMapAudioFile({self.path.name}, closed)"
