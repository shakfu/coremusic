#!/usr/bin/env python3
"""Tutorial: Audio File Basics

This module demonstrates basic audio file operations with coremusic.
All examples are executable doctests.

Run with: pytest tests/tutorials/test_audio_file_basics.py --doctest-modules -v
"""
from __future__ import annotations

import os
from pathlib import Path

# Test data path
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "wav"
TEST_AUDIO_FILE = TEST_DATA_DIR / "amen.wav"


def get_test_audio_path() -> str:
    """Get path to test audio file.

    >>> path = get_test_audio_path()
    >>> Path(path).exists()
    True
    """
    return str(TEST_AUDIO_FILE)


def open_audio_file_context_manager():
    """Open audio file using context manager (recommended).

    The AudioFile class supports context managers for automatic cleanup:

    >>> import coremusic as cm
    >>> path = get_test_audio_path()
    >>> with cm.AudioFile(path) as audio:
    ...     # File is automatically opened
    ...     assert audio.duration > 0
    ...     # Access format info
    ...     fmt = audio.format
    ...     assert fmt.sample_rate > 0
    >>> # File is automatically closed here
    """
    pass


def get_audio_duration():
    """Get the duration of an audio file.

    >>> import coremusic as cm
    >>> path = get_test_audio_path()
    >>> with cm.AudioFile(path) as audio:
    ...     duration = audio.duration
    ...     assert isinstance(duration, float)
    ...     assert duration > 0
    ...     print(f"Duration: {duration:.2f}s")  # doctest: +ELLIPSIS
    Duration: ...s
    """
    pass


def get_audio_format():
    """Get detailed format information from an audio file.

    >>> import coremusic as cm
    >>> path = get_test_audio_path()
    >>> with cm.AudioFile(path) as audio:
    ...     fmt = audio.format
    ...     # Sample rate
    ...     assert fmt.sample_rate > 0
    ...     # Channels
    ...     assert fmt.channels_per_frame >= 1
    ...     # Bit depth
    ...     assert fmt.bits_per_channel > 0
    ...     # Format ID (e.g., 'lpcm' for Linear PCM)
    ...     assert len(fmt.format_id) > 0
    """
    pass


def get_frame_count():
    """Calculate approximate frame count from duration and sample rate.

    >>> import coremusic as cm
    >>> path = get_test_audio_path()
    >>> with cm.AudioFile(path) as audio:
    ...     # Calculate frames from duration
    ...     frames = int(audio.duration * audio.format.sample_rate)
    ...     assert isinstance(frames, int)
    ...     assert frames > 0
    """
    pass


def read_audio_packets():
    """Read audio data as packets (frames).

    >>> import coremusic as cm
    >>> path = get_test_audio_path()
    >>> with cm.AudioFile(path) as audio:
    ...     # Read first 1000 packets starting at packet 0
    ...     data, packets_read = audio.read_packets(0, 1000)
    ...     assert isinstance(data, bytes)
    ...     assert len(data) > 0
    ...     assert packets_read > 0
    ...     assert packets_read <= 1000
    """
    pass


def read_audio_in_chunks():
    """Read audio file in chunks for memory efficiency.

    >>> import coremusic as cm
    >>> path = get_test_audio_path()
    >>> with cm.AudioFile(path) as audio:
    ...     chunk_size = 4096
    ...     # Calculate total frames from duration
    ...     total_frames = int(audio.duration * audio.format.sample_rate)
    ...     current_packet = 0
    ...     chunks_read = 0
    ...     while current_packet < total_frames:
    ...         remaining = total_frames - current_packet
    ...         to_read = min(chunk_size, remaining)
    ...         data, count = audio.read_packets(current_packet, to_read)
    ...         if count == 0:
    ...             break
    ...         current_packet += count
    ...         chunks_read += 1
    ...     assert chunks_read > 0
    """
    pass


def detect_audio_format_type():
    """Detect the type of audio format.

    >>> import coremusic as cm
    >>> path = get_test_audio_path()
    >>> with cm.AudioFile(path) as audio:
    ...     fmt = audio.format
    ...     format_id = fmt.format_id
    ...     # WAV files are typically Linear PCM
    ...     assert format_id == 'lpcm'
    """
    pass


def check_format_properties():
    """Check various format properties.

    >>> import coremusic as cm
    >>> path = get_test_audio_path()
    >>> with cm.AudioFile(path) as audio:
    ...     fmt = audio.format
    ...     # Check if PCM
    ...     is_pcm = fmt.format_id == 'lpcm'
    ...     assert is_pcm
    ...     # Check channel count
    ...     is_stereo = fmt.channels_per_frame == 2
    ...     is_mono = fmt.channels_per_frame == 1
    ...     assert is_stereo or is_mono
    """
    pass


def audio_file_error_handling():
    """Handle errors when opening audio files.

    >>> import coremusic as cm
    >>> try:
    ...     with cm.AudioFile("nonexistent_file.wav") as audio:
    ...         pass
    ... except (cm.AudioFileError, FileNotFoundError, OSError):
    ...     pass  # Expected error
    """
    pass


def format_bytes_per_frame():
    """Calculate bytes per frame from format.

    >>> import coremusic as cm
    >>> path = get_test_audio_path()
    >>> with cm.AudioFile(path) as audio:
    ...     fmt = audio.format
    ...     bytes_per_frame = fmt.bytes_per_frame
    ...     # For 16-bit stereo: 2 bytes * 2 channels = 4
    ...     # For 16-bit mono: 2 bytes * 1 channel = 2
    ...     expected = (fmt.bits_per_channel // 8) * fmt.channels_per_frame
    ...     assert bytes_per_frame == expected
    """
    pass


def calculate_bitrate():
    """Calculate the bitrate of an audio file.

    >>> import coremusic as cm
    >>> path = get_test_audio_path()
    >>> with cm.AudioFile(path) as audio:
    ...     fmt = audio.format
    ...     # Bitrate = sample_rate * bytes_per_frame * 8 bits/byte
    ...     bitrate_bps = fmt.sample_rate * fmt.bytes_per_frame * 8
    ...     bitrate_kbps = bitrate_bps / 1000
    ...     assert bitrate_kbps > 0
    ...     # CD quality (44.1kHz, 16-bit stereo) is ~1411 kbps
    ...     print(f"Bitrate: {bitrate_kbps:.0f} kbps")  # doctest: +ELLIPSIS
    Bitrate: ... kbps
    """
    pass


# Test runner
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
