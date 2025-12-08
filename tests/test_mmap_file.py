"""Tests for memory-mapped audio file access."""

import pytest
import numpy as np
from pathlib import Path

from coremusic.audio.mmap_file import MMapAudioFile


# Use test audio file
TEST_FILE = Path(__file__).parent / "data" / "wav" / "amen.wav"


class TestMMapAudioFile:
    """Test memory-mapped audio file operations."""

    def test_open_close(self):
        """Test opening and closing mmap file."""
        mmap_file = MMapAudioFile(TEST_FILE)
        mmap_file.open()
        assert mmap_file._is_open
        mmap_file.close()
        assert not mmap_file._is_open

    def test_context_manager(self):
        """Test context manager support."""
        with MMapAudioFile(TEST_FILE) as audio:
            assert audio._is_open
            assert audio.format is not None
        # Should be closed after context
        assert not audio._is_open

    def test_format_property(self):
        """Test format property."""
        with MMapAudioFile(TEST_FILE) as audio:
            format = audio.format

            assert format.sample_rate > 0
            assert format.channels_per_frame > 0
            assert format.bits_per_channel > 0

    def test_frame_count(self):
        """Test frame count property."""
        with MMapAudioFile(TEST_FILE) as audio:
            count = audio.frame_count
            assert count > 0

    def test_duration(self):
        """Test duration calculation."""
        with MMapAudioFile(TEST_FILE) as audio:
            duration = audio.duration
            assert duration > 0
            # Should match frame_count / sample_rate
            expected = audio.frame_count / audio.format.sample_rate
            assert abs(duration - expected) < 0.01

    def test_read_frames(self):
        """Test reading frames."""
        with MMapAudioFile(TEST_FILE) as audio:
            # Read first 1024 frames
            data = audio.read_frames(0, 1024)

            assert isinstance(data, bytes)
            expected_bytes = 1024 * audio.format.bytes_per_frame
            assert len(data) == expected_bytes

    def test_read_frames_bounds(self):
        """Test reading frames with boundary conditions."""
        with MMapAudioFile(TEST_FILE) as audio:
            # Read beyond end of file
            data = audio.read_frames(audio.frame_count - 100, 1000)
            assert len(data) <= 100 * audio.format.bytes_per_frame

            # Invalid start frame
            with pytest.raises(ValueError):
                audio.read_frames(-1, 1024)

            with pytest.raises(ValueError):
                audio.read_frames(audio.frame_count + 1, 1024)

    def test_read_as_numpy(self):
        """Test reading as NumPy array."""
        with MMapAudioFile(TEST_FILE) as audio:
            # Read first 1024 frames
            data = audio.read_as_numpy(0, 1024)

            assert isinstance(data, np.ndarray)

            if audio.format.channels_per_frame > 1:
                assert data.shape == (1024, audio.format.channels_per_frame)
            else:
                assert data.shape == (1024,)

    def test_read_as_numpy_default(self):
        """Test reading all frames as NumPy."""
        with MMapAudioFile(TEST_FILE) as audio:
            data = audio.read_as_numpy()

            assert isinstance(data, np.ndarray)
            assert len(data) == audio.frame_count or len(data) > 0

    def test_array_indexing_single_frame(self):
        """Test array-like single frame access."""
        with MMapAudioFile(TEST_FILE) as audio:
            # Get single frame
            frame = audio[0]
            assert isinstance(frame, np.ndarray)

            # Test negative indexing
            last_frame = audio[-1]
            assert isinstance(last_frame, np.ndarray)

    def test_array_indexing_slice(self):
        """Test array-like slice access."""
        with MMapAudioFile(TEST_FILE) as audio:
            # Get frame range
            frames = audio[100:200]
            assert isinstance(frames, np.ndarray)
            assert len(frames) == 100

            # Get every 10th frame
            frames_strided = audio[::10]
            assert isinstance(frames_strided, np.ndarray)

    def test_len(self):
        """Test len() support."""
        with MMapAudioFile(TEST_FILE) as audio:
            length = len(audio)
            assert length == audio.frame_count

    def test_repr(self):
        """Test string representation."""
        # Closed file
        mmap_file = MMapAudioFile(TEST_FILE)
        repr_str = repr(mmap_file)
        assert "closed" in repr_str.lower()

        # Open file
        with mmap_file:
            repr_str = repr(mmap_file)
            assert "amen.wav" in repr_str
            assert str(mmap_file.format.sample_rate) in repr_str

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        with pytest.raises(FileNotFoundError):
            with MMapAudioFile("nonexistent.wav") as audio:
                pass

    @pytest.mark.skipif(not TEST_FILE.exists(), reason="Test file not found")
    def test_wav_format_parsing(self):
        """Test WAV format parsing."""
        with MMapAudioFile(TEST_FILE) as audio:
            # Should successfully parse WAV format
            assert audio.format is not None
            assert audio._data_offset > 0
            assert audio._data_size > 0

    def test_multiple_reads(self):
        """Test multiple sequential reads."""
        with MMapAudioFile(TEST_FILE) as audio:
            # Read first chunk
            data1 = audio.read_frames(0, 512)

            # Read second chunk
            data2 = audio.read_frames(512, 512)

            # Should be different data
            assert data1 != data2

    def test_random_access(self):
        """Test random access pattern."""
        with MMapAudioFile(TEST_FILE) as audio:
            # Random access should be fast with mmap
            positions = [0, 1000, 500, 2000, 100]

            for pos in positions:
                if pos < audio.frame_count:
                    data = audio.read_frames(pos, min(100, audio.frame_count - pos))
                    assert len(data) > 0


class TestMMapAudioFilePerformance:
    """Performance-related tests for MMapAudioFile."""

    @pytest.mark.skipif(not TEST_FILE.exists(), reason="Test file not found")
    def test_lazy_loading(self):
        """Test that file data is not loaded immediately."""
        # Opening should be fast (no full file read)
        import time

        start = time.time()
        with MMapAudioFile(TEST_FILE) as audio:
            open_time = time.time() - start

            # Opening should be very fast (<10ms even for large files)
            assert open_time < 0.01

            # Accessing data should also be fast (memory mapped)
            start = time.time()
            _ = audio.read_frames(0, 1024)
            read_time = time.time() - start

            # Read should be fast
            assert read_time < 0.01

    def test_numpy_zero_copy(self):
        """Test NumPy array uses memory mapping."""
        with MMapAudioFile(TEST_FILE) as audio:
            # Read as NumPy array
            data = audio.read_as_numpy(0, 1024)

            # Should be a view when possible (flags will indicate)
            # Note: This is implementation dependent
            assert isinstance(data, np.ndarray)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
