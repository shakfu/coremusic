#!/usr/bin/env python3
"""
Tests for NumPy integration in AudioFormat and AudioFile
"""

import pytest
import coremusic as cm
import os


if not cm.NUMPY_AVAILABLE:
    pytest.skip("NumPy is not available", allow_module_level=True)

import numpy as np


class TestAudioFormatNumPy:
    """Tests for AudioFormat.to_numpy_dtype()"""

    def test_to_numpy_dtype_16bit_pcm(self):
        """Test converting 16-bit PCM format to NumPy dtype"""
        format = cm.AudioFormat(
            sample_rate=44100.0,
            format_id="lpcm",
            channels_per_frame=2,
            bits_per_channel=16,
        )

        dtype = format.to_numpy_dtype()
        assert dtype == np.dtype(np.int16)

    def test_to_numpy_dtype_8bit_signed_pcm(self):
        """Test converting 8-bit signed PCM format to NumPy dtype"""
        format = cm.AudioFormat(
            sample_rate=44100.0,
            format_id="lpcm",
            format_flags=0,  # Signed
            channels_per_frame=1,
            bits_per_channel=8,
        )

        dtype = format.to_numpy_dtype()
        assert dtype == np.dtype(np.int8)

    def test_to_numpy_dtype_8bit_unsigned_pcm(self):
        """Test converting 8-bit unsigned PCM format to NumPy dtype"""
        format = cm.AudioFormat(
            sample_rate=44100.0,
            format_id="lpcm",
            format_flags=2,  # kAudioFormatFlagIsSignedInteger = 2 (inverted)
            channels_per_frame=1,
            bits_per_channel=8,
        )

        dtype = format.to_numpy_dtype()
        assert dtype == np.dtype(np.uint8)

    def test_to_numpy_dtype_24bit_pcm(self):
        """Test converting 24-bit PCM format to NumPy dtype"""
        format = cm.AudioFormat(
            sample_rate=44100.0,
            format_id="lpcm",
            channels_per_frame=2,
            bits_per_channel=24,
        )

        dtype = format.to_numpy_dtype()
        assert dtype == np.dtype(np.int32)

    def test_to_numpy_dtype_32bit_int_pcm(self):
        """Test converting 32-bit integer PCM format to NumPy dtype"""
        format = cm.AudioFormat(
            sample_rate=44100.0,
            format_id="lpcm",
            format_flags=0,  # Not float
            channels_per_frame=2,
            bits_per_channel=32,
        )

        dtype = format.to_numpy_dtype()
        assert dtype == np.dtype(np.int32)

    def test_to_numpy_dtype_32bit_float_pcm(self):
        """Test converting 32-bit float PCM format to NumPy dtype"""
        format = cm.AudioFormat(
            sample_rate=44100.0,
            format_id="lpcm",
            format_flags=1,  # kAudioFormatFlagIsFloat = 1
            channels_per_frame=2,
            bits_per_channel=32,
        )

        dtype = format.to_numpy_dtype()
        assert dtype == np.dtype(np.float32)

    def test_to_numpy_dtype_64bit_float_pcm(self):
        """Test converting 64-bit float PCM format to NumPy dtype"""
        format = cm.AudioFormat(
            sample_rate=44100.0,
            format_id="lpcm",
            format_flags=1,  # kAudioFormatFlagIsFloat = 1
            channels_per_frame=2,
            bits_per_channel=64,
        )

        dtype = format.to_numpy_dtype()
        assert dtype == np.dtype(np.float64)

    def test_to_numpy_dtype_non_pcm_raises(self):
        """Test that non-PCM formats raise ValueError"""
        format = cm.AudioFormat(
            sample_rate=44100.0,
            format_id="aac ",  # AAC format
            channels_per_frame=2,
            bits_per_channel=0,
        )

        with pytest.raises(ValueError, match="Cannot convert non-PCM format"):
            format.to_numpy_dtype()

    def test_to_numpy_dtype_unsupported_float_depth_raises(self):
        """Test that unsupported float bit depths raise ValueError"""
        format = cm.AudioFormat(
            sample_rate=44100.0,
            format_id="lpcm",
            format_flags=1,  # kAudioFormatFlagIsFloat = 1
            channels_per_frame=2,
            bits_per_channel=16,  # Invalid float depth
        )

        with pytest.raises(ValueError, match="Unsupported float bit depth"):
            format.to_numpy_dtype()

    def test_to_numpy_dtype_unsupported_int_depth_raises(self):
        """Test that unsupported integer bit depths raise ValueError"""
        format = cm.AudioFormat(
            sample_rate=44100.0,
            format_id="lpcm",
            format_flags=0,
            channels_per_frame=2,
            bits_per_channel=48,  # Invalid int depth
        )

        with pytest.raises(ValueError, match="Unsupported integer bit depth"):
            format.to_numpy_dtype()


class TestAudioFileNumPy:
    """Tests for AudioFile.read_as_numpy()"""

    @pytest.fixture
    def test_audio_file(self, amen_wav_path):
        """Fixture providing path to test audio file"""
        return amen_wav_path

    def test_read_as_numpy_full_file(self, test_audio_file):
        """Test reading entire audio file as NumPy array"""
        with cm.AudioFile(test_audio_file) as audio:
            data = audio.read_as_numpy()

            # Check it's a NumPy array
            assert isinstance(data, np.ndarray)

            # Check dimensions (should be (frames, channels) for stereo)
            assert data.ndim == 2
            assert data.shape[1] == 2  # Stereo

            # Check dtype (16-bit PCM)
            assert data.dtype == np.int16

            # Check we got reasonable amount of data
            # File is ~2.74 seconds at 44.1kHz = ~120,000 frames
            assert data.shape[0] > 100000
            assert data.shape[0] < 150000

    def test_read_as_numpy_partial_file(self, test_audio_file):
        """Test reading partial audio file as NumPy array"""
        with cm.AudioFile(test_audio_file) as audio:
            # Read first 1000 packets
            data = audio.read_as_numpy(start_packet=0, packet_count=1000)

            # Check it's a NumPy array
            assert isinstance(data, np.ndarray)

            # Check dimensions
            assert data.ndim == 2
            assert data.shape[1] == 2  # Stereo

            # Check dtype
            assert data.dtype == np.int16

            # Should have exactly 1000 frames (for PCM, packet = frame)
            assert data.shape[0] == 1000

    def test_read_as_numpy_with_offset(self, test_audio_file):
        """Test reading audio from offset"""
        with cm.AudioFile(test_audio_file) as audio:
            # Read 500 packets starting from packet 1000
            data = audio.read_as_numpy(start_packet=1000, packet_count=500)

            # Check it's a NumPy array
            assert isinstance(data, np.ndarray)

            # Check dimensions
            assert data.ndim == 2
            assert data.shape[0] == 500
            assert data.shape[1] == 2

    def test_read_as_numpy_data_validity(self, test_audio_file):
        """Test that read NumPy data is valid audio data"""
        with cm.AudioFile(test_audio_file) as audio:
            data = audio.read_as_numpy(start_packet=0, packet_count=1000)

            # Audio data should be in valid range for int16
            assert data.min() >= -32768
            assert data.max() <= 32767

            # Audio data should have some variation (not all zeros)
            assert data.std() > 0

    def test_read_as_numpy_matches_read_packets(self, test_audio_file):
        """Test that read_as_numpy matches read_packets"""
        with cm.AudioFile(test_audio_file) as audio:
            # Read using both methods
            numpy_data = audio.read_as_numpy(start_packet=0, packet_count=100)
            bytes_data, count = audio.read_packets(start_packet=0, packet_count=100)

            # Convert bytes to NumPy for comparison
            bytes_as_numpy = np.frombuffer(bytes_data, dtype=np.int16)

            # Should contain the same raw data
            flattened_numpy = numpy_data.flatten()
            assert np.array_equal(
                flattened_numpy[: len(bytes_as_numpy)],
                bytes_as_numpy[: len(flattened_numpy)],
            )

    def test_read_as_numpy_format_consistency(self, test_audio_file):
        """Test that NumPy dtype matches AudioFormat"""
        with cm.AudioFile(test_audio_file) as audio:
            format = audio.format
            data = audio.read_as_numpy(start_packet=0, packet_count=100)

            # NumPy dtype should match format
            expected_dtype = format.to_numpy_dtype()
            assert data.dtype == expected_dtype

            # Channel count should match
            assert data.shape[1] == format.channels_per_frame

    def test_read_as_numpy_no_numpy_raises(self, test_audio_file, monkeypatch):
        """Test that read_as_numpy raises if NumPy not available"""
        # Temporarily mock NUMPY_AVAILABLE to False
        monkeypatch.setattr(cm.objects, "NUMPY_AVAILABLE", False)

        with cm.AudioFile(test_audio_file) as audio:
            with pytest.raises(ImportError, match="NumPy is not available"):
                audio.read_as_numpy()

    def test_read_as_numpy_sequential_reads(self, test_audio_file):
        """Test sequential reads produce correct data"""
        with cm.AudioFile(test_audio_file) as audio:
            # Read first chunk
            chunk1 = audio.read_as_numpy(start_packet=0, packet_count=500)

            # Read second chunk
            chunk2 = audio.read_as_numpy(start_packet=500, packet_count=500)

            # Read both together
            both = audio.read_as_numpy(start_packet=0, packet_count=1000)

            # Check chunks match combined read
            assert np.array_equal(chunk1, both[:500])
            assert np.array_equal(chunk2, both[500:1000])

    def test_read_as_numpy_channel_separation(self, test_audio_file):
        """Test that stereo channels are properly separated"""
        with cm.AudioFile(test_audio_file) as audio:
            data = audio.read_as_numpy(start_packet=0, packet_count=1000)

            # Extract left and right channels
            left_channel = data[:, 0]
            right_channel = data[:, 1]

            # Both channels should have data
            assert left_channel.std() > 0
            assert right_channel.std() > 0

            # Channels should be independent (not identical)
            # For real audio, channels are usually similar but not identical
            assert not np.array_equal(left_channel, right_channel)


class TestNumPyAvailabilityFlag:
    """Tests for NUMPY_AVAILABLE flag"""

    def test_numpy_available_flag_exists(self):
        """Test that NUMPY_AVAILABLE flag is defined"""
        assert hasattr(cm, "NUMPY_AVAILABLE")

    def test_numpy_available_flag_is_bool(self):
        """Test that NUMPY_AVAILABLE is a boolean"""
        assert isinstance(cm.NUMPY_AVAILABLE, bool)

    def test_numpy_available_when_installed(self):
        """Test that flag is True when NumPy is installed"""
        assert cm.NUMPY_AVAILABLE is True

    def test_numpy_import_from_module(self):
        """Test importing NUMPY_AVAILABLE from module"""
        from coremusic import NUMPY_AVAILABLE

        assert isinstance(NUMPY_AVAILABLE, bool)
