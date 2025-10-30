"""Tests for buffer management utilities."""

import pytest
import numpy as np
import coremusic as cm
from coremusic.buffer_utils import (
    AudioStreamBasicDescription,
    fourcc_to_int,
    int_to_fourcc,
    pack_audio_buffer,
    unpack_audio_buffer,
    convert_buffer_format,
    calculate_buffer_size,
    optimal_buffer_size,
)


class TestFourCCConversion:
    """Test FourCC conversion utilities."""

    def test_fourcc_to_int(self):
        """Test FourCC string to integer conversion."""
        result = fourcc_to_int('lpcm')
        assert isinstance(result, int)
        assert result == 0x6C70636D

    def test_fourcc_to_int_invalid_length(self):
        """Test FourCC conversion rejects invalid length."""
        with pytest.raises(ValueError):
            fourcc_to_int('abc')  # Too short

        with pytest.raises(ValueError):
            fourcc_to_int('abcde')  # Too long

    def test_int_to_fourcc(self):
        """Test integer to FourCC string conversion."""
        result = int_to_fourcc(0x6C70636D)
        assert result == 'lpcm'

    def test_fourcc_roundtrip(self):
        """Test FourCC conversion roundtrip."""
        original = 'WAVE'
        converted = fourcc_to_int(original)
        back = int_to_fourcc(converted)
        assert back == original


class TestAudioStreamBasicDescription:
    """Test AudioStreamBasicDescription dataclass."""

    def test_create_from_params(self):
        """Test creating ASBD from parameters."""
        asbd = AudioStreamBasicDescription(
            sample_rate=44100.0,
            format_id='lpcm',
            format_flags=0x09,  # float | packed
            bytes_per_packet=8,
            frames_per_packet=1,
            bytes_per_frame=8,
            channels_per_frame=2,
            bits_per_channel=32
        )

        assert asbd.sample_rate == 44100.0
        assert asbd.channels_per_frame == 2
        assert asbd.bits_per_channel == 32

    def test_create_invalid_sample_rate(self):
        """Test validation of invalid sample rate."""
        with pytest.raises(ValueError):
            AudioStreamBasicDescription(
                sample_rate=0,  # Invalid
                format_id='lpcm',
                format_flags=0,
                bytes_per_packet=8,
                frames_per_packet=1,
                bytes_per_frame=8,
                channels_per_frame=2,
                bits_per_channel=32
            )

    def test_create_invalid_channels(self):
        """Test validation of invalid channel count."""
        with pytest.raises(ValueError):
            AudioStreamBasicDescription(
                sample_rate=44100.0,
                format_id='lpcm',
                format_flags=0,
                bytes_per_packet=8,
                frames_per_packet=1,
                bytes_per_frame=8,
                channels_per_frame=0,  # Invalid
                bits_per_channel=32
            )

    def test_is_pcm_property(self):
        """Test is_pcm property."""
        asbd = AudioStreamBasicDescription.pcm_float32_stereo()
        assert asbd.is_pcm is True

    def test_is_float_property(self):
        """Test is_float property."""
        asbd = AudioStreamBasicDescription.pcm_float32_stereo()
        assert asbd.is_float is True

        asbd_int = AudioStreamBasicDescription.pcm_int16_stereo()
        assert asbd_int.is_float is False

    def test_is_signed_integer_property(self):
        """Test is_signed_integer property."""
        asbd = AudioStreamBasicDescription.pcm_int16_stereo()
        assert asbd.is_signed_integer is True

        asbd_float = AudioStreamBasicDescription.pcm_float32_stereo()
        assert asbd_float.is_signed_integer is False

    def test_is_packed_property(self):
        """Test is_packed property."""
        asbd = AudioStreamBasicDescription.pcm_float32_stereo()
        assert asbd.is_packed is True

    def test_is_interleaved_property(self):
        """Test is_interleaved property."""
        asbd = AudioStreamBasicDescription.pcm_float32_stereo()
        # Default stereo format is interleaved
        assert asbd.is_interleaved is True

    def test_bytes_for_frames(self):
        """Test calculating bytes from frames."""
        asbd = AudioStreamBasicDescription.pcm_float32_stereo(44100.0)
        # 2 channels * 4 bytes = 8 bytes per frame
        assert asbd.bytes_for_frames(1024) == 1024 * 8

    def test_frames_for_bytes(self):
        """Test calculating frames from bytes."""
        asbd = AudioStreamBasicDescription.pcm_float32_stereo(44100.0)
        # 8 bytes per frame
        assert asbd.frames_for_bytes(8192) == 1024

    def test_to_dict(self):
        """Test conversion to dictionary."""
        asbd = AudioStreamBasicDescription.pcm_float32_stereo(44100.0)
        d = asbd.to_dict()

        assert d['sample_rate'] == 44100.0
        assert d['channels_per_frame'] == 2
        assert d['bits_per_channel'] == 32
        assert isinstance(d['format_id'], int)

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'sample_rate': 48000.0,
            'format_id': 'lpcm',
            'format_flags': 0x09,
            'bytes_per_packet': 8,
            'frames_per_packet': 1,
            'bytes_per_frame': 8,
            'channels_per_frame': 2,
            'bits_per_channel': 32,
        }

        asbd = AudioStreamBasicDescription.from_dict(data)
        assert asbd.sample_rate == 48000.0
        assert asbd.channels_per_frame == 2

    def test_pcm_float32_stereo_factory(self):
        """Test pcm_float32_stereo factory method."""
        asbd = AudioStreamBasicDescription.pcm_float32_stereo(48000.0)

        assert asbd.sample_rate == 48000.0
        assert asbd.channels_per_frame == 2
        assert asbd.bits_per_channel == 32
        assert asbd.is_float is True

    def test_pcm_int16_stereo_factory(self):
        """Test pcm_int16_stereo factory method."""
        asbd = AudioStreamBasicDescription.pcm_int16_stereo(44100.0)

        assert asbd.sample_rate == 44100.0
        assert asbd.channels_per_frame == 2
        assert asbd.bits_per_channel == 16
        assert asbd.is_signed_integer is True

    def test_str_representation(self):
        """Test string representation."""
        asbd = AudioStreamBasicDescription.pcm_float32_stereo(44100.0)
        s = str(asbd)

        assert "44100" in s
        assert "2ch" in s
        assert "32-bit" in s
        assert "float" in s


class TestBufferPacking:
    """Test buffer packing/unpacking utilities."""

    def test_pack_audio_buffer_exact_size(self):
        """Test packing buffer with exact size."""
        format = AudioStreamBasicDescription.pcm_float32_stereo(44100.0)
        num_frames = 1024
        expected_bytes = format.bytes_for_frames(num_frames)
        data = b'\x00' * expected_bytes

        packed, actual_frames = pack_audio_buffer(data, format, num_frames)

        assert len(packed) == expected_bytes
        assert actual_frames == num_frames

    def test_pack_audio_buffer_undersized(self):
        """Test packing buffer that's too small (pads with zeros)."""
        format = AudioStreamBasicDescription.pcm_float32_stereo(44100.0)
        num_frames = 1024
        expected_bytes = format.bytes_for_frames(num_frames)
        data = b'\x00' * (expected_bytes // 2)  # Half size

        packed, actual_frames = pack_audio_buffer(data, format, num_frames)

        assert len(packed) == expected_bytes
        # Should be padded
        assert actual_frames < num_frames

    def test_pack_audio_buffer_oversized(self):
        """Test packing buffer that's too large (truncates)."""
        format = AudioStreamBasicDescription.pcm_float32_stereo(44100.0)
        num_frames = 1024
        expected_bytes = format.bytes_for_frames(num_frames)
        data = b'\x00' * (expected_bytes * 2)  # Double size

        packed, actual_frames = pack_audio_buffer(data, format, num_frames)

        assert len(packed) == expected_bytes
        assert actual_frames == num_frames

    def test_unpack_audio_buffer(self):
        """Test unpacking audio buffer."""
        format = AudioStreamBasicDescription.pcm_float32_stereo(44100.0)
        num_frames = 1024
        data = b'\x00' * format.bytes_for_frames(num_frames)

        unpacked, frames = unpack_audio_buffer(data, format)

        assert len(unpacked) == len(data)
        assert frames == num_frames


class TestBufferConversion:
    """Test buffer format conversion."""

    def test_convert_float32_to_int16(self):
        """Test converting float32 to int16."""
        from_format = AudioStreamBasicDescription.pcm_float32_stereo(44100.0)
        to_format = AudioStreamBasicDescription.pcm_int16_stereo(44100.0)

        # Create float32 data
        samples = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
        data = samples.tobytes()

        converted = convert_buffer_format(data, from_format, to_format)

        # Check converted data
        result = np.frombuffer(converted, dtype=np.int16)
        assert len(result) == len(samples)
        # Check approximate conversion
        assert abs(result[0] - 16383) < 100  # 0.5 * 32767
        assert abs(result[1] + 16383) < 100  # -0.5 * 32767

    def test_convert_int16_to_float32(self):
        """Test converting int16 to float32."""
        from_format = AudioStreamBasicDescription.pcm_int16_stereo(44100.0)
        to_format = AudioStreamBasicDescription.pcm_float32_stereo(44100.0)

        # Create int16 data
        samples = np.array([16384, -16384, 8192, -8192], dtype=np.int16)
        data = samples.tobytes()

        converted = convert_buffer_format(data, from_format, to_format)

        # Check converted data
        result = np.frombuffer(converted, dtype=np.float32)
        assert len(result) == len(samples)
        # Check approximate conversion
        assert abs(result[0] - 0.5) < 0.01  # 16384 / 32768
        assert abs(result[1] + 0.5) < 0.01  # -16384 / 32768

    def test_convert_mono_to_stereo(self):
        """Test converting mono to stereo (duplication)."""
        from_format = AudioStreamBasicDescription(
            sample_rate=44100.0,
            format_id='lpcm',
            format_flags=0x09,
            bytes_per_packet=4,
            frames_per_packet=1,
            bytes_per_frame=4,
            channels_per_frame=1,  # Mono
            bits_per_channel=32
        )
        to_format = AudioStreamBasicDescription.pcm_float32_stereo(44100.0)

        # Create mono data
        samples = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        data = samples.tobytes()

        converted = convert_buffer_format(data, from_format, to_format)

        # Check converted data (should be duplicated)
        result = np.frombuffer(converted, dtype=np.float32)
        assert len(result) == len(samples) * 2  # Doubled for stereo
        # Check duplication
        assert result[0] == result[1]  # Left == Right
        assert result[2] == result[3]

    def test_convert_stereo_to_mono(self):
        """Test converting stereo to mono (averaging)."""
        from_format = AudioStreamBasicDescription.pcm_float32_stereo(44100.0)
        to_format = AudioStreamBasicDescription(
            sample_rate=44100.0,
            format_id='lpcm',
            format_flags=0x09,
            bytes_per_packet=4,
            frames_per_packet=1,
            bytes_per_frame=4,
            channels_per_frame=1,  # Mono
            bits_per_channel=32
        )

        # Create stereo data (L, R, L, R)
        samples = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        data = samples.tobytes()

        converted = convert_buffer_format(data, from_format, to_format)

        # Check converted data (should be averaged)
        result = np.frombuffer(converted, dtype=np.float32)
        assert len(result) == len(samples) // 2  # Halved for mono
        # Check averaging
        assert result[0] == 1.5  # (1.0 + 2.0) / 2
        assert result[1] == 3.5  # (3.0 + 4.0) / 2


class TestBufferCalculations:
    """Test buffer size calculation utilities."""

    def test_calculate_buffer_size(self):
        """Test calculating buffer size for duration."""
        format = AudioStreamBasicDescription.pcm_float32_stereo(44100.0)

        # Calculate for 1 second
        size = calculate_buffer_size(1.0, format)
        expected = 44100 * 8  # 44100 frames * 8 bytes per frame
        assert size == expected

    def test_calculate_buffer_size_fractional(self):
        """Test calculating buffer size for fractional duration."""
        format = AudioStreamBasicDescription.pcm_float32_stereo(44100.0)

        # Calculate for 0.1 seconds
        size = calculate_buffer_size(0.1, format)
        expected = 4410 * 8  # 4410 frames * 8 bytes per frame
        assert size == expected

    def test_optimal_buffer_size_default(self):
        """Test optimal buffer size with default latency."""
        format = AudioStreamBasicDescription.pcm_float32_stereo(44100.0)

        bytes_size, frames = optimal_buffer_size(format)

        # Check it's a power of 2
        import math
        assert frames == 2 ** math.ceil(math.log2(frames))

        # Check bytes match frames
        assert bytes_size == format.bytes_for_frames(frames)

    def test_optimal_buffer_size_custom_latency(self):
        """Test optimal buffer size with custom latency."""
        format = AudioStreamBasicDescription.pcm_float32_stereo(44100.0)

        # 5ms latency
        bytes_size, frames = optimal_buffer_size(format, latency_ms=5.0)

        # Should be around 220 frames (44100 * 0.005), rounded to power of 2
        assert frames == 256  # Next power of 2 above 220

        # Check bytes
        expected_bytes = 256 * 8  # 256 frames * 8 bytes per frame
        assert bytes_size == expected_bytes


class TestASBDIntegration:
    """Test ASBD integration with CoreMusic."""

    def test_asbd_compatible_with_audioformat(self):
        """Test ASBD can be converted to AudioFormat dict."""
        asbd = AudioStreamBasicDescription.pcm_float32_stereo(44100.0)
        format_dict = asbd.to_dict()

        # Should have all required fields for AudioFormat
        assert 'sample_rate' in format_dict
        assert 'format_id' in format_dict
        assert 'channels_per_frame' in format_dict

    def test_asbd_roundtrip_with_dict(self):
        """Test ASBD can roundtrip through dict."""
        original = AudioStreamBasicDescription.pcm_float32_stereo(48000.0)
        d = original.to_dict()
        restored = AudioStreamBasicDescription.from_dict(d)

        assert restored.sample_rate == original.sample_rate
        assert restored.channels_per_frame == original.channels_per_frame
        assert restored.bits_per_channel == original.bits_per_channel


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
