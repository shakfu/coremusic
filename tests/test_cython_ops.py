"""Tests for Cython-optimized audio operations."""

import pytest
import numpy as np

# Import from capi directly
try:
    from coremusic import capi as cython_ops
    CYTHON_OPS_AVAILABLE = True
except (ImportError, AttributeError):
    CYTHON_OPS_AVAILABLE = False


# Skip all tests if Cython ops not available
pytestmark = pytest.mark.skipif(not CYTHON_OPS_AVAILABLE, reason="Cython ops not available")


class TestNormalization:
    """Test audio normalization functions."""

    def test_normalize_audio_basic(self):
        """Test basic normalization."""
        # Create test data with known peak
        audio = np.array([[0.5, -0.5], [1.0, -1.0], [0.25, -0.25]], dtype=np.float32)
        normalized = cython_ops.normalize_audio(audio, target_peak=0.8)

        # Check peak is at target
        peak = np.max(np.abs(normalized))
        assert abs(peak - 0.8) < 0.001

    def test_normalize_audio_mono(self):
        """Test normalization with mono audio."""
        audio = np.array([0.5, 1.0, 0.25], dtype=np.float32)
        normalized = cython_ops.normalize_audio(audio, target_peak=0.9)

        peak = np.max(np.abs(normalized))
        assert abs(peak - 0.9) < 0.001

    def test_normalize_audio_preserves_shape(self):
        """Test that normalization preserves array shape."""
        audio = np.random.randn(1000, 2).astype(np.float32)
        normalized = cython_ops.normalize_audio(audio)

        assert normalized.shape == audio.shape

    def test_normalize_audio_zero_signal(self):
        """Test normalization with zero signal."""
        audio = np.zeros((100, 2), dtype=np.float32)
        normalized = cython_ops.normalize_audio(audio)

        # Should remain zero
        assert np.all(normalized == 0)


class TestGain:
    """Test gain application functions."""

    def test_apply_gain_positive(self):
        """Test applying positive gain."""
        audio = np.ones((100, 2), dtype=np.float32) * 0.5
        gained = cython_ops.apply_gain(audio, gain_db=6.0)

        # +6dB is approximately 2x linear gain
        expected_peak = 0.5 * (10 ** (6.0 / 20.0))
        assert abs(np.max(gained) - expected_peak) < 0.01

    def test_apply_gain_negative(self):
        """Test applying negative gain (attenuation)."""
        audio = np.ones((100, 2), dtype=np.float32)
        gained = cython_ops.apply_gain(audio, gain_db=-6.0)

        # -6dB is approximately 0.5x linear gain
        expected_peak = 1.0 * (10 ** (-6.0 / 20.0))
        assert abs(np.max(gained) - expected_peak) < 0.01

    def test_apply_gain_zero_db(self):
        """Test that 0dB gain doesn't change signal."""
        audio = np.random.randn(100, 2).astype(np.float32)
        gained = cython_ops.apply_gain(audio, gain_db=0.0)

        np.testing.assert_array_almost_equal(audio, gained, decimal=5)


class TestAnalysis:
    """Test signal analysis functions."""

    def test_calculate_rms(self):
        """Test RMS calculation."""
        # Create signal with known RMS
        audio = np.ones((1000, 2), dtype=np.float32) * 0.5
        rms = cython_ops.calculate_rms(audio)

        # RMS of constant 0.5 should be 0.5
        assert abs(rms - 0.5) < 0.001

    def test_calculate_peak(self):
        """Test peak calculation."""
        audio = np.array([[0.1, 0.2], [0.8, 0.3], [0.4, 0.5]], dtype=np.float32)
        peak = cython_ops.calculate_peak(audio)

        # Peak should be 0.8
        assert abs(peak - 0.8) < 0.001

    def test_calculate_rms_silent(self):
        """Test RMS of silent signal."""
        audio = np.zeros((100, 2), dtype=np.float32)
        rms = cython_ops.calculate_rms(audio)

        assert rms == 0.0

    def test_calculate_peak_silent(self):
        """Test peak of silent signal."""
        audio = np.zeros((100, 2), dtype=np.float32)
        peak = cython_ops.calculate_peak(audio)

        assert peak == 0.0


class TestConversions:
    """Test audio format conversion functions."""

    def test_float32_to_int16(self):
        """Test float32 to int16 conversion."""
        float_data = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32)
        int_data = np.zeros((2, 2), dtype=np.int16)

        cython_ops.convert_float32_to_int16(float_data, int_data)

        # 1.0 should convert to 32767, -1.0 to -32767 (clipped)
        assert int_data[0, 0] == 32767
        assert int_data[0, 1] in (-32768, -32767)  # May be clipped
        assert abs(int_data[1, 0] - 16383) < 10  # 0.5 * 32767

    def test_int16_to_float32(self):
        """Test int16 to float32 conversion."""
        int_data = np.array([[32767, -32768], [16384, -16384]], dtype=np.int16)
        float_data = np.zeros((2, 2), dtype=np.float32)

        cython_ops.convert_int16_to_float32(int_data, float_data)

        # Check conversion
        assert abs(float_data[0, 0] - 0.999969) < 0.001  # 32767/32768
        assert abs(float_data[0, 1] - (-1.0)) < 0.001
        assert abs(float_data[1, 0] - 0.5) < 0.01

    def test_stereo_to_mono(self):
        """Test stereo to mono conversion."""
        stereo = np.array([[0.5, 1.0], [0.2, 0.8]], dtype=np.float32)
        mono = np.zeros(2, dtype=np.float32)

        cython_ops.stereo_to_mono_float32(stereo, mono)

        # Should average the two channels
        assert abs(mono[0] - 0.75) < 0.001  # (0.5 + 1.0) / 2
        assert abs(mono[1] - 0.5) < 0.001   # (0.2 + 0.8) / 2

    def test_mono_to_stereo(self):
        """Test mono to stereo conversion."""
        mono = np.array([0.5, 0.8], dtype=np.float32)
        stereo = np.zeros((2, 2), dtype=np.float32)

        cython_ops.mono_to_stereo_float32(mono, stereo)

        # Both channels should be the same as mono
        assert stereo[0, 0] == 0.5
        assert stereo[0, 1] == 0.5
        assert stereo[1, 0] == 0.8
        assert stereo[1, 1] == 0.8


class TestMixing:
    """Test audio mixing functions."""

    def test_mix_audio_equal(self):
        """Test mixing two signals equally."""
        input1 = np.ones((100, 2), dtype=np.float32) * 0.5
        input2 = np.ones((100, 2), dtype=np.float32) * 1.0
        output = np.zeros((100, 2), dtype=np.float32)

        cython_ops.mix_audio_float32(output, input1, input2, mix_ratio=0.5)

        # 50% mix should be average
        expected = (0.5 + 1.0) / 2
        assert abs(output[0, 0] - expected) < 0.001

    def test_mix_audio_full_input1(self):
        """Test mixing with 100% input1."""
        input1 = np.ones((100, 2), dtype=np.float32) * 0.3
        input2 = np.ones((100, 2), dtype=np.float32) * 0.7
        output = np.zeros((100, 2), dtype=np.float32)

        cython_ops.mix_audio_float32(output, input1, input2, mix_ratio=0.0)

        # Should be all input1
        assert abs(output[0, 0] - 0.3) < 0.001

    def test_mix_audio_full_input2(self):
        """Test mixing with 100% input2."""
        input1 = np.ones((100, 2), dtype=np.float32) * 0.3
        input2 = np.ones((100, 2), dtype=np.float32) * 0.7
        output = np.zeros((100, 2), dtype=np.float32)

        cython_ops.mix_audio_float32(output, input1, input2, mix_ratio=1.0)

        # Should be all input2
        assert abs(output[0, 0] - 0.7) < 0.001


class TestFades:
    """Test fade in/out functions."""

    def test_fade_in(self):
        """Test fade-in."""
        audio = np.ones((100, 2), dtype=np.float32)
        cython_ops.apply_fade_in_float32(audio, fade_frames=50)

        # First sample should be silent
        assert audio[0, 0] == 0.0

        # Middle should be half volume
        assert abs(audio[25, 0] - 0.5) < 0.1

        # After fade should be full volume
        assert audio[60, 0] == 1.0

    def test_fade_out(self):
        """Test fade-out."""
        audio = np.ones((100, 2), dtype=np.float32)
        cython_ops.apply_fade_out_float32(audio, fade_frames=50)

        # Before fade should be full volume
        assert audio[40, 0] == 1.0

        # Middle of fade should be half volume
        assert abs(audio[75, 0] - 0.5) < 0.1

        # Last sample should be near silent (allows for rounding)
        assert audio[99, 0] < 0.03

    def test_fade_longer_than_audio(self):
        """Test fade longer than audio length."""
        audio = np.ones((100, 2), dtype=np.float32)

        # Fade longer than audio should not crash
        cython_ops.apply_fade_in_float32(audio, fade_frames=200)

        # First should be silent, last should be near full
        assert audio[0, 0] == 0.0
        assert audio[99, 0] > 0.9


class TestPerformance:
    """Performance tests for Cython ops."""

    def test_large_array_performance(self):
        """Test that large arrays are processed efficiently."""
        import time

        # Create large audio buffer (10 seconds at 44.1kHz stereo)
        audio = np.random.randn(441000, 2).astype(np.float32)

        # Time normalization
        start = time.time()
        normalized = cython_ops.normalize_audio(audio)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 100ms)
        assert elapsed < 0.1
        assert normalized.shape == audio.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
