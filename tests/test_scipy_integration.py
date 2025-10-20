#!/usr/bin/env python3
"""Tests for SciPy signal processing integration."""

import os
import pytest
import coremusic as cm


class TestScipyAvailability:
    """Test SciPy availability detection"""

    def test_scipy_available_flag_exists(self):
        """Test that SCIPY_AVAILABLE flag is defined"""
        assert hasattr(cm, 'SCIPY_AVAILABLE')
        assert isinstance(cm.SCIPY_AVAILABLE, bool)

    @pytest.mark.skipif(not cm.SCIPY_AVAILABLE, reason="SciPy not available")
    def test_scipy_functions_available(self):
        """Test that SciPy functions are available when SciPy is installed"""
        assert cm.design_butterworth_filter is not None
        assert cm.apply_lowpass_filter is not None
        assert cm.resample_audio is not None
        assert cm.compute_spectrum is not None
        assert cm.AudioSignalProcessor is not None


@pytest.mark.skipif(not cm.SCIPY_AVAILABLE or not cm.NUMPY_AVAILABLE,
                   reason="SciPy and NumPy required")
class TestFilterDesign:
    """Test filter design functions"""

    def test_design_butterworth_lowpass(self):
        """Test Butterworth lowpass filter design"""
        b, a = cm.design_butterworth_filter(
            cutoff=1000,
            sample_rate=44100,
            order=5,
            filter_type='lowpass'
        )

        assert b is not None
        assert a is not None
        assert len(b) == 6  # order + 1
        assert len(a) == 6

    def test_design_butterworth_highpass(self):
        """Test Butterworth highpass filter design"""
        b, a = cm.design_butterworth_filter(
            cutoff=100,
            sample_rate=44100,
            order=4,
            filter_type='highpass'
        )

        assert b is not None
        assert a is not None
        assert len(b) == 5  # order + 1
        assert len(a) == 5

    def test_design_butterworth_bandpass(self):
        """Test Butterworth bandpass filter design"""
        b, a = cm.design_butterworth_filter(
            cutoff=(300, 3000),
            sample_rate=44100,
            order=3,
            filter_type='bandpass'
        )

        assert b is not None
        assert a is not None
        assert len(b) == 7  # 2*order + 1 for bandpass
        assert len(a) == 7

    def test_design_chebyshev_filter(self):
        """Test Chebyshev filter design"""
        b, a = cm.design_chebyshev_filter(
            cutoff=1000,
            sample_rate=44100,
            order=5,
            ripple_db=0.5,
            filter_type='lowpass'
        )

        assert b is not None
        assert a is not None
        assert len(b) == 6
        assert len(a) == 6


@pytest.mark.skipif(not cm.SCIPY_AVAILABLE or not cm.NUMPY_AVAILABLE,
                   reason="SciPy and NumPy required")
class TestFilterApplication:
    """Test filter application functions"""

    @pytest.fixture
    def test_signal_mono(self):
        """Generate test mono signal"""
        import numpy as np
        sample_rate = 44100
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Signal with 440Hz and 2000Hz components
        signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t)
        return signal, sample_rate

    @pytest.fixture
    def test_signal_stereo(self):
        """Generate test stereo signal"""
        import numpy as np
        sample_rate = 44100
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration))
        left = np.sin(2 * np.pi * 440 * t)
        right = np.sin(2 * np.pi * 880 * t)
        signal = np.column_stack([left, right])
        return signal, sample_rate

    def test_apply_lowpass_filter_mono(self, test_signal_mono):
        """Test lowpass filter on mono signal"""
        signal, sample_rate = test_signal_mono

        # Apply lowpass filter at 1000Hz (should remove 2000Hz component)
        filtered = cm.apply_lowpass_filter(signal, cutoff=1000, sample_rate=sample_rate)

        assert filtered is not None
        assert filtered.shape == signal.shape
        assert filtered.dtype == signal.dtype

        # Verify attenuation of high frequencies
        import numpy as np
        assert np.max(np.abs(filtered)) < np.max(np.abs(signal))

    def test_apply_highpass_filter_mono(self, test_signal_mono):
        """Test highpass filter on mono signal"""
        signal, sample_rate = test_signal_mono

        # Apply highpass filter at 1500Hz (should remove 440Hz component)
        filtered = cm.apply_highpass_filter(signal, cutoff=1500, sample_rate=sample_rate)

        assert filtered is not None
        assert filtered.shape == signal.shape

    def test_apply_bandpass_filter_mono(self, test_signal_mono):
        """Test bandpass filter on mono signal"""
        signal, sample_rate = test_signal_mono

        # Apply bandpass filter 300-500Hz (should attenuate both components)
        filtered = cm.apply_bandpass_filter(
            signal,
            lowcut=300,
            highcut=500,
            sample_rate=sample_rate
        )

        assert filtered is not None
        assert filtered.shape == signal.shape

    def test_apply_filter_stereo(self, test_signal_stereo):
        """Test filter application on stereo signal"""
        signal, sample_rate = test_signal_stereo

        # Design and apply filter
        b, a = cm.design_butterworth_filter(1000, sample_rate, order=5)
        filtered = cm.apply_filter(signal, b, a, zero_phase=True)

        assert filtered is not None
        assert filtered.shape == signal.shape
        assert filtered.ndim == 2  # Stereo

    def test_apply_lowpass_filter_stereo(self, test_signal_stereo):
        """Test lowpass filter on stereo signal"""
        signal, sample_rate = test_signal_stereo

        filtered = cm.apply_lowpass_filter(signal, cutoff=1000, sample_rate=sample_rate)

        assert filtered is not None
        assert filtered.shape == signal.shape
        assert filtered.ndim == 2


@pytest.mark.skipif(not cm.SCIPY_AVAILABLE or not cm.NUMPY_AVAILABLE,
                   reason="SciPy and NumPy required")
class TestResampling:
    """Test resampling functions"""

    @pytest.fixture
    def test_audio(self):
        """Generate test audio"""
        import numpy as np
        sample_rate = 44100
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        return audio, sample_rate

    def test_resample_audio_upsample(self, test_audio):
        """Test upsampling (44.1kHz -> 48kHz)"""
        audio, original_rate = test_audio

        resampled = cm.resample_audio(audio, original_rate=original_rate,
                                     target_rate=48000, method='fft')

        assert resampled is not None
        # Check length ratio
        expected_len = int(len(audio) * (48000 / original_rate))
        assert abs(len(resampled) - expected_len) <= 1  # Allow 1 sample difference

    def test_resample_audio_downsample(self, test_audio):
        """Test downsampling (44.1kHz -> 22.05kHz)"""
        audio, original_rate = test_audio

        resampled = cm.resample_audio(audio, original_rate=original_rate,
                                     target_rate=22050, method='fft')

        assert resampled is not None
        expected_len = int(len(audio) * (22050 / original_rate))
        assert abs(len(resampled) - expected_len) <= 1

    def test_resample_audio_polyphase_method(self, test_audio):
        """Test polyphase resampling method"""
        audio, original_rate = test_audio

        resampled = cm.resample_audio(audio, original_rate=original_rate,
                                     target_rate=48000, method='polyphase')

        assert resampled is not None
        assert len(resampled) > 0

    def test_resample_audio_same_rate(self, test_audio):
        """Test resampling with same rate (should return copy)"""
        audio, original_rate = test_audio

        resampled = cm.resample_audio(audio, original_rate=original_rate,
                                     target_rate=original_rate)

        import numpy as np
        assert np.array_equal(audio, resampled)

    def test_resample_audio_stereo(self):
        """Test resampling stereo audio"""
        import numpy as np
        sample_rate = 44100
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration))
        left = np.sin(2 * np.pi * 440 * t)
        right = np.sin(2 * np.pi * 880 * t)
        stereo = np.column_stack([left, right])

        resampled = cm.resample_audio(stereo, original_rate=sample_rate,
                                     target_rate=48000)

        assert resampled is not None
        assert resampled.ndim == 2
        assert resampled.shape[1] == 2  # Still stereo


@pytest.mark.skipif(not cm.SCIPY_AVAILABLE or not cm.NUMPY_AVAILABLE,
                   reason="SciPy and NumPy required")
class TestSpectralAnalysis:
    """Test spectral analysis functions"""

    @pytest.fixture
    def test_signal(self):
        """Generate test signal with known frequency"""
        import numpy as np
        sample_rate = 44100
        duration = 0.5
        freq = 440  # A4 note
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = np.sin(2 * np.pi * freq * t)
        return signal, sample_rate, freq

    def test_compute_spectrum(self, test_signal):
        """Test spectrum computation"""
        signal, sample_rate, freq = test_signal

        frequencies, spectrum = cm.compute_spectrum(signal, sample_rate)

        assert frequencies is not None
        assert spectrum is not None
        assert len(frequencies) == len(spectrum)
        assert frequencies[0] >= 0
        assert frequencies[-1] <= sample_rate / 2

    def test_compute_fft(self, test_signal):
        """Test FFT computation"""
        signal, sample_rate, freq = test_signal

        frequencies, magnitudes = cm.compute_fft(signal, sample_rate)

        assert frequencies is not None
        assert magnitudes is not None
        assert len(frequencies) == len(magnitudes)

        # Find peak frequency
        import numpy as np
        peak_idx = np.argmax(magnitudes)
        peak_freq = frequencies[peak_idx]

        # Should be close to 440Hz
        assert abs(peak_freq - freq) < 5  # Within 5Hz

    def test_compute_fft_with_window(self, test_signal):
        """Test FFT with different windows"""
        signal, sample_rate, _ = test_signal

        for window in ['hann', 'hamming', 'blackman']:
            frequencies, magnitudes = cm.compute_fft(signal, sample_rate, window=window)
            assert frequencies is not None
            assert magnitudes is not None

    def test_compute_spectrogram(self, test_signal):
        """Test spectrogram computation"""
        signal, sample_rate, _ = test_signal

        frequencies, times, spectrogram = cm.compute_spectrogram(signal, sample_rate)

        assert frequencies is not None
        assert times is not None
        assert spectrogram is not None
        assert spectrogram.shape[0] == len(frequencies)
        assert spectrogram.shape[1] == len(times)


@pytest.mark.skipif(not cm.SCIPY_AVAILABLE or not cm.NUMPY_AVAILABLE,
                   reason="SciPy and NumPy required")
class TestAudioSignalProcessor:
    """Test AudioSignalProcessor class"""

    @pytest.fixture
    def test_audio(self):
        """Generate test audio"""
        import numpy as np
        sample_rate = 44100
        duration = 0.2
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t)
        return audio, sample_rate

    def test_processor_creation(self, test_audio):
        """Test AudioSignalProcessor creation"""
        audio, sample_rate = test_audio

        processor = cm.AudioSignalProcessor(audio, sample_rate)

        assert processor is not None
        assert processor.get_sample_rate() == sample_rate

    def test_processor_lowpass(self, test_audio):
        """Test lowpass filter method"""
        audio, sample_rate = test_audio

        processor = cm.AudioSignalProcessor(audio, sample_rate)
        processor.lowpass(1000)

        processed = processor.get_audio()
        assert processed is not None
        assert processed.shape == audio.shape

    def test_processor_highpass(self, test_audio):
        """Test highpass filter method"""
        audio, sample_rate = test_audio

        processor = cm.AudioSignalProcessor(audio, sample_rate)
        processor.highpass(500)

        processed = processor.get_audio()
        assert processed is not None

    def test_processor_bandpass(self, test_audio):
        """Test bandpass filter method"""
        audio, sample_rate = test_audio

        processor = cm.AudioSignalProcessor(audio, sample_rate)
        processor.bandpass(300, 3000)

        processed = processor.get_audio()
        assert processed is not None

    def test_processor_resample(self, test_audio):
        """Test resample method"""
        audio, sample_rate = test_audio

        processor = cm.AudioSignalProcessor(audio, sample_rate)
        processor.resample(48000)

        assert processor.get_sample_rate() == 48000
        processed = processor.get_audio()
        assert len(processed) != len(audio)  # Different length after resampling

    def test_processor_normalize(self, test_audio):
        """Test normalize method"""
        audio, sample_rate = test_audio

        processor = cm.AudioSignalProcessor(audio, sample_rate)
        processor.normalize(target_level=0.5)

        import numpy as np
        processed = processor.get_audio()
        assert np.max(np.abs(processed)) <= 0.51  # Allow small tolerance

    def test_processor_method_chaining(self, test_audio):
        """Test method chaining"""
        audio, sample_rate = test_audio

        processor = cm.AudioSignalProcessor(audio, sample_rate)

        # Chain multiple operations
        processed = (processor
                    .highpass(100)
                    .lowpass(5000)
                    .normalize(0.8)
                    .get_audio())

        assert processed is not None
        assert processed.shape == audio.shape

    def test_processor_reset(self, test_audio):
        """Test reset method"""
        audio, sample_rate = test_audio

        processor = cm.AudioSignalProcessor(audio, sample_rate)
        processor.lowpass(1000)

        # Reset to original
        processor.reset()

        import numpy as np
        assert np.array_equal(processor.get_audio(), audio)

    def test_processor_spectrum(self, test_audio):
        """Test spectrum method"""
        audio, sample_rate = test_audio

        processor = cm.AudioSignalProcessor(audio, sample_rate)
        frequencies, spectrum = processor.spectrum()

        assert frequencies is not None
        assert spectrum is not None

    def test_processor_fft(self, test_audio):
        """Test FFT method"""
        audio, sample_rate = test_audio

        processor = cm.AudioSignalProcessor(audio, sample_rate)
        frequencies, magnitudes = processor.fft()

        assert frequencies is not None
        assert magnitudes is not None

    def test_processor_spectrogram(self, test_audio):
        """Test spectrogram method"""
        audio, sample_rate = test_audio

        processor = cm.AudioSignalProcessor(audio, sample_rate)
        frequencies, times, spectrogram = processor.spectrogram()

        assert frequencies is not None
        assert times is not None
        assert spectrogram is not None


@pytest.mark.skipif(not cm.SCIPY_AVAILABLE or not cm.NUMPY_AVAILABLE,
                   reason="SciPy and NumPy required")
class TestIntegrationWithAudioFile:
    """Test integration with AudioFile class"""

    @pytest.fixture
    def amen_wav_path(self):
        """Fixture providing path to amen.wav test file"""
        path = os.path.join("tests", "amen.wav")
        if not os.path.exists(path):
            pytest.skip(f"Test audio file not found: {path}")
        return path

    def test_read_and_filter_real_audio(self, amen_wav_path):
        """Test reading real audio and applying filter"""
        with cm.AudioFile(amen_wav_path) as af:
            audio = af.read_as_numpy()
            sample_rate = af.format.sample_rate

            # Apply lowpass filter
            filtered = cm.apply_lowpass_filter(audio, cutoff=2000, sample_rate=sample_rate)

            assert filtered is not None
            assert filtered.shape == audio.shape

    def test_signal_processor_with_real_audio(self, amen_wav_path):
        """Test AudioSignalProcessor with real audio file"""
        with cm.AudioFile(amen_wav_path) as af:
            audio = af.read_as_numpy()
            sample_rate = af.format.sample_rate

            # Create processor and apply operations
            processor = cm.AudioSignalProcessor(audio, sample_rate)
            processed = (processor
                        .highpass(50)       # Remove rumble
                        .lowpass(15000)     # Remove ultrasonic
                        .normalize(0.9)     # Normalize
                        .get_audio())

            assert processed is not None
            assert processed.shape == audio.shape

    def test_resample_real_audio(self, amen_wav_path):
        """Test resampling real audio file"""
        with cm.AudioFile(amen_wav_path) as af:
            audio = af.read_as_numpy()
            sample_rate = af.format.sample_rate

            # Resample to 48kHz
            resampled = cm.resample_audio(audio, original_rate=sample_rate,
                                         target_rate=48000)

            assert resampled is not None
            # Check approximate length ratio
            import numpy as np
            expected_ratio = 48000 / sample_rate
            actual_ratio = len(resampled) / len(audio)
            assert abs(actual_ratio - expected_ratio) < 0.01  # Within 1%

    def test_spectrum_analysis_real_audio(self, amen_wav_path):
        """Test spectrum analysis on real audio"""
        with cm.AudioFile(amen_wav_path) as af:
            audio = af.read_as_numpy()
            sample_rate = af.format.sample_rate

            # Compute spectrum
            frequencies, spectrum = cm.compute_spectrum(audio, sample_rate)

            assert frequencies is not None
            assert spectrum is not None
            assert len(frequencies) > 0
            assert len(spectrum) > 0
