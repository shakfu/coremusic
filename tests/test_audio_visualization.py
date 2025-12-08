#!/usr/bin/env python3
"""Tests for audio visualization module."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from conftest import AMEN_WAV_PATH

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for testing
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

if NUMPY_AVAILABLE and MATPLOTLIB_AVAILABLE:
    from coremusic.audio.visualization import (
        WaveformPlotter,
        SpectrogramPlotter,
        FrequencySpectrumPlotter,
    )

pytestmark = pytest.mark.skipif(
    not (NUMPY_AVAILABLE and MATPLOTLIB_AVAILABLE),
    reason="NumPy and matplotlib required for visualization tests",
)


class TestWaveformPlotter:
    """Tests for WaveformPlotter class."""

    def test_create_plotter(self):
        """Test creating WaveformPlotter."""
        plotter = WaveformPlotter(AMEN_WAV_PATH)

        assert plotter.audio_file == Path(AMEN_WAV_PATH)
        assert plotter._audio_data is None
        assert plotter._sample_rate is None

    def test_load_audio(self):
        """Test audio loading."""
        plotter = WaveformPlotter(AMEN_WAV_PATH)
        data, sr = plotter._load_audio()

        assert isinstance(data, np.ndarray)
        assert len(data) > 0
        assert sr == 44100.0

        # Second call should return cached data
        data2, sr2 = plotter._load_audio()
        assert np.array_equal(data, data2)
        assert sr == sr2

    def test_plot_basic(self):
        """Test basic waveform plotting."""
        plotter = WaveformPlotter(AMEN_WAV_PATH)
        fig, ax = plotter.plot()

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_with_rms(self):
        """Test waveform plotting with RMS envelope."""
        plotter = WaveformPlotter(AMEN_WAV_PATH)
        fig, ax = plotter.plot(show_rms=True, rms_window=0.05)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_with_peaks(self):
        """Test waveform plotting with peak envelope."""
        plotter = WaveformPlotter(AMEN_WAV_PATH)
        fig, ax = plotter.plot(show_peaks=True, peak_window=0.03)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_with_rms_and_peaks(self):
        """Test waveform plotting with both envelopes."""
        plotter = WaveformPlotter(AMEN_WAV_PATH)
        fig, ax = plotter.plot(show_rms=True, show_peaks=True)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_time_range(self):
        """Test plotting specific time range."""
        plotter = WaveformPlotter(AMEN_WAV_PATH)
        fig, ax = plotter.plot(time_range=(0.5, 1.5))

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_custom_figsize(self):
        """Test plotting with custom figure size."""
        plotter = WaveformPlotter(AMEN_WAV_PATH)
        fig, ax = plotter.plot(figsize=(10, 3))

        assert fig is not None
        assert ax is not None
        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 3
        plt.close(fig)

    def test_plot_custom_title(self):
        """Test plotting with custom title."""
        plotter = WaveformPlotter(AMEN_WAV_PATH)
        fig, ax = plotter.plot(title="My Custom Waveform")

        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "My Custom Waveform"
        plt.close(fig)

    def test_calculate_rms_envelope(self):
        """Test RMS envelope calculation."""
        plotter = WaveformPlotter(AMEN_WAV_PATH)
        data, _ = plotter._load_audio()

        if data.ndim > 1:
            data = np.mean(data, axis=1)

        window_size = 2048
        rms = plotter._calculate_rms_envelope(data, window_size)

        assert isinstance(rms, np.ndarray)
        assert len(rms) == len(data)
        assert np.all(rms >= 0)
        # RMS should be less than or equal to max absolute value in data
        assert np.all(rms <= np.max(np.abs(data)))

    def test_calculate_peak_envelope(self):
        """Test peak envelope calculation."""
        plotter = WaveformPlotter(AMEN_WAV_PATH)
        data, _ = plotter._load_audio()

        if data.ndim > 1:
            data = np.mean(data, axis=1)

        window_size = 1024
        peak_pos, peak_neg = plotter._calculate_peak_envelope(data, window_size)

        assert isinstance(peak_pos, np.ndarray)
        assert isinstance(peak_neg, np.ndarray)
        assert len(peak_pos) == len(data)
        assert len(peak_neg) == len(data)
        assert np.all(peak_pos >= data)
        assert np.all(peak_neg <= data)

    def test_save(self, tmp_path):
        """Test saving waveform plot to file."""
        plotter = WaveformPlotter(AMEN_WAV_PATH)
        output_path = tmp_path / "waveform.png"

        plotter.save(str(output_path), show_rms=True, dpi=100)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestSpectrogramPlotter:
    """Tests for SpectrogramPlotter class."""

    def test_create_plotter(self):
        """Test creating SpectrogramPlotter."""
        plotter = SpectrogramPlotter(AMEN_WAV_PATH)

        assert plotter.audio_file == Path(AMEN_WAV_PATH)
        assert plotter._audio_data is None
        assert plotter._sample_rate is None

    def test_load_audio(self):
        """Test audio loading."""
        plotter = SpectrogramPlotter(AMEN_WAV_PATH)
        data, sr = plotter._load_audio()

        assert isinstance(data, np.ndarray)
        assert len(data) > 0
        assert sr == 44100.0

    def test_plot_basic(self):
        """Test basic spectrogram plotting."""
        plotter = SpectrogramPlotter(AMEN_WAV_PATH)
        fig, ax = plotter.plot()

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_custom_window_size(self):
        """Test spectrogram with custom window size."""
        plotter = SpectrogramPlotter(AMEN_WAV_PATH)
        fig, ax = plotter.plot(window_size=1024, hop_size=256)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_custom_colormap(self):
        """Test spectrogram with different colormaps."""
        plotter = SpectrogramPlotter(AMEN_WAV_PATH)

        for cmap in ["viridis", "magma", "plasma", "inferno"]:
            fig, ax = plotter.plot(cmap=cmap)
            assert fig is not None
            assert ax is not None
            plt.close(fig)

    def test_plot_db_range(self):
        """Test spectrogram with custom dB range."""
        plotter = SpectrogramPlotter(AMEN_WAV_PATH)
        fig, ax = plotter.plot(min_db=-60, max_db=0)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_custom_window_type(self):
        """Test spectrogram with different window functions."""
        plotter = SpectrogramPlotter(AMEN_WAV_PATH)

        for window in ["hann", "hamming", "blackman"]:
            fig, ax = plotter.plot(window=window)
            assert fig is not None
            assert ax is not None
            plt.close(fig)

    def test_plot_custom_title(self):
        """Test spectrogram with custom title."""
        plotter = SpectrogramPlotter(AMEN_WAV_PATH)
        fig, ax = plotter.plot(title="My Custom Spectrogram")

        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "My Custom Spectrogram"
        plt.close(fig)

    def test_compute_spectrogram(self):
        """Test spectrogram computation."""
        plotter = SpectrogramPlotter(AMEN_WAV_PATH)
        data, sr = plotter._load_audio()

        if data.ndim > 1:
            data = np.mean(data, axis=1)

        window_size = 2048
        hop_size = 512

        f, t, Sxx = plotter._compute_spectrogram(
            data, sr, window_size, hop_size, "hann"
        )

        assert isinstance(f, np.ndarray)
        assert isinstance(t, np.ndarray)
        assert isinstance(Sxx, np.ndarray)
        assert len(f) == window_size // 2 + 1
        assert len(t) > 0
        assert Sxx.shape == (len(f), len(t))

    def test_save(self, tmp_path):
        """Test saving spectrogram plot to file."""
        plotter = SpectrogramPlotter(AMEN_WAV_PATH)
        output_path = tmp_path / "spectrogram.png"

        plotter.save(str(output_path), dpi=100)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestFrequencySpectrumPlotter:
    """Tests for FrequencySpectrumPlotter class."""

    def test_create_plotter(self):
        """Test creating FrequencySpectrumPlotter."""
        plotter = FrequencySpectrumPlotter(AMEN_WAV_PATH)

        assert plotter.audio_file == Path(AMEN_WAV_PATH)
        assert plotter._audio_data is None
        assert plotter._sample_rate is None

    def test_load_audio(self):
        """Test audio loading."""
        plotter = FrequencySpectrumPlotter(AMEN_WAV_PATH)
        data, sr = plotter._load_audio()

        assert isinstance(data, np.ndarray)
        assert len(data) > 0
        assert sr == 44100.0

    def test_plot_basic(self):
        """Test basic frequency spectrum plotting."""
        plotter = FrequencySpectrumPlotter(AMEN_WAV_PATH)
        fig, ax = plotter.plot()

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_at_time(self):
        """Test plotting spectrum at specific time."""
        plotter = FrequencySpectrumPlotter(AMEN_WAV_PATH)
        fig, ax = plotter.plot(time=1.0)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_custom_window_size(self):
        """Test spectrum with custom window size."""
        plotter = FrequencySpectrumPlotter(AMEN_WAV_PATH)

        for window_size in [2048, 4096, 8192]:
            fig, ax = plotter.plot(window_size=window_size)
            assert fig is not None
            assert ax is not None
            plt.close(fig)

    def test_plot_frequency_range(self):
        """Test spectrum with limited frequency range."""
        plotter = FrequencySpectrumPlotter(AMEN_WAV_PATH)
        fig, ax = plotter.plot(min_freq=100, max_freq=10000)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_custom_window_type(self):
        """Test spectrum with different window functions."""
        plotter = FrequencySpectrumPlotter(AMEN_WAV_PATH)

        for window in ["hann", "hamming", "blackman"]:
            fig, ax = plotter.plot(window=window)
            assert fig is not None
            assert ax is not None
            plt.close(fig)

    def test_plot_custom_title(self):
        """Test spectrum with custom title."""
        plotter = FrequencySpectrumPlotter(AMEN_WAV_PATH)
        fig, ax = plotter.plot(title="My Custom Spectrum")

        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "My Custom Spectrum"
        plt.close(fig)

    def test_plot_average_entire_file(self):
        """Test average spectrum over entire file."""
        plotter = FrequencySpectrumPlotter(AMEN_WAV_PATH)
        fig, ax = plotter.plot_average()

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_average_time_range(self):
        """Test average spectrum over time range."""
        plotter = FrequencySpectrumPlotter(AMEN_WAV_PATH)
        fig, ax = plotter.plot_average(time_range=(0.5, 1.5))

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_average_custom_params(self):
        """Test average spectrum with custom parameters."""
        plotter = FrequencySpectrumPlotter(AMEN_WAV_PATH)
        fig, ax = plotter.plot_average(
            window_size=2048, hop_size=512, min_freq=50, max_freq=8000
        )

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_save(self, tmp_path):
        """Test saving frequency spectrum plot to file."""
        plotter = FrequencySpectrumPlotter(AMEN_WAV_PATH)
        output_path = tmp_path / "spectrum.png"

        plotter.save(str(output_path), time=0.5, dpi=100)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestIntegration:
    """Integration tests for visualization workflows."""

    def test_visualize_complete_workflow(self, tmp_path):
        """Test complete visualization workflow."""
        audio_file = AMEN_WAV_PATH

        # 1. Waveform
        waveform = WaveformPlotter(audio_file)
        fig1, _ = waveform.plot(show_rms=True, show_peaks=True)
        assert fig1 is not None
        waveform.save(str(tmp_path / "waveform.png"))
        plt.close(fig1)

        # 2. Spectrogram
        spectrogram = SpectrogramPlotter(audio_file)
        fig2, _ = spectrogram.plot(cmap="magma")
        assert fig2 is not None
        spectrogram.save(str(tmp_path / "spectrogram.png"))
        plt.close(fig2)

        # 3. Frequency spectrum (instant)
        spectrum = FrequencySpectrumPlotter(audio_file)
        fig3, _ = spectrum.plot(time=1.0)
        assert fig3 is not None
        spectrum.save(str(tmp_path / "spectrum.png"))
        plt.close(fig3)

        # 4. Average frequency spectrum
        fig4, _ = spectrum.plot_average(time_range=(0, 2))
        assert fig4 is not None
        plt.close(fig4)

        # Verify all files were created
        assert (tmp_path / "waveform.png").exists()
        assert (tmp_path / "spectrogram.png").exists()
        assert (tmp_path / "spectrum.png").exists()

    def test_plot_different_time_ranges(self):
        """Test plotting different time ranges from same file."""
        plotter = WaveformPlotter(AMEN_WAV_PATH)

        # Plot first second
        fig1, _ = plotter.plot(time_range=(0, 1))
        assert fig1 is not None
        plt.close(fig1)

        # Plot middle section
        fig2, _ = plotter.plot(time_range=(1, 2))
        assert fig2 is not None
        plt.close(fig2)

        # Plot last section
        fig3, _ = plotter.plot(time_range=(2, 2.5))
        assert fig3 is not None
        plt.close(fig3)

    def test_compare_window_functions(self):
        """Test comparing different window functions."""
        plotter = SpectrogramPlotter(AMEN_WAV_PATH)

        for window in ["hann", "hamming", "blackman"]:
            fig, ax = plotter.plot(window=window, title=f"Window: {window}")
            assert fig is not None
            assert ax.get_title() == f"Window: {window}"
            plt.close(fig)
