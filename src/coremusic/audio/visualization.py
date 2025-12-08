#!/usr/bin/env python3
"""Audio visualization module.

This module provides tools for visualizing audio data:
- Waveform plots with optional RMS/peak envelope
- Spectrograms with configurable parameters
- Frequency spectrum analysis
- Real-time audio visualization
- Audio level meters

Example:
    >>> plotter = WaveformPlotter("audio.wav")
    >>> plotter.plot(show_rms=True)
    >>>
    >>> spec = SpectrogramPlotter("audio.wav")
    >>> spec.plot(cmap='magma')
"""

import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple

# Type checking imports
if TYPE_CHECKING:
    try:
        import numpy as np
        from numpy.typing import NDArray as NDArray_
    except ImportError:
        NDArray_ = Any  # type: ignore[misc,assignment]
    NDArray = NDArray_

# Runtime imports
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False

try:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore[assignment]
    animation = None  # type: ignore[assignment]
    Figure = Any  # type: ignore[misc,assignment]
    Axes = Any  # type: ignore[misc,assignment]
    MATPLOTLIB_AVAILABLE = False

# Import coremusic for audio file operations

# Import base class
from ._base import AudioFileLoaderMixin

# Logger
logger = logging.getLogger(__name__)


# ============================================================================
# WaveformPlotter Class
# ============================================================================


class WaveformPlotter(AudioFileLoaderMixin):
    """Plot audio waveforms.

    Provides visualization of audio waveforms with optional overlays for
    RMS envelope, peak envelope, and other features.

    Example:
        >>> plotter = WaveformPlotter("audio.wav")
        >>> plotter.plot(show_rms=True, show_peaks=True)
        >>> plotter.save("waveform.png")
    """

    def __init__(self, audio_file: str):
        """Initialize waveform plotter.

        Args:
            audio_file: Path to audio file

        Raises:
            ImportError: If NumPy or matplotlib not available
        """
        self._init_audio_loader(audio_file)

    def _check_dependencies(self) -> None:
        """Check that NumPy and matplotlib are available."""
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "NumPy is required for visualization. Install with: pip install numpy"
            )
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install matplotlib"
            )

    def plot(
        self,
        time_range: Optional[Tuple[float, float]] = None,
        show_rms: bool = False,
        show_peaks: bool = False,
        rms_window: float = 0.1,
        peak_window: float = 0.05,
        figsize: Tuple[int, int] = (12, 4),
        title: Optional[str] = None,
    ) -> Tuple[Figure, Axes]:
        """Plot waveform.

        Args:
            time_range: (start, end) in seconds, or None for entire file
            show_rms: Whether to overlay RMS envelope
            show_peaks: Whether to overlay peak envelope
            rms_window: RMS window size in seconds
            peak_window: Peak detection window size in seconds
            figsize: Figure size (width, height)
            title: Custom title, or None for default

        Returns:
            Tuple of (figure, axes) for further customization
        """
        data, sr = self._load_audio()

        # Convert to mono if stereo
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Extract time range
        if time_range:
            start_sample = int(time_range[0] * sr)
            end_sample = int(time_range[1] * sr)
            data = data[start_sample:end_sample]
            time_offset = time_range[0]
        else:
            time_offset = 0.0

        # Time axis
        time = np.arange(len(data)) / sr + time_offset

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(time, data, linewidth=0.5, alpha=0.7, label="Waveform")

        if show_rms:
            # Calculate RMS envelope
            window_samples = int(rms_window * sr)
            rms = self._calculate_rms_envelope(data, window_samples)
            ax.plot(time, rms, "r-", linewidth=1.5, label="RMS Envelope")
            ax.plot(time, -rms, "r-", linewidth=1.5)

        if show_peaks:
            # Calculate peak envelope
            window_samples = int(peak_window * sr)
            peak_pos, peak_neg = self._calculate_peak_envelope(data, window_samples)
            ax.plot(
                time, peak_pos, "g--", linewidth=1, alpha=0.7, label="Peak Envelope"
            )
            ax.plot(time, peak_neg, "g--", linewidth=1, alpha=0.7)

        # Formatting
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Waveform: {self.audio_file.name}")

        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)

        if show_rms or show_peaks:
            ax.legend(loc="upper right")

        plt.tight_layout()
        return fig, ax

    def save(
        self,
        output_path: str,
        time_range: Optional[Tuple[float, float]] = None,
        show_rms: bool = False,
        show_peaks: bool = False,
        dpi: int = 150,
        **kwargs: Any,
    ) -> None:
        """Save waveform plot to file.

        Args:
            output_path: Output file path
            time_range: Time range to plot
            show_rms: Show RMS envelope
            show_peaks: Show peak envelope
            dpi: Image resolution
            **kwargs: Additional arguments passed to plot()
        """
        fig, _ = self.plot(
            time_range=time_range, show_rms=show_rms, show_peaks=show_peaks, **kwargs
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved waveform plot to {output_path}")

    def _calculate_rms_envelope(
        self, data: "NDArray", window_size: int
    ) -> "NDArray":
        """Calculate RMS envelope.

        Args:
            data: Audio data
            window_size: Window size in samples

        Returns:
            RMS envelope
        """
        squared = data**2
        window = np.ones(window_size) / window_size
        rms = np.sqrt(np.convolve(squared, window, mode="same"))
        return rms

    def _calculate_peak_envelope(
        self, data: "NDArray", window_size: int
    ) -> Tuple["NDArray", "NDArray"]:
        """Calculate peak envelope.

        Args:
            data: Audio data
            window_size: Window size in samples

        Returns:
            Tuple of (positive_peaks, negative_peaks)
        """
        # Use maximum filter for positive peaks
        peak_pos = np.maximum.accumulate(
            data[: window_size // 2]
        )
        for i in range(window_size // 2, len(data) - window_size // 2):
            window = data[i - window_size // 2 : i + window_size // 2]
            peak_pos = np.append(peak_pos, np.max(window))

        peak_pos = np.append(
            peak_pos,
            np.maximum.accumulate(data[-(window_size // 2) :]),
        )

        # Use minimum filter for negative peaks
        peak_neg = np.minimum.accumulate(
            data[: window_size // 2]
        )
        for i in range(window_size // 2, len(data) - window_size // 2):
            window = data[i - window_size // 2 : i + window_size // 2]
            peak_neg = np.append(peak_neg, np.min(window))

        peak_neg = np.append(
            peak_neg,
            np.minimum.accumulate(data[-(window_size // 2) :]),
        )

        return peak_pos, peak_neg


# ============================================================================
# SpectrogramPlotter Class
# ============================================================================


class SpectrogramPlotter(AudioFileLoaderMixin):
    """Plot audio spectrograms.

    Provides time-frequency visualization of audio using STFT.

    Example:
        >>> plotter = SpectrogramPlotter("audio.wav")
        >>> plotter.plot(cmap='magma', min_db=-80)
        >>> plotter.save("spectrogram.png")
    """

    def __init__(self, audio_file: str):
        """Initialize spectrogram plotter.

        Args:
            audio_file: Path to audio file

        Raises:
            ImportError: If NumPy or matplotlib not available
        """
        self._init_audio_loader(audio_file)

    def _check_dependencies(self) -> None:
        """Check that NumPy and matplotlib are available."""
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "NumPy is required for visualization. Install with: pip install numpy"
            )
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install matplotlib"
            )

    def plot(
        self,
        window_size: int = 2048,
        hop_size: int = 512,
        window: str = "hann",
        cmap: str = "viridis",
        min_db: float = -80.0,
        max_db: Optional[float] = None,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
    ) -> Tuple[Figure, Axes]:
        """Plot spectrogram.

        Args:
            window_size: FFT window size (power of 2)
            hop_size: Hop size between windows
            window: Window function ('hann', 'hamming', 'blackman', etc.)
            cmap: Colormap name ('viridis', 'magma', 'plasma', 'inferno', etc.)
            min_db: Minimum dB value for color scale
            max_db: Maximum dB value for color scale (None for auto)
            figsize: Figure size (width, height)
            title: Custom title, or None for default

        Returns:
            Tuple of (figure, axes) for further customization
        """
        data, sr = self._load_audio()

        # Convert to mono if stereo
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Compute spectrogram
        f, t, Sxx = self._compute_spectrogram(data, sr, window_size, hop_size, window)

        # Convert to dB scale
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # Clamp to min_db
        Sxx_db = np.maximum(Sxx_db, min_db)

        if max_db is not None:
            Sxx_db = np.minimum(Sxx_db, max_db)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.pcolormesh(t, f, Sxx_db, cmap=cmap, shading="gouraud")

        # Formatting
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (seconds)")

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Spectrogram: {self.audio_file.name}")

        _ = fig.colorbar(im, ax=ax, label="Power (dB)")  # colorbar for visual
        plt.tight_layout()

        return fig, ax

    def save(
        self,
        output_path: str,
        window_size: int = 2048,
        hop_size: int = 512,
        dpi: int = 150,
        **kwargs: Any,
    ) -> None:
        """Save spectrogram plot to file.

        Args:
            output_path: Output file path
            window_size: FFT window size
            hop_size: Hop size
            dpi: Image resolution
            **kwargs: Additional arguments passed to plot()
        """
        fig, _ = self.plot(window_size=window_size, hop_size=hop_size, **kwargs)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved spectrogram plot to {output_path}")

    def _compute_spectrogram(
        self,
        data: "NDArray",
        sr: float,
        window_size: int,
        hop_size: int,
        window_type: str,
    ) -> Tuple["NDArray", "NDArray", "NDArray"]:
        """Compute spectrogram using STFT.

        Args:
            data: Audio data
            sr: Sample rate
            window_size: FFT window size
            hop_size: Hop size
            window_type: Window function type

        Returns:
            Tuple of (frequencies, times, spectrogram)
        """
        # Get window function
        if window_type == "hann":
            window = np.hanning(window_size)
        elif window_type == "hamming":
            window = np.hamming(window_size)
        elif window_type == "blackman":
            window = np.blackman(window_size)
        else:
            window = np.ones(window_size)

        # Compute STFT
        num_frames = (len(data) - window_size) // hop_size + 1
        spectrogram = np.zeros((window_size // 2 + 1, num_frames))

        for i in range(num_frames):
            start = i * hop_size
            end = start + window_size
            frame = data[start:end] * window

            # Compute FFT
            fft = np.fft.rfft(frame)
            spectrogram[:, i] = np.abs(fft) ** 2

        # Frequency and time axes
        frequencies = np.fft.rfftfreq(window_size, 1 / sr)
        times = np.arange(num_frames) * hop_size / sr

        return frequencies, times, spectrogram


# ============================================================================
# FrequencySpectrumPlotter Class
# ============================================================================


class FrequencySpectrumPlotter(AudioFileLoaderMixin):
    """Plot frequency spectrum.

    Provides visualization of frequency content at a specific time or
    averaged over a time range.

    Example:
        >>> plotter = FrequencySpectrumPlotter("audio.wav")
        >>> plotter.plot(time=1.5, window_size=4096)
        >>> plotter.plot_average(time_range=(0, 5))
    """

    def __init__(self, audio_file: str):
        """Initialize frequency spectrum plotter.

        Args:
            audio_file: Path to audio file

        Raises:
            ImportError: If NumPy or matplotlib not available
        """
        self._init_audio_loader(audio_file)

    def _check_dependencies(self) -> None:
        """Check that NumPy and matplotlib are available."""
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "NumPy is required for visualization. Install with: pip install numpy"
            )
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install matplotlib"
            )

    def plot(
        self,
        time: float = 0.0,
        window_size: int = 4096,
        window: str = "hann",
        min_freq: float = 20.0,
        max_freq: Optional[float] = None,
        min_db: float = -80.0,
        figsize: Tuple[int, int] = (12, 4),
        title: Optional[str] = None,
    ) -> Tuple[Figure, Axes]:
        """Plot frequency spectrum at specific time.

        Args:
            time: Time in seconds to analyze
            window_size: FFT window size
            window: Window function ('hann', 'hamming', 'blackman')
            min_freq: Minimum frequency to display
            max_freq: Maximum frequency to display (None for Nyquist)
            min_db: Minimum dB value
            figsize: Figure size (width, height)
            title: Custom title, or None for default

        Returns:
            Tuple of (figure, axes) for further customization
        """
        data, sr = self._load_audio()

        # Convert to mono if stereo
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Extract window at specified time
        center_sample = int(time * sr)
        start = max(0, center_sample - window_size // 2)
        end = min(len(data), start + window_size)

        # Pad if necessary
        if end - start < window_size:
            frame = np.zeros(window_size)
            frame[: end - start] = data[start:end]
        else:
            frame = data[start:end]

        # Apply window
        if window == "hann":
            win = np.hanning(window_size)
        elif window == "hamming":
            win = np.hamming(window_size)
        elif window == "blackman":
            win = np.blackman(window_size)
        else:
            win = np.ones(window_size)

        frame = frame * win

        # Compute FFT
        fft = np.fft.rfft(frame)
        frequencies = np.fft.rfftfreq(window_size, 1 / sr)
        magnitude = np.abs(fft)

        # Convert to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        magnitude_db = np.maximum(magnitude_db, min_db)

        # Filter frequency range
        if max_freq is None:
            max_freq = sr / 2

        freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        frequencies = frequencies[freq_mask]
        magnitude_db = magnitude_db[freq_mask]

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(frequencies, magnitude_db, linewidth=1)

        # Formatting
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")

        if title:
            ax.set_title(title)
        else:
            ax.set_title(
                f"Frequency Spectrum: {self.audio_file.name} @ {time:.2f}s"
            )

        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        plt.tight_layout()

        return fig, ax

    def plot_average(
        self,
        time_range: Optional[Tuple[float, float]] = None,
        window_size: int = 4096,
        hop_size: int = 1024,
        min_freq: float = 20.0,
        max_freq: Optional[float] = None,
        min_db: float = -80.0,
        figsize: Tuple[int, int] = (12, 4),
        title: Optional[str] = None,
    ) -> Tuple[Figure, Axes]:
        """Plot average frequency spectrum over time range.

        Args:
            time_range: (start, end) in seconds, or None for entire file
            window_size: FFT window size
            hop_size: Hop size between windows
            min_freq: Minimum frequency to display
            max_freq: Maximum frequency to display (None for Nyquist)
            min_db: Minimum dB value
            figsize: Figure size (width, height)
            title: Custom title, or None for default

        Returns:
            Tuple of (figure, axes) for further customization
        """
        data, sr = self._load_audio()

        # Convert to mono if stereo
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Extract time range
        if time_range:
            start_sample = int(time_range[0] * sr)
            end_sample = int(time_range[1] * sr)
            data = data[start_sample:end_sample]

        # Compute average spectrum
        num_frames = (len(data) - window_size) // hop_size + 1
        spectra = []

        window = np.hanning(window_size)

        for i in range(num_frames):
            start = i * hop_size
            end = start + window_size
            frame = data[start:end] * window

            fft = np.fft.rfft(frame)
            magnitude = np.abs(fft)
            spectra.append(magnitude)

        # Average
        avg_spectrum = np.mean(spectra, axis=0)
        frequencies = np.fft.rfftfreq(window_size, 1 / sr)

        # Convert to dB
        magnitude_db = 20 * np.log10(avg_spectrum + 1e-10)
        magnitude_db = np.maximum(magnitude_db, min_db)

        # Filter frequency range
        if max_freq is None:
            max_freq = sr / 2

        freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        frequencies = frequencies[freq_mask]
        magnitude_db = magnitude_db[freq_mask]

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(frequencies, magnitude_db, linewidth=1)

        # Formatting
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")

        if title:
            ax.set_title(title)
        else:
            if time_range:
                ax.set_title(
                    f"Average Spectrum: {self.audio_file.name} "
                    f"({time_range[0]:.2f}s - {time_range[1]:.2f}s)"
                )
            else:
                ax.set_title(
                    f"Average Spectrum: {self.audio_file.name}"
                )

        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        plt.tight_layout()

        return fig, ax

    def save(
        self,
        output_path: str,
        time: float = 0.0,
        window_size: int = 4096,
        dpi: int = 150,
        **kwargs: Any,
    ) -> None:
        """Save frequency spectrum plot to file.

        Args:
            output_path: Output file path
            time: Time to analyze
            window_size: FFT window size
            dpi: Image resolution
            **kwargs: Additional arguments passed to plot()
        """
        fig, _ = self.plot(time=time, window_size=window_size, **kwargs)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved frequency spectrum plot to {output_path}")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "WaveformPlotter",
    "SpectrogramPlotter",
    "FrequencySpectrumPlotter",
    "MATPLOTLIB_AVAILABLE",
    "NUMPY_AVAILABLE",
]
