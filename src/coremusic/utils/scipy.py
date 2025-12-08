"""SciPy signal processing integration for CoreMusic.

This module provides seamless integration between CoreMusic and SciPy
for advanced signal processing operations including filtering, resampling,
and spectral analysis.

Features:
    - Filter design and application (Butterworth, Chebyshev, etc.)
    - Signal resampling using scipy.signal.resample
    - FFT and spectral analysis
    - Convenience utilities for common DSP workflows

Example:
    ```python
    import coremusic as cm
    import coremusic.scipy_utils as spu

    # Load audio file
    with cm.AudioFile("audio.wav") as af:
        # Read as NumPy array
        data, sr = af.read_frames_numpy()

        # Apply lowpass filter
        filtered = spu.apply_lowpass_filter(data, cutoff=1000, sample_rate=sr)

        # Resample to different rate
        resampled = spu.resample_audio(data, original_rate=sr, target_rate=48000)

        # Compute spectrum
        freqs, spectrum = spu.compute_spectrum(data, sample_rate=sr)
    ```
"""

from typing import Any, Literal, Optional, Tuple, Union

# Check for NumPy availability
try:
    import numpy as np
    from numpy.typing import NDArray

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore
    NDArray = None  # type: ignore

# Check for SciPy availability
try:
    import scipy.fft
    import scipy.signal

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy = None


def _require_scipy(func_name: str = "this function"):
    """Raise ImportError if SciPy is not available."""
    if not SCIPY_AVAILABLE:
        raise ImportError(
            f"SciPy is required to use {func_name}. Install it with: pip install scipy"
        )


def _require_numpy(func_name: str = "this function"):
    """Raise ImportError if NumPy is not available."""
    if not NUMPY_AVAILABLE:
        raise ImportError(
            f"NumPy is required to use {func_name}. Install it with: pip install numpy"
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "SCIPY_AVAILABLE",
    # Filter design
    "design_butterworth_filter",
    "design_chebyshev_filter",
    # Filter application
    "apply_filter",
    "apply_lowpass_filter",
    "apply_highpass_filter",
    "apply_bandpass_filter",
    # Resampling
    "resample_audio",
    # Spectral analysis
    "compute_spectrum",
    "compute_fft",
    "compute_spectrogram",
    # High-level processor
    "AudioSignalProcessor",
]

# ============================================================================
# Filter Design Utilities
# ============================================================================


def design_butterworth_filter(
    cutoff: Union[float, Tuple[float, float]],
    sample_rate: float,
    order: int = 5,
    filter_type: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass",
) -> Tuple["NDArray[Any]", "NDArray[Any]"]:
    """Design a Butterworth filter.

    Args:
        cutoff: Cutoff frequency in Hz. For bandpass/bandstop, provide tuple (low, high)
        sample_rate: Sample rate in Hz
        order: Filter order (higher = steeper rolloff)
        filter_type: Type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop')

    Returns:
        Tuple of (b, a) filter coefficients for use with scipy.signal.filtfilt

    Example:
        ```python
        # Design 5th-order lowpass filter at 1kHz
        b, a = design_butterworth_filter(1000, sample_rate=44100, order=5)

        # Design bandpass filter 300-3000 Hz
        b, a = design_butterworth_filter((300, 3000), sample_rate=44100,
                                        filter_type='bandpass')
        ```

    Raises:
        ImportError: If SciPy is not installed
    """
    _require_scipy("design_butterworth_filter")

    # Normalize frequency to Nyquist frequency
    nyquist = sample_rate / 2.0

    if isinstance(cutoff, tuple):
        normalized_cutoff: Union[float, list[float]] = [f / nyquist for f in cutoff]
    else:
        normalized_cutoff = cutoff / nyquist

    b, a = scipy.signal.butter(
        order, normalized_cutoff, btype=filter_type, analog=False
    )
    return b, a


def design_chebyshev_filter(
    cutoff: Union[float, Tuple[float, float]],
    sample_rate: float,
    order: int = 5,
    ripple_db: float = 0.5,
    filter_type: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass",
) -> Tuple["NDArray[Any]", "NDArray[Any]"]:
    """Design a Chebyshev Type I filter.

    Args:
        cutoff: Cutoff frequency in Hz. For bandpass/bandstop, provide tuple (low, high)
        sample_rate: Sample rate in Hz
        order: Filter order
        ripple_db: Maximum ripple allowed in passband (dB)
        filter_type: Type of filter

    Returns:
        Tuple of (b, a) filter coefficients

    Raises:
        ImportError: If SciPy is not installed
    """
    _require_scipy("design_chebyshev_filter")

    nyquist = sample_rate / 2.0

    if isinstance(cutoff, tuple):
        normalized_cutoff: Union[float, list[float]] = [f / nyquist for f in cutoff]
    else:
        normalized_cutoff = cutoff / nyquist

    b, a = scipy.signal.cheby1(
        order, ripple_db, normalized_cutoff, btype=filter_type, analog=False
    )
    return b, a


# ============================================================================
# Filter Application
# ============================================================================


def apply_filter(
    audio_data: "NDArray[Any]", b: "NDArray[Any]", a: "NDArray[Any]", zero_phase: bool = True
) -> "NDArray[Any]":
    """Apply a digital filter to audio data.

    Args:
        audio_data: Input audio array (1D for mono, 2D for multi-channel)
        b: Numerator coefficients of the filter
        a: Denominator coefficients of the filter
        zero_phase: If True, use filtfilt for zero-phase filtering (recommended for audio)

    Returns:
        Filtered audio data with same shape as input

    Example:
        ```python
        # Design and apply filter
        b, a = design_butterworth_filter(1000, sample_rate=44100)
        filtered = apply_filter(audio_data, b, a)
        ```

    Raises:
        ImportError: If SciPy is not installed
    """
    _require_scipy("apply_filter")
    _require_numpy("apply_filter")

    # Handle multi-channel audio
    if audio_data.ndim == 1:
        # Mono audio
        if zero_phase:
            return scipy.signal.filtfilt(b, a, audio_data)  # type: ignore[no-any-return]
        else:
            return scipy.signal.lfilter(b, a, audio_data)  # type: ignore[no-any-return]
    elif audio_data.ndim == 2:
        # Multi-channel audio - filter each channel
        filtered = np.zeros_like(audio_data)
        for ch in range(audio_data.shape[1]):
            if zero_phase:
                filtered[:, ch] = scipy.signal.filtfilt(b, a, audio_data[:, ch])
            else:
                filtered[:, ch] = scipy.signal.lfilter(b, a, audio_data[:, ch])
        return filtered
    else:
        raise ValueError(f"Audio data must be 1D or 2D, got {audio_data.ndim}D")


def apply_scipy_filter(
    audio_data: "NDArray[Any]",
    filter_output: Union[Tuple["NDArray[Any]", "NDArray[Any]"], Any],
    zero_phase: bool = True,
) -> "NDArray[Any]":
    """Apply a filter designed by scipy.signal functions directly.

    This is a convenience wrapper that accepts the output from scipy.signal
    filter design functions (like butter, cheby1, etc.) and applies them to
    audio data.

    Args:
        audio_data: Input audio array (mono or stereo)
        filter_output: Output from scipy.signal.butter(), cheby1(), etc.
                      Should be a tuple of (b, a) coefficients
        zero_phase: If True, use filtfilt for zero-phase filtering (default)

    Returns:
        Filtered audio data with same shape as input

    Example:
        ```python
        import scipy.signal
        import coremusic.scipy_utils as spu


        # Design and apply filter in one line
        filtered = spu.apply_scipy_filter(
            audio_data,
            scipy.signal.butter(5, 1000, 'low', fs=44100)
        )

        # Also works with other scipy filter functions
        filtered = spu.apply_scipy_filter(
            audio_data,
            scipy.signal.cheby1(4, 0.5, 2000, 'high', fs=44100)
        )
        ```

    Raises:
        ImportError: If SciPy is not installed
        ValueError: If filter_output is not a valid (b, a) tuple
    """
    _require_scipy("apply_scipy_filter")
    _require_numpy("apply_scipy_filter")

    # Validate filter output
    if isinstance(filter_output, tuple) and len(filter_output) == 2:
        b, a = filter_output
        # Ensure they are arrays
        if not (hasattr(b, "__len__") and hasattr(a, "__len__")):
            raise ValueError(
                "filter_output must contain array-like b and a coefficients"
            )
    else:
        raise ValueError(
            "filter_output must be a (b, a) tuple from scipy.signal filter design functions. "
            f"Got {type(filter_output)}"
        )

    # Use the existing apply_filter function
    return apply_filter(audio_data, b, a, zero_phase)


def apply_lowpass_filter(
    audio_data: "NDArray[Any]", cutoff: float, sample_rate: float, order: int = 5
) -> "NDArray[Any]":
    """Apply a lowpass filter to audio data.

    Args:
        audio_data: Input audio array
        cutoff: Cutoff frequency in Hz
        sample_rate: Sample rate in Hz
        order: Filter order

    Returns:
        Filtered audio data

    Example:
        ```python
        # Remove frequencies above 1kHz
        filtered = apply_lowpass_filter(audio_data, cutoff=1000, sample_rate=44100)
        ```

    Raises:
        ImportError: If SciPy is not installed
    """
    b, a = design_butterworth_filter(cutoff, sample_rate, order, "lowpass")
    return apply_filter(audio_data, b, a, zero_phase=True)


def apply_highpass_filter(
    audio_data: "NDArray[Any]", cutoff: float, sample_rate: float, order: int = 5
) -> "NDArray[Any]":
    """Apply a highpass filter to audio data.

    Args:
        audio_data: Input audio array
        cutoff: Cutoff frequency in Hz
        sample_rate: Sample rate in Hz
        order: Filter order

    Returns:
        Filtered audio data

    Example:
        ```python
        # Remove frequencies below 100Hz
        filtered = apply_highpass_filter(audio_data, cutoff=100, sample_rate=44100)
        ```

    Raises:
        ImportError: If SciPy is not installed
    """
    b, a = design_butterworth_filter(cutoff, sample_rate, order, "highpass")
    return apply_filter(audio_data, b, a, zero_phase=True)


def apply_bandpass_filter(
    audio_data: "NDArray[Any]",
    lowcut: float,
    highcut: float,
    sample_rate: float,
    order: int = 5,
) -> "NDArray[Any]":
    """Apply a bandpass filter to audio data.

    Args:
        audio_data: Input audio array
        lowcut: Low cutoff frequency in Hz
        highcut: High cutoff frequency in Hz
        sample_rate: Sample rate in Hz
        order: Filter order

    Returns:
        Filtered audio data

    Example:
        ```python
        # Keep only frequencies between 300Hz and 3kHz
        filtered = apply_bandpass_filter(audio_data, lowcut=300, highcut=3000,
                                        sample_rate=44100)
        ```

    Raises:
        ImportError: If SciPy is not installed
    """
    b, a = design_butterworth_filter((lowcut, highcut), sample_rate, order, "bandpass")
    return apply_filter(audio_data, b, a, zero_phase=True)


# ============================================================================
# Resampling
# ============================================================================


def resample_audio(
    audio_data: "NDArray[Any]",
    original_rate: float,
    target_rate: float,
    method: Literal["fft", "polyphase"] = "fft",
) -> "NDArray[Any]":
    """Resample audio to a different sample rate using SciPy.

    Args:
        audio_data: Input audio array (1D for mono, 2D for multi-channel)
        original_rate: Original sample rate in Hz
        target_rate: Target sample rate in Hz
        method: Resampling method ('fft' or 'polyphase')
            - 'fft': Uses scipy.signal.resample (Fourier method, high quality)
            - 'polyphase': Uses scipy.signal.resample_poly (efficient for integer ratios)

    Returns:
        Resampled audio data

    Example:
        ```python
        # Resample from 44.1kHz to 48kHz
        resampled = resample_audio(audio_data, original_rate=44100, target_rate=48000)
        ```

    Note:
        For high-quality resampling, consider using AudioConverter.convert_with_callback()
        which uses CoreAudio's native resampling (may be higher quality for some use cases).

    Raises:
        ImportError: If SciPy is not installed
    """
    _require_scipy("resample_audio")
    _require_numpy("resample_audio")

    if original_rate == target_rate:
        return audio_data.copy()

    # Calculate number of samples in resampled signal
    num_samples = int(len(audio_data) * target_rate / original_rate)

    if method == "fft":
        # Handle multi-channel audio
        if audio_data.ndim == 1:
            return scipy.signal.resample(audio_data, num_samples)  # type: ignore[no-any-return]
        elif audio_data.ndim == 2:
            resampled = np.zeros((num_samples, audio_data.shape[1]))
            for ch in range(audio_data.shape[1]):
                resampled[:, ch] = scipy.signal.resample(audio_data[:, ch], num_samples)
            return resampled
        else:
            raise ValueError(f"Audio data must be 1D or 2D, got {audio_data.ndim}D")

    elif method == "polyphase":
        # Compute rational approximation of rate ratio
        from fractions import Fraction

        ratio = Fraction(target_rate / original_rate).limit_denominator(1000)
        up = ratio.numerator
        down = ratio.denominator

        if audio_data.ndim == 1:
            return scipy.signal.resample_poly(audio_data, up, down)  # type: ignore[no-any-return]
        elif audio_data.ndim == 2:
            resampled = np.zeros((num_samples, audio_data.shape[1]))
            for ch in range(audio_data.shape[1]):
                resampled[:, ch] = scipy.signal.resample_poly(
                    audio_data[:, ch], up, down
                )
            return resampled
        else:
            raise ValueError(f"Audio data must be 1D or 2D, got {audio_data.ndim}D")

    else:
        raise ValueError(f"Unknown resampling method: {method}")


# ============================================================================
# Spectral Analysis
# ============================================================================


def compute_spectrum(
    audio_data: "NDArray[Any]",
    sample_rate: float,
    window: Optional[str] = "hann",
    nperseg: Optional[int] = None,
) -> Tuple["NDArray[Any]", "NDArray[Any]"]:
    """Compute the frequency spectrum of audio data.

    Args:
        audio_data: Input audio array (1D for mono)
        sample_rate: Sample rate in Hz
        window: Window function to apply ('hann', 'hamming', 'blackman', None)
        nperseg: Length of each segment for Welch's method (default: len(audio_data))

    Returns:
        Tuple of (frequencies, spectrum) where:
            - frequencies: Array of frequency bins in Hz
            - spectrum: Power spectral density

    Example:
        ```python
        freqs, spectrum = compute_spectrum(audio_data, sample_rate=44100)

        # Plot spectrum
        import matplotlib.pyplot as plt
        plt.semilogy(freqs, spectrum)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')
        ```

    Raises:
        ImportError: If SciPy is not installed
    """
    _require_scipy("compute_spectrum")
    _require_numpy("compute_spectrum")

    if audio_data.ndim > 1:
        # Use first channel for multi-channel audio
        audio_data = audio_data[:, 0]

    if nperseg is None:
        nperseg = len(audio_data)

    frequencies, spectrum = scipy.signal.welch(
        audio_data,
        fs=sample_rate,
        window=window if window else "boxcar",
        nperseg=nperseg,
    )

    return frequencies, spectrum


def compute_fft(
    audio_data: "NDArray[Any]", sample_rate: float, window: Optional[str] = "hann"
) -> Tuple["NDArray[Any]", "NDArray[Any]"]:
    """Compute the Fast Fourier Transform of audio data.

    Args:
        audio_data: Input audio array (1D for mono)
        sample_rate: Sample rate in Hz
        window: Window function to apply before FFT ('hann', 'hamming', 'blackman', None)

    Returns:
        Tuple of (frequencies, magnitudes) where:
            - frequencies: Array of frequency bins in Hz
            - magnitudes: Magnitude of FFT (absolute values)

    Example:
        ```python
        freqs, mags = compute_fft(audio_data, sample_rate=44100)

        # Find dominant frequency
        dominant_freq = freqs[np.argmax(mags)]
        print(f"Dominant frequency: {dominant_freq:.1f} Hz")
        ```

    Raises:
        ImportError: If SciPy is not installed
    """
    _require_scipy("compute_fft")
    _require_numpy("compute_fft")

    if audio_data.ndim > 1:
        # Use first channel for multi-channel audio
        audio_data = audio_data[:, 0]

    # Apply window if specified
    if window:
        window_func = scipy.signal.get_window(window, len(audio_data))
        audio_data = audio_data * window_func

    # Compute FFT
    fft_result = scipy.fft.rfft(audio_data)
    magnitudes = np.abs(fft_result)

    # Compute frequency bins
    frequencies = scipy.fft.rfftfreq(len(audio_data), 1.0 / sample_rate)

    return frequencies, magnitudes


def compute_spectrogram(
    audio_data: "NDArray[Any]",
    sample_rate: float,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: Optional[int] = None,
) -> Tuple["NDArray[Any]", "NDArray[Any]", "NDArray[Any]"]:
    """Compute spectrogram of audio data.

    Args:
        audio_data: Input audio array (1D for mono)
        sample_rate: Sample rate in Hz
        window: Window function
        nperseg: Length of each segment
        noverlap: Number of points to overlap between segments

    Returns:
        Tuple of (frequencies, times, spectrogram) where:
            - frequencies: Array of frequency bins
            - times: Array of time bins
            - spectrogram: Spectrogram (2D array)

    Example:
        ```python
        freqs, times, Sxx = compute_spectrogram(audio_data, sample_rate=44100)

        # Plot spectrogram
        import matplotlib.pyplot as plt
        plt.pcolormesh(times, freqs, 10 * np.log10(Sxx), shading='gouraud')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.colorbar(label='Power (dB)')
        ```

    Raises:
        ImportError: If SciPy is not installed
    """
    _require_scipy("compute_spectrogram")
    _require_numpy("compute_spectrogram")

    if audio_data.ndim > 1:
        # Use first channel for multi-channel audio
        audio_data = audio_data[:, 0]

    frequencies, times, spectrogram = scipy.signal.spectrogram(
        audio_data, fs=sample_rate, window=window, nperseg=nperseg, noverlap=noverlap
    )

    return frequencies, times, spectrogram


# ============================================================================
# AudioSignalProcessor Class
# ============================================================================


class AudioSignalProcessor:
    """High-level audio signal processing interface using SciPy.

    This class provides a convenient interface for common audio DSP operations
    including filtering, resampling, and spectral analysis.

    Example:
        ```python
        import coremusic as cm
        import coremusic.scipy_utils as spu

        # Load audio
        with cm.AudioFile("audio.wav") as af:
            data, sr = af.read_frames_numpy()

        # Create processor
        processor = spu.AudioSignalProcessor(data, sr)

        # Chain operations
        processor.lowpass(1000).highpass(100).normalize()

        # Get processed audio
        processed = processor.get_audio()
        ```
    """

    def __init__(self, audio_data: "NDArray[Any]", sample_rate: float):
        """Initialize AudioSignalProcessor.

        Args:
            audio_data: Audio data as NumPy array
            sample_rate: Sample rate in Hz

        Raises:
            ImportError: If NumPy or SciPy is not installed
        """
        _require_numpy("AudioSignalProcessor")
        _require_scipy("AudioSignalProcessor")

        self.audio_data = audio_data.copy()
        self.sample_rate = float(sample_rate)
        self._original_data = audio_data.copy()

    def lowpass(self, cutoff: float, order: int = 5) -> "AudioSignalProcessor":
        """Apply lowpass filter (chainable).

        Args:
            cutoff: Cutoff frequency in Hz
            order: Filter order

        Returns:
            Self for method chaining
        """
        self.audio_data = apply_lowpass_filter(
            self.audio_data, cutoff, self.sample_rate, order
        )
        return self

    def highpass(self, cutoff: float, order: int = 5) -> "AudioSignalProcessor":
        """Apply highpass filter (chainable).

        Args:
            cutoff: Cutoff frequency in Hz
            order: Filter order

        Returns:
            Self for method chaining
        """
        self.audio_data = apply_highpass_filter(
            self.audio_data, cutoff, self.sample_rate, order
        )
        return self

    def bandpass(
        self, lowcut: float, highcut: float, order: int = 5
    ) -> "AudioSignalProcessor":
        """Apply bandpass filter (chainable).

        Args:
            lowcut: Low cutoff frequency in Hz
            highcut: High cutoff frequency in Hz
            order: Filter order

        Returns:
            Self for method chaining
        """
        self.audio_data = apply_bandpass_filter(
            self.audio_data, lowcut, highcut, self.sample_rate, order
        )
        return self

    def resample(
        self, target_rate: float, method: Literal["fft", "polyphase"] = "fft"
    ) -> "AudioSignalProcessor":
        """Resample audio to different sample rate (chainable).

        Args:
            target_rate: Target sample rate in Hz
            method: Resampling method ('fft' or 'polyphase')

        Returns:
            Self for method chaining
        """
        self.audio_data = resample_audio(
            self.audio_data, self.sample_rate, target_rate, method
        )
        self.sample_rate = target_rate
        return self

    def normalize(self, target_level: float = 1.0) -> "AudioSignalProcessor":
        """Normalize audio to target level (chainable).

        Args:
            target_level: Target peak level (default: 1.0 for full scale)

        Returns:
            Self for method chaining
        """
        max_val = np.max(np.abs(self.audio_data))
        if max_val > 0:
            self.audio_data = self.audio_data * (target_level / max_val)
        return self

    def get_audio(self) -> "NDArray[Any]":
        """Get processed audio data.

        Returns:
            Processed audio as NumPy array
        """
        return self.audio_data

    def get_sample_rate(self) -> float:
        """Get current sample rate.

        Returns:
            Sample rate in Hz
        """
        return self.sample_rate

    def reset(self) -> "AudioSignalProcessor":
        """Reset to original audio data (chainable).

        Returns:
            Self for method chaining
        """
        self.audio_data = self._original_data.copy()
        return self

    def spectrum(self, **kwargs: Any) -> Tuple["NDArray[Any]", "NDArray[Any]"]:
        """Compute frequency spectrum of current audio.

        Args:
            **kwargs: Additional arguments passed to compute_spectrum()

        Returns:
            Tuple of (frequencies, spectrum)
        """
        return compute_spectrum(self.audio_data, self.sample_rate, **kwargs)

    def fft(self, **kwargs: Any) -> Tuple["NDArray[Any]", "NDArray[Any]"]:
        """Compute FFT of current audio.

        Args:
            **kwargs: Additional arguments passed to compute_fft()

        Returns:
            Tuple of (frequencies, magnitudes)
        """
        return compute_fft(self.audio_data, self.sample_rate, **kwargs)

    def spectrogram(self, **kwargs: Any) -> Tuple["NDArray[Any]", "NDArray[Any]", "NDArray[Any]"]:
        """Compute spectrogram of current audio.

        Args:
            **kwargs: Additional arguments passed to compute_spectrogram()

        Returns:
            Tuple of (frequencies, times, spectrogram)
        """
        return compute_spectrogram(self.audio_data, self.sample_rate, **kwargs)
