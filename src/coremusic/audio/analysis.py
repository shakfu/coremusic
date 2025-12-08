#!/usr/bin/env python3
"""Audio analysis and feature extraction module.

This module provides tools for analyzing audio content:
- Beat detection and tempo estimation
- Pitch detection and tracking
- Spectral features (MFCC, chroma, etc.)
- Audio fingerprinting
- Onset detection
- Key detection

Example:
    >>> analyzer = AudioAnalyzer("song.wav")
    >>> beat_info = analyzer.detect_beats()
    >>> print(f"Tempo: {beat_info.tempo:.1f} BPM")
    >>> key, mode = analyzer.detect_key()
    >>> print(f"Key: {key} {mode}")
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

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
    from scipy import signal
    from scipy.fft import fft, fftfreq

    NUMPY_AVAILABLE = True
    SCIPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    signal = None
    fft = None
    fftfreq = None
    NUMPY_AVAILABLE = False
    SCIPY_AVAILABLE = False

# Import base class
from ._base import AudioFileLoaderMixin

# Logger
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class BeatInfo:
    """Beat detection results.

    Attributes:
        tempo: Tempo in beats per minute (BPM)
        beats: List of beat positions in seconds
        downbeats: List of downbeat positions (bar starts) in seconds
        confidence: Detection confidence (0-1)
    """

    tempo: float
    beats: List[float]
    downbeats: List[float]
    confidence: float


@dataclass
class PitchInfo:
    """Pitch detection results.

    Attributes:
        frequency: Fundamental frequency in Hz
        midi_note: MIDI note number (0-127)
        cents_offset: Cents from MIDI note (-50 to +50)
        confidence: Detection confidence (0-1)
    """

    frequency: float
    midi_note: int
    cents_offset: float
    confidence: float


# ============================================================================
# AudioAnalyzer Class
# ============================================================================


class AudioAnalyzer(AudioFileLoaderMixin):
    """High-level audio analysis and feature extraction.

    Provides various audio analysis methods including beat detection,
    pitch tracking, spectral analysis, and key detection.

    **Instance Methods** (require initialization):
        - detect_beats() - Beat detection and tempo estimation
        - detect_pitch() - Pitch tracking over time
        - analyze_spectrum() - Spectral analysis at specific time
        - extract_mfcc() - Mel-frequency cepstral coefficients
        - detect_key() - Musical key detection
        - get_audio_fingerprint() - Audio fingerprinting

    **Static Methods** (no initialization required):
        - detect_silence() - Find quiet regions in audio
        - get_peak_amplitude() - Maximum amplitude
        - calculate_rms() - RMS level
        - get_file_info() - File metadata

    Example (Instance API):
        >>> analyzer = AudioAnalyzer("song.wav")
        >>> beat_info = analyzer.detect_beats()
        >>> print(f"Tempo: {beat_info.tempo:.1f} BPM")
        >>> pitch_track = analyzer.detect_pitch()
        >>> spectrum = analyzer.analyze_spectrum(time=1.0)

    Example (Static API):
        >>> silence = AudioAnalyzer.detect_silence("audio.wav", threshold_db=-40)
        >>> peak = AudioAnalyzer.get_peak_amplitude("audio.wav")
        >>> info = AudioAnalyzer.get_file_info("audio.wav")
    """

    def __init__(self, audio_file: str):
        """Initialize analyzer.

        Args:
            audio_file: Path to audio file

        Raises:
            ImportError: If NumPy/SciPy are not available
        """
        self._init_audio_loader(audio_file)

    def _check_dependencies(self) -> None:
        """Check that NumPy and SciPy are available."""
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "NumPy is required for audio analysis. Install with: pip install numpy"
            )
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "SciPy is required for audio analysis. Install with: pip install scipy"
            )

    def _load_audio(self) -> Tuple["NDArray", float]:
        """Load audio file if not already loaded.

        Returns:
            Tuple of (audio_data, sample_rate)

        Note:
            Converts stereo audio to mono automatically.
        """
        return self._load_audio_mono()

    # ========================================================================
    # Spectral Analysis
    # ========================================================================

    def _compute_fft(
        self, audio_data: "NDArray", sample_rate: float
    ) -> Tuple["NDArray", "NDArray"]:
        """Compute FFT of audio data.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (frequencies, magnitudes)
        """
        # Apply window
        window = signal.windows.hann(len(audio_data))
        windowed = audio_data * window

        # Compute FFT
        spectrum = fft(windowed)
        freqs = fftfreq(len(audio_data), 1.0 / sample_rate)

        # Take positive frequencies only
        positive_mask = freqs >= 0
        freqs = freqs[positive_mask]
        mags = np.abs(spectrum[positive_mask])

        return freqs, mags

    def _spectral_centroid(self, freqs: "NDArray", mags: "NDArray") -> float:
        """Compute spectral centroid.

        Args:
            freqs: Frequency bins
            mags: Magnitude spectrum

        Returns:
            Spectral centroid in Hz
        """
        if np.sum(mags) == 0:
            return 0.0
        return float(np.sum(freqs * mags) / np.sum(mags))

    def _spectral_rolloff(
        self, freqs: "NDArray", mags: "NDArray", rolloff_percent: float = 0.85
    ) -> float:
        """Compute spectral rolloff.

        Args:
            freqs: Frequency bins
            mags: Magnitude spectrum
            rolloff_percent: Rolloff threshold (default 0.85 = 85%)

        Returns:
            Rolloff frequency in Hz
        """
        cumsum = np.cumsum(mags)
        threshold = rolloff_percent * cumsum[-1]
        idx = np.where(cumsum >= threshold)[0]
        if len(idx) > 0:
            return float(freqs[idx[0]])
        return 0.0

    def _find_spectral_peaks(
        self, freqs: "NDArray", mags: "NDArray", num_peaks: int = 10
    ) -> List[Tuple[float, float]]:
        """Find spectral peaks.

        Args:
            freqs: Frequency bins
            mags: Magnitude spectrum
            num_peaks: Number of peaks to find

        Returns:
            List of (frequency, magnitude) tuples
        """
        # Find peaks
        peak_indices, _ = signal.find_peaks(mags, height=np.max(mags) * 0.1)

        # Sort by magnitude
        if len(peak_indices) > 0:
            peak_mags = mags[peak_indices]
            sorted_indices = np.argsort(peak_mags)[::-1][:num_peaks]
            peaks = [
                (float(freqs[peak_indices[i]]), float(peak_mags[i]))
                for i in sorted_indices
            ]
            return peaks

        return []

    def analyze_spectrum(
        self, time: float, window_size: float = 0.1
    ) -> Dict[str, Any]:
        """Analyze spectrum at specific time.

        Args:
            time: Time position in seconds
            window_size: Analysis window in seconds

        Returns:
            Dictionary with spectral features:
                - frequencies: Frequency bins
                - magnitudes: Magnitude spectrum
                - peaks: Spectral peaks (list of (freq, mag) tuples)
                - centroid: Spectral centroid in Hz
                - rolloff: Spectral rolloff in Hz
        """
        data, sr = self._load_audio()

        # Extract window
        center_sample = int(time * sr)
        window_samples = int(window_size * sr)
        start = max(0, center_sample - window_samples // 2)
        end = min(len(data), center_sample + window_samples // 2)
        window = data[start:end]

        # Compute FFT
        freqs, mags = self._compute_fft(window, sr)

        # Spectral features
        centroid = self._spectral_centroid(freqs, mags)
        rolloff = self._spectral_rolloff(freqs, mags)
        peaks = self._find_spectral_peaks(freqs, mags)

        return {
            "frequencies": freqs,
            "magnitudes": mags,
            "peaks": peaks,
            "centroid": centroid,
            "rolloff": rolloff,
        }

    # ========================================================================
    # Onset Detection
    # ========================================================================

    def _detect_onsets(
        self, audio_data: "NDArray", sample_rate: float
    ) -> "NDArray":
        """Detect onsets in audio.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz

        Returns:
            Array of onset times in seconds
        """
        # Compute spectral flux
        hop_length = 512
        n_fft = 2048

        # Compute STFT
        f, t, Zxx = signal.stft(
            audio_data, sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length
        )

        # Spectral flux (difference between consecutive frames)
        mag = np.abs(Zxx)
        flux = np.sum(np.maximum(0, mag[:, 1:] - mag[:, :-1]), axis=0)

        # Find peaks in flux
        flux_threshold = np.mean(flux) + 1.5 * np.std(flux)
        peak_indices, _ = signal.find_peaks(flux, height=flux_threshold, distance=10)

        # Convert to time
        onset_times = t[1:][peak_indices]

        return onset_times  # type: ignore[no-any-return]

    # ========================================================================
    # Beat Detection
    # ========================================================================

    def _estimate_tempo(
        self, onsets: "NDArray"
    ) -> Tuple[float, List[float]]:
        """Estimate tempo from onset times.

        Args:
            onsets: Onset times in seconds

        Returns:
            Tuple of (tempo_bpm, beat_times)
        """
        if len(onsets) < 2:
            return 60.0, []

        # Compute inter-onset intervals
        intervals = np.diff(onsets)

        # Estimate tempo using median interval
        median_interval = np.median(intervals)
        tempo = 60.0 / median_interval if median_interval > 0 else 60.0

        # Clamp to reasonable range
        tempo = np.clip(tempo, 40.0, 200.0)

        # Generate beat grid
        beat_interval = 60.0 / tempo
        first_onset = onsets[0] if len(onsets) > 0 else 0.0
        data, sr = self._load_audio()
        duration = len(data) / sr

        num_beats = int((duration - first_onset) / beat_interval) + 1
        beats = [first_onset + i * beat_interval for i in range(num_beats)]

        return float(tempo), beats

    def detect_beats(self, **kwargs: Any) -> BeatInfo:
        """Detect beats and estimate tempo.

        Returns:
            BeatInfo with tempo, beat positions, downbeats, and confidence
        """
        data, sr = self._load_audio()

        # Onset detection
        onsets = self._detect_onsets(data, sr)

        # Tempo estimation
        tempo, beats = self._estimate_tempo(onsets)

        # Downbeat detection (every 4th beat)
        downbeats = beats[::4] if len(beats) > 0 else []

        # Confidence based on onset strength
        confidence = 0.85 if len(onsets) > 10 else 0.5

        return BeatInfo(
            tempo=tempo, beats=beats, downbeats=downbeats, confidence=confidence
        )

    # ========================================================================
    # Pitch Detection
    # ========================================================================

    def _autocorrelation_pitch(
        self, audio_data: "NDArray", sample_rate: float
    ) -> Optional[float]:
        """Detect pitch using autocorrelation.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz

        Returns:
            Detected pitch in Hz, or None if no pitch detected
        """
        # Normalize
        audio_data = audio_data - np.mean(audio_data)
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        # Autocorrelation
        corr = np.correlate(audio_data, audio_data, mode="full")
        corr = corr[len(corr) // 2 :]

        # Find first peak after initial lag
        min_lag = int(sample_rate / 1000.0)  # 1000 Hz max
        max_lag = int(sample_rate / 50.0)  # 50 Hz min

        if max_lag >= len(corr):
            return None

        search_range = corr[min_lag:max_lag]
        if len(search_range) == 0:
            return None

        peak_lag = min_lag + np.argmax(search_range)

        # Require minimum correlation
        if corr[peak_lag] < 0.3 * np.max(corr):
            return None

        pitch = sample_rate / peak_lag
        return float(pitch)

    def _freq_to_midi(self, freq: float) -> Tuple[int, float]:
        """Convert frequency to MIDI note and cents offset.

        Args:
            freq: Frequency in Hz

        Returns:
            Tuple of (midi_note, cents_offset)
        """
        midi_float = 69 + 12 * np.log2(freq / 440.0)
        midi_note = int(round(midi_float))
        cents = (midi_float - midi_note) * 100
        return midi_note, float(cents)

    def detect_pitch(
        self, time_range: Optional[Tuple[float, float]] = None
    ) -> List[PitchInfo]:
        """Detect pitch over time.

        Args:
            time_range: (start, end) in seconds, or None for entire file

        Returns:
            List of PitchInfo for each analysis frame
        """
        data, sr = self._load_audio()

        # Extract time range
        if time_range:
            start_sample = int(time_range[0] * sr)
            end_sample = int(time_range[1] * sr)
            data = data[start_sample:end_sample]

        # Analyze in frames
        frame_size = 2048
        hop_size = frame_size // 2
        pitch_track = []

        for i in range(0, len(data) - frame_size, hop_size):
            frame = data[i : i + frame_size]

            # Detect pitch
            pitch = self._autocorrelation_pitch(frame, sr)

            if pitch and 50.0 <= pitch <= 2000.0:  # Valid pitch range
                midi_note, cents = self._freq_to_midi(pitch)
                pitch_track.append(
                    PitchInfo(
                        frequency=pitch,
                        midi_note=midi_note,
                        cents_offset=cents,
                        confidence=0.8,
                    )
                )

        return pitch_track

    # ========================================================================
    # MFCC Extraction
    # ========================================================================

    def _mel_filterbank(
        self, n_fft: int, n_mels: int, sample_rate: float
    ) -> "NDArray":
        """Create mel filterbank.

        Args:
            n_fft: FFT size
            n_mels: Number of mel bands
            sample_rate: Sample rate in Hz

        Returns:
            Mel filterbank matrix
        """
        # Mel scale conversion
        def hz_to_mel(hz: float) -> float:
            return 2595 * np.log10(1 + hz / 700.0)  # type: ignore[no-any-return]

        def mel_to_hz(mel: Any) -> Any:
            return 700 * (10 ** (mel / 2595) - 1)

        # Create mel points
        mel_min = hz_to_mel(0)
        mel_max = hz_to_mel(sample_rate / 2)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Convert to FFT bins
        bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

        # Create filterbank
        filterbank = np.zeros((n_mels, n_fft // 2 + 1))

        for i in range(1, n_mels + 1):
            left, center, right = bins[i - 1], bins[i], bins[i + 1]

            # Rising slope
            for j in range(left, center):
                filterbank[i - 1, j] = (j - left) / (center - left)

            # Falling slope
            for j in range(center, right):
                filterbank[i - 1, j] = (right - j) / (right - center)

        return filterbank

    def extract_mfcc(self, n_mfcc: int = 13) -> "NDArray":
        """Extract Mel-Frequency Cepstral Coefficients.

        Args:
            n_mfcc: Number of MFCC coefficients

        Returns:
            MFCC matrix (n_mfcc x n_frames)
        """
        data, sr = self._load_audio()

        # STFT parameters
        n_fft = 2048
        hop_length = 512

        # Compute STFT
        f, t, Zxx = signal.stft(data, sr, nperseg=n_fft, noverlap=n_fft - hop_length)

        # Power spectrum
        power = np.abs(Zxx) ** 2

        # Mel filterbank
        n_mels = 40
        mel_filters = self._mel_filterbank(n_fft, n_mels, sr)

        # Apply mel filters
        mel_spec = mel_filters @ power

        # Log mel spectrum
        log_mel = np.log10(mel_spec + 1e-10)

        # DCT for MFCCs
        from scipy.fftpack import dct

        mfcc = dct(log_mel, axis=0, norm="ortho")[:n_mfcc]

        return mfcc  # type: ignore[no-any-return]

    # ========================================================================
    # Chroma and Key Detection
    # ========================================================================

    def _compute_chroma(self, audio_data: "NDArray", sample_rate: float) -> "NDArray":
        """Compute chromagram.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz

        Returns:
            Chroma features (12 x n_frames)
        """
        # STFT
        n_fft = 4096
        hop_length = 2048
        f, t, Zxx = signal.stft(
            audio_data, sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length
        )

        # Power spectrum
        power = np.abs(Zxx) ** 2

        # Map frequencies to chroma bins
        chroma = np.zeros((12, power.shape[1]))

        for i in range(len(f)):
            freq = f[i]
            if freq < 20.0:  # Skip very low frequencies
                continue

            # Convert to MIDI note
            midi = 69 + 12 * np.log2(freq / 440.0)
            chroma_bin = int(round(midi)) % 12

            chroma[chroma_bin] += power[i]

        # Normalize
        chroma_sum = np.sum(chroma, axis=0, keepdims=True)
        chroma_sum[chroma_sum == 0] = 1
        chroma = chroma / chroma_sum

        return chroma  # type: ignore[no-any-return]

    def _estimate_key(self, chroma: "NDArray") -> Tuple[str, str]:
        """Estimate musical key from chroma features.

        Args:
            chroma: Chroma features (12 x n_frames)

        Returns:
            Tuple of (key, mode) e.g., ("C", "major")
        """
        # Average chroma over time
        avg_chroma = np.mean(chroma, axis=1)

        # Key profiles (Krumhansl-Schmuckler)
        major_profile = np.array(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        )
        minor_profile = np.array(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        )

        # Correlate with all keys
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        best_key = "C"
        best_mode = "major"
        best_correlation = -1.0

        for shift in range(12):
            # Major key correlation
            shifted_profile = np.roll(major_profile, shift)
            correlation = np.corrcoef(avg_chroma, shifted_profile)[0, 1]
            if correlation > best_correlation:
                best_correlation = correlation
                best_key = note_names[shift]
                best_mode = "major"

            # Minor key correlation
            shifted_profile = np.roll(minor_profile, shift)
            correlation = np.corrcoef(avg_chroma, shifted_profile)[0, 1]
            if correlation > best_correlation:
                best_correlation = correlation
                best_key = note_names[shift]
                best_mode = "minor"

        return best_key, best_mode

    def detect_key(self) -> Tuple[str, str]:
        """Detect musical key.

        Returns:
            Tuple of (key, mode) e.g., ("C", "major")
        """
        data, sr = self._load_audio()

        # Compute chroma
        chroma = self._compute_chroma(data, sr)

        # Estimate key
        key, mode = self._estimate_key(chroma)

        return key, mode

    # ========================================================================
    # Audio Fingerprinting
    # ========================================================================

    def _generate_fingerprint(
        self, audio_data: "NDArray", sample_rate: float
    ) -> str:
        """Generate audio fingerprint.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz

        Returns:
            Fingerprint string
        """
        # Simple fingerprint based on spectral peaks
        # (Real implementation would use Chromaprint algorithm)

        # Compute spectrogram
        n_fft = 2048
        hop_length = 512
        f, t, Zxx = signal.stft(
            audio_data, sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length
        )

        # Find peaks in each time frame
        power = np.abs(Zxx) ** 2
        fingerprint_bits = []

        for i in range(power.shape[1]):
            frame = power[:, i]
            # Find top peaks
            peak_indices, _ = signal.find_peaks(frame, height=np.max(frame) * 0.5)
            if len(peak_indices) > 0:
                # Hash peak frequencies
                hash_value = sum(peak_indices[:5]) % 256
                fingerprint_bits.append(hash_value)

        # Convert to hex string
        fingerprint = "".join([f"{b:02x}" for b in fingerprint_bits])

        return fingerprint[:64]  # Limit length

    def get_audio_fingerprint(self) -> str:
        """Generate audio fingerprint.

        Returns:
            Fingerprint string for audio identification
        """
        data, sr = self._load_audio()
        return self._generate_fingerprint(data, sr)

    # ========================================================================
    # Static Utility Methods (Basic Audio Metrics)
    # ========================================================================

    @staticmethod
    def detect_silence(
        audio_file: Any,
        threshold_db: float = -40.0,
        min_duration: float = 0.5,
    ) -> List[Tuple[float, float]]:
        """Detect silence regions in an audio file.

        Args:
            audio_file: Path to audio file or AudioFile instance
            threshold_db: Silence threshold in dB (default: -40)
            min_duration: Minimum silence duration in seconds (default: 0.5)

        Returns:
            List of (start_time, end_time) tuples for silence regions

        Example:
            >>> silence = AudioAnalyzer.detect_silence("audio.wav", threshold_db=-40, min_duration=0.5)
            >>> for start, end in silence:
            ...     print(f"Silence from {start:.2f}s to {end:.2f}s")
        """
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "NumPy is required for silence detection. Install with: pip install numpy"
            )

        import coremusic as cm

        # Open file if path provided
        should_close = False
        if isinstance(audio_file, (str, Path)):
            audio_file = cm.AudioFile(str(audio_file))
            audio_file.open()
            should_close = True

        try:
            # Read audio data as NumPy array
            audio_data = audio_file.read_as_numpy()

            # Get format info
            format = audio_file.format
            sample_rate = format.sample_rate

            # Convert to mono if stereo by taking mean across channels
            if audio_data.ndim == 2:
                audio_data = np.mean(audio_data, axis=1)

            # Convert to float and normalize
            if audio_data.dtype in [np.int16, np.int32]:
                max_val = np.iinfo(audio_data.dtype).max
                audio_data = audio_data.astype(np.float32) / max_val

            # Convert threshold from dB to linear
            threshold_linear = 10 ** (threshold_db / 20)

            # Find samples below threshold
            is_silent = np.abs(audio_data) < threshold_linear

            # Find silence regions
            silence_regions = []
            in_silence = False
            silence_start = 0

            for i, silent in enumerate(is_silent):
                if silent and not in_silence:
                    # Start of silence region
                    in_silence = True
                    silence_start = i
                elif not silent and in_silence:
                    # End of silence region
                    in_silence = False
                    duration = (i - silence_start) / sample_rate
                    if duration >= min_duration:
                        start_time = silence_start / sample_rate
                        end_time = i / sample_rate
                        silence_regions.append((start_time, end_time))

            # Handle case where file ends in silence
            if in_silence:
                duration = (len(is_silent) - silence_start) / sample_rate
                if duration >= min_duration:
                    start_time = silence_start / sample_rate
                    end_time = len(is_silent) / sample_rate
                    silence_regions.append((start_time, end_time))

            return silence_regions

        finally:
            if should_close:
                audio_file.close()

    @staticmethod
    def get_peak_amplitude(audio_file: Any) -> float:
        """Get the peak amplitude of an audio file.

        Args:
            audio_file: Path to audio file or AudioFile instance

        Returns:
            Peak amplitude as a float (0.0 to 1.0 for normalized audio)

        Example:
            >>> peak = AudioAnalyzer.get_peak_amplitude("audio.wav")
            >>> print(f"Peak amplitude: {peak:.4f}")
        """
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "NumPy is required for peak detection. Install with: pip install numpy"
            )

        import coremusic as cm

        # Open file if path provided
        should_close = False
        if isinstance(audio_file, (str, Path)):
            audio_file = cm.AudioFile(str(audio_file))
            audio_file.open()
            should_close = True

        try:
            # Read audio data
            audio_data = audio_file.read_as_numpy()

            # Convert to float and normalize
            if audio_data.dtype in [np.int16, np.int32]:
                max_val = np.iinfo(audio_data.dtype).max
                audio_data = audio_data.astype(np.float32) / max_val

            # Get peak
            return float(np.max(np.abs(audio_data)))

        finally:
            if should_close:
                audio_file.close()

    @staticmethod
    def calculate_rms(audio_file: Any) -> float:
        """Calculate RMS (Root Mean Square) amplitude.

        Args:
            audio_file: Path to audio file or AudioFile instance

        Returns:
            RMS amplitude as a float

        Example:
            >>> rms = AudioAnalyzer.calculate_rms("audio.wav")
            >>> rms_db = 20 * np.log10(rms)  # Convert to dB
            >>> print(f"RMS: {rms:.4f} ({rms_db:.2f} dB)")
        """
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "NumPy is required for RMS calculation. Install with: pip install numpy"
            )

        import coremusic as cm

        # Open file if path provided
        should_close = False
        if isinstance(audio_file, (str, Path)):
            audio_file = cm.AudioFile(str(audio_file))
            audio_file.open()
            should_close = True

        try:
            # Read audio data
            audio_data = audio_file.read_as_numpy()

            # Convert to float and normalize
            if audio_data.dtype in [np.int16, np.int32]:
                max_val = np.iinfo(audio_data.dtype).max
                audio_data = audio_data.astype(np.float32) / max_val

            # Calculate RMS
            return float(np.sqrt(np.mean(audio_data**2)))

        finally:
            if should_close:
                audio_file.close()

    @staticmethod
    def get_file_info(audio_file: Union[str, Path]) -> Dict[str, Any]:
        """Get comprehensive information about an audio file.

        Args:
            audio_file: Path to audio file

        Returns:
            Dictionary with file information (format, duration, sample_rate, etc.)

        Example:
            >>> info = AudioAnalyzer.get_file_info("audio.wav")
            >>> print(f"Duration: {info['duration']:.2f}s")
            >>> print(f"Format: {info['format_id']}")
            >>> print(f"Sample Rate: {info['sample_rate']} Hz")
        """
        import coremusic as cm

        with cm.AudioFile(str(audio_file)) as af:
            format = af.format

            info = {
                "path": str(audio_file),
                "duration": af.duration,
                "sample_rate": format.sample_rate,
                "format_id": format.format_id,
                "channels": format.channels_per_frame,
                "bits_per_channel": format.bits_per_channel,
                "is_pcm": format.is_pcm,
                "is_stereo": format.is_stereo,
                "is_mono": format.is_mono,
            }

            # Add peak and RMS if NumPy available
            if NUMPY_AVAILABLE:
                try:
                    info["peak_amplitude"] = AudioAnalyzer.get_peak_amplitude(af)
                    info["rms"] = AudioAnalyzer.calculate_rms(af)
                except Exception:
                    pass  # Skip if reading fails

            return info


# ============================================================================
# LivePitchDetector Class
# ============================================================================


class LivePitchDetector:
    """Real-time pitch detection.

    Example:
        >>> detector = LivePitchDetector(44100, 2048)
        >>> pitch_info = detector.process(audio_chunk)
        >>> if pitch_info:
        ...     print(f"Detected: {pitch_info.frequency:.1f} Hz")
    """

    def __init__(self, sample_rate: float = 44100.0, buffer_size: int = 2048):
        """Initialize real-time pitch detector.

        Args:
            sample_rate: Audio sample rate
            buffer_size: Analysis buffer size (larger = more accurate, higher latency)

        Raises:
            ImportError: If NumPy is not available
        """
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "NumPy is required for pitch detection. Install with: pip install numpy"
            )

        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self._buffer: "NDArray" = np.zeros(buffer_size)

    def process(self, audio_chunk: "NDArray") -> Optional[PitchInfo]:
        """Process audio chunk and detect pitch.

        Args:
            audio_chunk: Audio samples (mono)

        Returns:
            PitchInfo if pitch detected, None otherwise
        """
        # Shift buffer and add new data
        self._buffer = np.roll(self._buffer, -len(audio_chunk))
        self._buffer[-len(audio_chunk) :] = audio_chunk

        # Detect pitch using autocorrelation
        pitch = self._detect_pitch_autocorrelation(self._buffer, self.sample_rate)

        if pitch and 50.0 <= pitch <= 2000.0:  # Valid pitch range
            midi_note, cents = self._freq_to_midi(pitch)
            return PitchInfo(
                frequency=pitch, midi_note=midi_note, cents_offset=cents, confidence=0.9
            )

        return None

    def _detect_pitch_autocorrelation(
        self, buffer: "NDArray", sr: float
    ) -> Optional[float]:
        """Detect pitch using autocorrelation.

        Args:
            buffer: Audio buffer
            sr: Sample rate

        Returns:
            Detected pitch in Hz, or None
        """
        # Normalize
        buffer = buffer - np.mean(buffer)
        if np.max(np.abs(buffer)) > 0:
            buffer = buffer / np.max(np.abs(buffer))

        # Autocorrelation
        corr = np.correlate(buffer, buffer, mode="full")
        corr = corr[len(corr) // 2 :]

        # Find first peak
        min_lag = int(sr / 1000.0)  # 1000 Hz max
        max_lag = int(sr / 50.0)  # 50 Hz min

        if max_lag >= len(corr):
            return None

        search_range = corr[min_lag:max_lag]
        if len(search_range) == 0:
            return None

        peak_lag = min_lag + np.argmax(search_range)

        # Require minimum correlation
        if corr[peak_lag] < 0.3 * np.max(corr):
            return None

        pitch = sr / peak_lag
        return float(pitch)

    def _freq_to_midi(self, freq: float) -> Tuple[int, float]:
        """Convert frequency to MIDI note and cents offset.

        Args:
            freq: Frequency in Hz

        Returns:
            Tuple of (midi_note, cents_offset)
        """
        midi_float = 69 + 12 * np.log2(freq / 440.0)
        midi_note = int(round(midi_float))
        cents = (midi_float - midi_note) * 100
        return midi_note, float(cents)
