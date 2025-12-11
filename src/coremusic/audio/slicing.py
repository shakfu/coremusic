#!/usr/bin/env python3
"""Audio slicing and recombination module.

This module provides tools for slicing audio into segments and recombining them:
- Automatic onset/transient detection slicing
- Manual threshold-based slicing
- Zero-crossing and grid-based slicing
- Individual slice export
- Creative recombination (random, reverse, pattern-based)
- Beat slicing for rhythm manipulation

Example:
    >>> slicer = AudioSlicer("drums.wav", method="onset")
    >>> slices = slicer.detect_slices(max_slices=16)
    >>> recombinator = SliceRecombinator(slices)
    >>> recombinator.export("shuffled.wav", method="random")
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, List, Literal, Optional,
                    Tuple, Union)

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
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

# Logger
logger = logging.getLogger(__name__)

# Type aliases
SliceMethod = Literal["onset", "transient", "zero_crossing", "grid", "manual"]
RecombineMethod = Literal["original", "random", "reverse", "pattern", "custom"]


# ============================================================================
# Slice Data Class
# ============================================================================


@dataclass
class Slice:
    """Represents a single audio slice.

    Attributes:
        start: Start time in seconds
        end: End time in seconds
        data: Audio data (mono or stereo)
        sample_rate: Sample rate in Hz
        index: Original position in sequence
        confidence: Detection confidence (0-1)
    """

    start: float
    end: float
    data: "NDArray"
    sample_rate: float
    index: int
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        """Slice duration in seconds."""
        return self.end - self.start

    @property
    def num_samples(self) -> int:
        """Number of samples in slice."""
        return len(self.data) if self.data.ndim == 1 else self.data.shape[0]

    def export(self, output_path: str, format: str = "wav") -> None:
        """Export slice as audio file.

        Args:
            output_path: Output file path
            format: Audio format ('wav', 'aiff', etc.)

        Note:
            This is a placeholder - full write support would require
            extending AudioFile with write capabilities.
        """
        logger.info(
            f"Export slice {self.index} to {output_path} "
            f"(duration: {self.duration:.3f}s)"
        )
        # Placeholder for actual file writing
        # In full implementation, would use cm.AudioFile in write mode


# ============================================================================
# AudioSlicer Class
# ============================================================================


class AudioSlicer:
    """Slice audio files into segments.

    Supports multiple slicing methods:
    - onset: Detect onsets using spectral flux
    - transient: Detect transients using envelope analysis
    - zero_crossing: Slice at zero crossings (glitch-free)
    - grid: Regular grid slicing (optionally beat-aligned)
    - manual: Slice at specified time points

    Example:
        >>> slicer = AudioSlicer("audio.wav", method="onset", sensitivity=0.7)
        >>> slices = slicer.detect_slices(max_slices=16)
        >>> print(f"Detected {len(slices)} slices")
    """

    def __init__(
        self,
        audio_file: str,
        method: SliceMethod = "onset",
        sensitivity: float = 0.5,
    ):
        """Initialize audio slicer.

        Args:
            audio_file: Path to audio file
            method: Slicing method
            sensitivity: Detection sensitivity 0.0 (loose) to 1.0 (strict)

        Raises:
            ImportError: If NumPy is not available
        """
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "NumPy is required for audio slicing. Install with: pip install numpy"
            )

        self.audio_file = audio_file
        self.method = method
        self.sensitivity = sensitivity
        self._audio_data: Optional["NDArray"] = None
        self._sample_rate: Optional[float] = None
        self._slices: Optional[List[Slice]] = None

    def _load_audio(self) -> Tuple["NDArray", float]:
        """Load audio file if not already loaded.

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if self._audio_data is None:
            import coremusic as cm

            with cm.AudioFile(self.audio_file) as af:
                self._audio_data = af.read_as_numpy()
                self._sample_rate = af.format.sample_rate

                # Convert to mono for analysis
                if len(self._audio_data.shape) > 1 and self._audio_data.shape[1] > 1:
                    self._audio_data = np.mean(self._audio_data, axis=1)

        return self._audio_data, self._sample_rate  # type: ignore[return-value]

    def detect_slices(self, **kwargs: Any) -> List[Slice]:
        """Detect slice points using configured method.

        Returns:
            List of Slice objects

        Raises:
            ValueError: If unknown slicing method
        """
        if self.method == "onset":
            return self._detect_onset_slices(**kwargs)
        elif self.method == "transient":
            return self._detect_transient_slices(**kwargs)
        elif self.method == "zero_crossing":
            return self._detect_zero_crossing_slices(**kwargs)
        elif self.method == "grid":
            return self._detect_grid_slices(**kwargs)
        elif self.method == "manual":
            return self._detect_manual_slices(**kwargs)
        else:
            raise ValueError(f"Unknown slicing method: {self.method}")

    # ========================================================================
    # Onset-based Slicing
    # ========================================================================

    def _detect_onset_slices(
        self, min_slice_duration: float = 0.05, max_slices: Optional[int] = None
    ) -> List[Slice]:
        """Detect slices based on onset detection.

        Args:
            min_slice_duration: Minimum slice duration in seconds
            max_slices: Maximum number of slices (None = unlimited)

        Returns:
            List of slices based on detected onsets
        """
        data, sr = self._load_audio()

        # Detect onsets with sensitivity adjustment
        onset_times = self._detect_onsets(data, sr, self.sensitivity)

        # Filter by minimum duration
        filtered_onsets = self._filter_onsets(onset_times, min_slice_duration)

        # Limit number of slices
        if max_slices and len(filtered_onsets) > max_slices:
            # Keep most prominent onsets
            filtered_onsets = self._select_prominent_onsets(
                filtered_onsets, data, sr, max_slices
            )

        # Create slices
        slices = self._create_slices_from_times(
            filtered_onsets, data, sr, confidence=0.9
        )

        self._slices = slices
        return slices

    def _detect_onsets(
        self, data: "NDArray", sr: float, sensitivity: float
    ) -> List[float]:
        """Detect onset times in audio.

        Args:
            data: Audio data
            sr: Sample rate
            sensitivity: Sensitivity (0-1)

        Returns:
            List of onset times in seconds
        """
        # Simplified onset detection using spectral flux
        hop_size = 512
        window_size = 2048

        # Compute spectrogram
        num_frames = (len(data) - window_size) // hop_size + 1
        spectral_flux = []
        prev_spectrum = None

        for i in range(num_frames):
            start = i * hop_size
            end = start + window_size

            if end > len(data):
                break

            window = data[start:end] * np.hanning(window_size)
            spectrum = np.abs(np.fft.rfft(window))

            if prev_spectrum is not None:
                flux = np.sum(np.maximum(0, spectrum - prev_spectrum))
                spectral_flux.append(flux)

            prev_spectrum = spectrum

        spectral_flux_array = np.array(spectral_flux)

        # Threshold based on sensitivity
        threshold = np.percentile(spectral_flux_array, 100 * (1.0 - sensitivity))
        onset_frames = np.where(spectral_flux_array > threshold)[0]

        # Convert to time
        onset_times = onset_frames * hop_size / sr

        return onset_times.tolist()  # type: ignore[no-any-return]

    def _filter_onsets(
        self, onsets: List[float], min_duration: float
    ) -> List[float]:
        """Filter onsets by minimum duration.

        Args:
            onsets: List of onset times
            min_duration: Minimum duration between onsets

        Returns:
            Filtered list of onset times
        """
        if not onsets:
            return []

        filtered = [onsets[0]]

        for onset in onsets[1:]:
            if onset - filtered[-1] >= min_duration:
                filtered.append(onset)

        return filtered

    def _select_prominent_onsets(
        self, onsets: List[float], data: "NDArray", sr: float, max_slices: int
    ) -> List[float]:
        """Select most prominent onsets.

        Args:
            onsets: List of onset times
            data: Audio data
            sr: Sample rate
            max_slices: Maximum number to select

        Returns:
            Selected onset times
        """
        # Calculate onset strength for each onset
        onset_strengths = []
        window_size = int(0.05 * sr)  # 50ms window

        for onset in onsets:
            sample = int(onset * sr)
            start = max(0, sample - window_size // 2)
            end = min(len(data), sample + window_size // 2)

            segment = data[start:end]
            strength = float(np.max(np.abs(segment)))
            onset_strengths.append(strength)

        # Select top N by strength
        onset_strength_pairs = list(zip(onsets, onset_strengths))
        onset_strength_pairs.sort(key=lambda x: x[1], reverse=True)

        selected_onsets = [onset for onset, _ in onset_strength_pairs[:max_slices]]
        selected_onsets.sort()

        return selected_onsets

    # ========================================================================
    # Transient-based Slicing
    # ========================================================================

    def _detect_transient_slices(
        self, window_size: float = 0.02, threshold_db: float = -40.0
    ) -> List[Slice]:
        """Detect slices based on transient detection.

        Args:
            window_size: Analysis window size in seconds
            threshold_db: Threshold in dB relative to peak

        Returns:
            List of slices based on detected transients
        """
        data, sr = self._load_audio()

        # Calculate envelope
        window_samples = int(window_size * sr)
        envelope = np.abs(data)
        envelope = np.convolve(
            envelope, np.ones(window_samples) / window_samples, mode="same"
        )

        # Convert to dB
        envelope_db = 20 * np.log10(envelope + 1e-10)
        peak_db = float(np.max(envelope_db))

        # Adjust threshold with sensitivity
        adjusted_threshold = peak_db + threshold_db * (1.0 - self.sensitivity)

        # Find transients (envelope crosses threshold)
        transient_indices = np.where(
            np.diff((envelope_db > adjusted_threshold).astype(int)) > 0
        )[0]

        # Convert to time positions
        transient_times = (transient_indices / sr).tolist()

        # Create slices
        slices = self._create_slices_from_times(transient_times, data, sr)

        self._slices = slices
        return slices

    # ========================================================================
    # Zero-crossing Slicing
    # ========================================================================

    def _detect_zero_crossing_slices(
        self, target_slices: int = 16, snap_to_zero: bool = True
    ) -> List[Slice]:
        """Detect slices at zero-crossings (for glitch-free slicing).

        Args:
            target_slices: Target number of slices
            snap_to_zero: If True, snap slice boundaries to zero crossings

        Returns:
            List of slices at zero crossing points
        """
        data, sr = self._load_audio()

        # Calculate ideal slice positions
        total_duration = len(data) / sr
        ideal_slice_duration = total_duration / target_slices
        ideal_times = [i * ideal_slice_duration for i in range(target_slices + 1)]

        if snap_to_zero:
            # Find nearest zero crossings to ideal positions
            slice_times = []
            for ideal_time in ideal_times:
                ideal_sample = int(ideal_time * sr)
                # Search window around ideal position
                search_window = int(0.01 * sr)  # 10ms search window
                start_search = max(0, ideal_sample - search_window)
                end_search = min(len(data), ideal_sample + search_window)

                # Find zero crossing
                segment = data[start_search:end_search]
                zero_crossings = np.where(np.diff(np.sign(segment)))[0]

                if len(zero_crossings) > 0:
                    # Choose closest zero crossing
                    closest_idx = zero_crossings[
                        np.argmin(np.abs(zero_crossings - search_window))
                    ]
                    actual_sample = start_search + closest_idx
                else:
                    actual_sample = ideal_sample

                slice_times.append(actual_sample / sr)
        else:
            slice_times = ideal_times

        # Create slices
        slices = []
        for i in range(len(slice_times) - 1):
            start_time = slice_times[i]
            end_time = slice_times[i + 1]

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            slice_data = data[start_sample:end_sample]

            slices.append(
                Slice(
                    start=start_time,
                    end=end_time,
                    data=slice_data,
                    sample_rate=sr,
                    index=i,
                )
            )

        self._slices = slices
        return slices

    # ========================================================================
    # Grid-based Slicing
    # ========================================================================

    def _detect_grid_slices(
        self,
        divisions: int = 16,
        tempo: Optional[float] = None,
        time_signature: Tuple[int, int] = (4, 4),
    ) -> List[Slice]:
        """Slice audio on a regular grid.

        Args:
            divisions: Number of equal divisions
            tempo: Optional tempo in BPM for beat-aligned slicing
            time_signature: Time signature (numerator, denominator)

        Returns:
            List of equal-duration slices
        """
        data, sr = self._load_audio()

        if tempo is not None:
            # Beat-aligned slicing
            beat_duration = 60.0 / tempo
            bar_duration = beat_duration * time_signature[0]
            total_duration = len(data) / sr

            # Calculate number of bars
            num_bars = int(np.ceil(total_duration / bar_duration))
            slice_duration = bar_duration / (divisions / time_signature[0])

            slice_times = np.arange(0, num_bars * bar_duration, slice_duration)
        else:
            # Simple equal divisions
            total_duration = len(data) / sr
            slice_times = np.linspace(0, total_duration, divisions + 1)

        # Create slices
        slices = []
        for i in range(len(slice_times) - 1):
            start_time = float(slice_times[i])
            end_time = float(slice_times[i + 1])

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            # Ensure we don't exceed array bounds
            end_sample = min(end_sample, len(data))

            slice_data = data[start_sample:end_sample]

            slices.append(
                Slice(
                    start=start_time,
                    end=end_time,
                    data=slice_data,
                    sample_rate=sr,
                    index=i,
                )
            )

        self._slices = slices
        return slices

    # ========================================================================
    # Manual Slicing
    # ========================================================================

    def _detect_manual_slices(self, slice_points: List[float]) -> List[Slice]:
        """Create slices at manually specified time points.

        Args:
            slice_points: List of time points in seconds

        Returns:
            List of slices at specified positions
        """
        data, sr = self._load_audio()

        # Ensure slice points are sorted
        slice_points = sorted(slice_points)

        # Add start and end if not present
        if not slice_points or slice_points[0] != 0:
            slice_points.insert(0, 0)
        total_duration = len(data) / sr
        if slice_points[-1] != total_duration:
            slice_points.append(total_duration)

        # Create slices
        slices = self._create_slices_from_times(slice_points, data, sr)

        self._slices = slices
        return slices

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _create_slices_from_times(
        self,
        slice_times: List[float],
        data: "NDArray",
        sr: float,
        confidence: float = 1.0,
    ) -> List[Slice]:
        """Create Slice objects from time points.

        Args:
            slice_times: List of time points
            data: Audio data
            sr: Sample rate
            confidence: Detection confidence

        Returns:
            List of Slice objects
        """
        slices = []
        for i in range(len(slice_times) - 1):
            start_time = slice_times[i]
            end_time = slice_times[i + 1]

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            slice_data = data[start_sample:end_sample]

            slices.append(
                Slice(
                    start=start_time,
                    end=end_time,
                    data=slice_data,
                    sample_rate=sr,
                    index=i,
                    confidence=confidence,
                )
            )

        # Handle last slice if needed
        if slice_times:
            start_time = slice_times[-1]
            end_time = len(data) / sr

            if end_time > start_time:
                start_sample = int(start_time * sr)
                slice_data = data[start_sample:]

                slices.append(
                    Slice(
                        start=start_time,
                        end=end_time,
                        data=slice_data,
                        sample_rate=sr,
                        index=len(slices),
                        confidence=confidence,
                    )
                )

        return slices

    @property
    def slices(self) -> List[Slice]:
        """Get current slices (detect if not already done).

        Returns:
            List of slices
        """
        if self._slices is None:
            self.detect_slices()
        return self._slices if self._slices is not None else []

    def export_slices(
        self,
        output_dir: str,
        name_template: str = "slice_{index:03d}.wav",
        format: str = "wav",
    ) -> List[str]:
        """Export all slices as individual files.

        Args:
            output_dir: Output directory
            name_template: Filename template (use {index} for slice number)
            format: Audio format

        Returns:
            List of exported file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported_files = []
        for slice_obj in self.slices:
            filename = name_template.format(index=slice_obj.index)
            filepath = output_path / filename

            slice_obj.export(str(filepath), format=format)
            exported_files.append(str(filepath))

        return exported_files


# ============================================================================
# SliceCollection Class
# ============================================================================


class SliceCollection:
    """Collection of audio slices with manipulation methods.

    Provides fluent API for manipulating collections of slices.

    Example:
        >>> collection = SliceCollection(slices)
        >>> shuffled = collection.shuffle()
        >>> pattern = collection.apply_pattern([0, 1, 2, 1, 0])
    """

    def __init__(self, slices: List[Slice]):
        """Initialize slice collection.

        Args:
            slices: List of Slice objects
        """
        self.slices = slices

    def shuffle(self) -> "SliceCollection":
        """Shuffle slices randomly.

        Returns:
            New SliceCollection with shuffled slices
        """
        shuffled = self.slices.copy()
        random.shuffle(shuffled)
        return SliceCollection(shuffled)

    def reverse(self) -> "SliceCollection":
        """Reverse slice order.

        Returns:
            New SliceCollection with reversed slices
        """
        return SliceCollection(self.slices[::-1])

    def repeat(self, times: int) -> "SliceCollection":
        """Repeat entire sequence.

        Args:
            times: Number of times to repeat

        Returns:
            New SliceCollection with repeated slices
        """
        return SliceCollection(self.slices * times)

    def filter(self, predicate: Callable[[Slice], bool]) -> "SliceCollection":
        """Filter slices by predicate.

        Args:
            predicate: Function that returns True to keep slice

        Returns:
            New SliceCollection with filtered slices
        """
        filtered = [s for s in self.slices if predicate(s)]
        return SliceCollection(filtered)

    def sort_by_duration(self, reverse: bool = False) -> "SliceCollection":
        """Sort slices by duration.

        Args:
            reverse: If True, sort descending

        Returns:
            New SliceCollection with sorted slices
        """
        sorted_slices = sorted(
            self.slices, key=lambda s: s.duration, reverse=reverse
        )
        return SliceCollection(sorted_slices)

    def select(self, indices: List[int]) -> "SliceCollection":
        """Select specific slices by index.

        Args:
            indices: List of indices to select

        Returns:
            New SliceCollection with selected slices
        """
        selected = [
            self.slices[i] for i in indices if 0 <= i < len(self.slices)
        ]
        return SliceCollection(selected)

    def apply_pattern(self, pattern: List[int]) -> "SliceCollection":
        """Apply pattern (e.g., [0, 1, 2, 1] to repeat certain slices).

        Args:
            pattern: List of slice indices

        Returns:
            New SliceCollection with pattern applied
        """
        patterned = [self.slices[i % len(self.slices)] for i in pattern]
        return SliceCollection(patterned)

    def __len__(self) -> int:
        """Get number of slices."""
        return len(self.slices)

    def __getitem__(self, index: int) -> Slice:
        """Get slice by index."""
        return self.slices[index]


# ============================================================================
# SliceRecombinator Class
# ============================================================================


class SliceRecombinator:
    """Recombine slices into new audio files.

    Supports multiple recombination methods with crossfading.

    Example:
        >>> recombinator = SliceRecombinator(slices)
        >>> audio = recombinator.recombine(method="random")
        >>> recombinator.export("output.wav", method="reverse")
    """

    def __init__(self, slices: Union[List[Slice], SliceCollection]):
        """Initialize recombinator.

        Args:
            slices: List of slices or SliceCollection
        """
        if isinstance(slices, SliceCollection):
            self.slices = slices.slices
        else:
            self.slices = slices

    def recombine(
        self,
        method: RecombineMethod = "original",
        crossfade_duration: float = 0.01,
        normalize: bool = True,
        **kwargs: Any,
    ) -> "NDArray":
        """Recombine slices into continuous audio.

        Args:
            method: Recombination method
            crossfade_duration: Crossfade between slices in seconds
            normalize: Normalize output audio
            **kwargs: Method-specific parameters

        Returns:
            Recombined audio data

        Raises:
            ValueError: If unknown recombination method
        """
        if method == "original":
            return self._recombine_original(crossfade_duration, normalize)
        elif method == "random":
            return self._recombine_random(crossfade_duration, normalize, **kwargs)
        elif method == "reverse":
            return self._recombine_reverse(crossfade_duration, normalize)
        elif method == "pattern":
            return self._recombine_pattern(crossfade_duration, normalize, **kwargs)
        elif method == "custom":
            return self._recombine_custom(crossfade_duration, normalize, **kwargs)
        else:
            raise ValueError(f"Unknown recombination method: {method}")

    def _recombine_original(
        self, crossfade_duration: float, normalize: bool
    ) -> "NDArray":
        """Recombine in original order."""
        return self._concatenate_slices(self.slices, crossfade_duration, normalize)

    def _recombine_random(
        self,
        crossfade_duration: float,
        normalize: bool,
        num_slices: Optional[int] = None,
    ) -> "NDArray":
        """Recombine in random order.

        Args:
            num_slices: Number of slices to use (None = all, with replacement)
        """
        if num_slices is None:
            num_slices = len(self.slices)

        random_slices = random.choices(self.slices, k=num_slices)
        return self._concatenate_slices(random_slices, crossfade_duration, normalize)

    def _recombine_reverse(
        self, crossfade_duration: float, normalize: bool
    ) -> "NDArray":
        """Recombine in reverse order."""
        return self._concatenate_slices(
            self.slices[::-1], crossfade_duration, normalize
        )

    def _recombine_pattern(
        self,
        crossfade_duration: float,
        normalize: bool,
        pattern: Optional[List[int]] = None,
    ) -> "NDArray":
        """Recombine using pattern.

        Args:
            pattern: List of slice indices (e.g., [0, 1, 2, 1, 0])

        Raises:
            ValueError: If pattern not provided
        """
        if pattern is None:
            raise ValueError("pattern parameter required for pattern method")

        patterned_slices = [self.slices[i % len(self.slices)] for i in pattern]
        return self._concatenate_slices(patterned_slices, crossfade_duration, normalize)

    def _recombine_custom(
        self,
        crossfade_duration: float,
        normalize: bool,
        order_func: Optional[Callable[[List[Slice]], List[Slice]]] = None,
    ) -> "NDArray":
        """Recombine using custom ordering function.

        Args:
            order_func: Function that takes list of slices and returns reordered list

        Raises:
            ValueError: If order_func not provided
        """
        if order_func is None:
            raise ValueError("order_func parameter required for custom method")

        ordered_slices = order_func(self.slices)
        return self._concatenate_slices(ordered_slices, crossfade_duration, normalize)

    def _concatenate_slices(
        self, slices: List[Slice], crossfade_duration: float, normalize: bool
    ) -> "NDArray":
        """Concatenate slices with crossfading.

        Args:
            slices: List of slices to concatenate
            crossfade_duration: Crossfade duration in seconds
            normalize: Whether to normalize output

        Returns:
            Concatenated audio data
        """
        if not slices:
            return np.array([])

        # Get sample rate (assume all slices have same rate)
        sr = slices[0].sample_rate
        crossfade_samples = int(crossfade_duration * sr)

        # Calculate total length
        total_samples = sum(len(s.data) for s in slices)
        # Subtract overlaps
        total_samples -= crossfade_samples * (len(slices) - 1)

        # Prepare output buffer
        output = np.zeros(total_samples)

        current_pos = 0
        for i, slice_obj in enumerate(slices):
            slice_data = slice_obj.data

            if i > 0 and crossfade_samples > 0:
                # Apply crossfade with previous slice
                overlap_start = current_pos - crossfade_samples
                overlap_end = current_pos

                # Create fade curves
                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)

                # Apply crossfade
                output[overlap_start:overlap_end] *= fade_out
                output[overlap_start:overlap_end] += (
                    slice_data[:crossfade_samples] * fade_in
                )

                # Add rest of slice
                slice_start = crossfade_samples
                slice_end = len(slice_data)
                output[current_pos : current_pos + slice_end - slice_start] = (
                    slice_data[slice_start:]
                )

                current_pos += slice_end - slice_start
            else:
                # No crossfade for first slice
                output[current_pos : current_pos + len(slice_data)] = slice_data
                current_pos += len(slice_data)

        if normalize:
            # Normalize to -1 to 1 range
            max_val = np.max(np.abs(output))
            if max_val > 0:
                output = output / max_val

        return output

    def export(
        self,
        output_path: str,
        method: RecombineMethod = "original",
        format: str = "wav",
        **kwargs: Any,
    ) -> None:
        """Recombine and export to file.

        Args:
            output_path: Output file path
            method: Recombination method
            format: Audio format
            **kwargs: Method-specific parameters

        Note:
            This is a placeholder - full write support would require
            extending AudioFile with write capabilities.
        """
        audio_data = self.recombine(method, **kwargs)

        logger.info(
            f"Export recombined audio to {output_path} "
            f"(method: {method}, duration: {len(audio_data) / self.slices[0].sample_rate:.3f}s)"
        )
        # Placeholder for actual file writing
        # In full implementation, would use cm.AudioFile in write mode


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "Slice",
    "AudioSlicer",
    "SliceCollection",
    "SliceRecombinator",
    "SliceMethod",
    "RecombineMethod",
]
