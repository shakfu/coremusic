#!/usr/bin/env python3
"""Tests for audio slicing and recombination module."""

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
    from scipy import signal

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

if NUMPY_AVAILABLE and SCIPY_AVAILABLE:
    from coremusic.audio.slicing import (
        Slice,
        AudioSlicer,
        SliceCollection,
        SliceRecombinator,
        SliceMethod,
        RecombineMethod,
    )

pytestmark = pytest.mark.skipif(
    not (NUMPY_AVAILABLE and SCIPY_AVAILABLE),
    reason="NumPy and SciPy required for audio slicing tests",
)


class TestSliceDataclass:
    """Tests for Slice dataclass."""

    def test_slice_creation(self):
        """Test creating a Slice."""
        data = np.random.randn(4800)  # 0.1 seconds at 48000 Hz
        slice_obj = Slice(
            start=0.0,
            end=0.1,
            data=data,
            sample_rate=48000.0,
            index=0,
            confidence=0.95,
        )

        assert slice_obj.start == 0.0
        assert slice_obj.end == 0.1
        assert np.array_equal(slice_obj.data, data)
        assert slice_obj.sample_rate == 48000.0
        assert slice_obj.index == 0
        assert slice_obj.confidence == 0.95

    def test_slice_duration_property(self):
        """Test duration property calculation."""
        data = np.random.randn(8820)  # 0.2 seconds at 48000 Hz
        slice_obj = Slice(
            start=1.0, end=1.2, data=data, sample_rate=48000.0, index=1
        )

        assert abs(slice_obj.duration - 0.2) < 0.001

    def test_slice_num_samples_property(self):
        """Test num_samples property."""
        data = np.random.randn(2048)
        slice_obj = Slice(
            start=0.0, end=0.046, data=data, sample_rate=48000.0, index=0
        )

        assert slice_obj.num_samples == 2048

    def test_slice_default_confidence(self):
        """Test default confidence value."""
        data = np.random.randn(1000)
        slice_obj = Slice(start=0.0, end=0.1, data=data, sample_rate=48000.0, index=0)

        assert slice_obj.confidence == 1.0

    def test_slice_export_placeholder(self):
        """Test export method (placeholder)."""
        data = np.random.randn(1000)
        slice_obj = Slice(start=0.0, end=0.1, data=data, sample_rate=48000.0, index=0)

        # Should not raise an error
        result = slice_obj.export("output.wav")
        assert result is None


class TestAudioSlicer:
    """Tests for AudioSlicer class."""

    def test_create_slicer_onset(self):
        """Test creating AudioSlicer with onset detection."""
        slicer = AudioSlicer(AMEN_WAV_PATH, method="onset", sensitivity=0.5)

        assert slicer.method == "onset"
        assert slicer.sensitivity == 0.5
        assert slicer.audio_file == AMEN_WAV_PATH

    def test_create_slicer_transient(self):
        """Test creating AudioSlicer with transient detection."""
        slicer = AudioSlicer(AMEN_WAV_PATH, method="transient", sensitivity=0.7)

        assert slicer.method == "transient"
        assert slicer.sensitivity == 0.7

    def test_create_slicer_zero_crossing(self):
        """Test creating AudioSlicer with zero-crossing method."""
        slicer = AudioSlicer(AMEN_WAV_PATH, method="zero_crossing")

        assert slicer.method == "zero_crossing"

    def test_create_slicer_grid(self):
        """Test creating AudioSlicer with grid method."""
        slicer = AudioSlicer(AMEN_WAV_PATH, method="grid")

        assert slicer.method == "grid"

    def test_create_slicer_manual(self):
        """Test creating AudioSlicer with manual method."""
        slicer = AudioSlicer(AMEN_WAV_PATH, method="manual")

        assert slicer.method == "manual"

    def test_detect_onset_slices(self):
        """Test onset detection slicing."""
        slicer = AudioSlicer(AMEN_WAV_PATH, method="onset", sensitivity=0.5)
        slices = slicer.detect_slices(min_slice_duration=0.05)

        assert len(slices) > 0
        assert all(isinstance(s, Slice) for s in slices)
        assert all(s.duration >= 0.05 for s in slices)
        assert all(s.index == i for i, s in enumerate(slices))

    def test_onset_slices_ordered(self):
        """Test that onset slices are in temporal order."""
        slicer = AudioSlicer(AMEN_WAV_PATH, method="onset")
        slices = slicer.detect_slices()

        for i in range(len(slices) - 1):
            assert slices[i].start <= slices[i + 1].start
            assert slices[i].end <= slices[i + 1].start

    def test_onset_slices_max_slices(self):
        """Test limiting maximum number of onset slices."""
        slicer = AudioSlicer(AMEN_WAV_PATH, method="onset", sensitivity=0.5)
        slices = slicer.detect_slices(max_slices=5)

        assert len(slices) <= 5

    def test_detect_transient_slices(self):
        """Test transient detection slicing."""
        slicer = AudioSlicer(AMEN_WAV_PATH, method="transient", sensitivity=0.5)
        slices = slicer.detect_slices(window_size=0.02, threshold_db=-40.0)

        assert len(slices) > 0
        assert all(isinstance(s, Slice) for s in slices)
        assert all(s.duration > 0 for s in slices)

    def test_transient_slices_sensitivity(self):
        """Test transient detection with different sensitivities."""
        slicer_low = AudioSlicer(AMEN_WAV_PATH, method="transient", sensitivity=0.2)
        slicer_high = AudioSlicer(AMEN_WAV_PATH, method="transient", sensitivity=0.8)

        slices_low = slicer_low.detect_slices()
        slices_high = slicer_high.detect_slices()

        # Higher sensitivity should detect more slices
        assert len(slices_high) >= len(slices_low)

    def test_detect_zero_crossing_slices(self):
        """Test zero-crossing detection slicing."""
        slicer = AudioSlicer(AMEN_WAV_PATH, method="zero_crossing")
        slices = slicer.detect_slices(target_slices=16, snap_to_zero=True)

        assert len(slices) == 16
        assert all(isinstance(s, Slice) for s in slices)

    def test_zero_crossing_without_snap(self):
        """Test zero-crossing detection without snapping."""
        slicer = AudioSlicer(AMEN_WAV_PATH, method="zero_crossing")
        slices = slicer.detect_slices(target_slices=8, snap_to_zero=False)

        assert len(slices) == 8
        # Slices should be roughly equal duration
        durations = [s.duration for s in slices]
        avg_duration = np.mean(durations)
        assert all(abs(d - avg_duration) < avg_duration * 0.5 for d in durations)

    def test_detect_grid_slices_simple(self):
        """Test simple grid-based slicing."""
        slicer = AudioSlicer(AMEN_WAV_PATH, method="grid")
        slices = slicer.detect_slices(divisions=8)

        assert len(slices) == 8
        assert all(isinstance(s, Slice) for s in slices)

        # Check roughly equal durations
        durations = [s.duration for s in slices]
        avg_duration = np.mean(durations)
        assert all(abs(d - avg_duration) < 0.01 for d in durations)

    def test_detect_grid_slices_with_tempo(self):
        """Test grid slicing with tempo alignment."""
        slicer = AudioSlicer(AMEN_WAV_PATH, method="grid")
        slices = slicer.detect_slices(divisions=16, tempo=165.0, time_signature=(4, 4))

        # With tempo, slicing is beat-aligned so number of slices depends on
        # audio duration and beat timing (not exactly equal to divisions)
        assert len(slices) > 0
        assert all(isinstance(s, Slice) for s in slices)

    def test_detect_manual_slices(self):
        """Test manual slicing with specified points."""
        slicer = AudioSlicer(AMEN_WAV_PATH, method="manual")
        slice_points = [0.0, 0.5, 1.0, 1.5, 2.0]
        slices = slicer.detect_slices(slice_points=slice_points)

        # Manual slicing automatically adds the end of audio, so we get
        # slices between consecutive points plus final segment to end
        assert len(slices) == 5

        # Check slice boundaries
        assert abs(slices[0].start - 0.0) < 0.01
        assert abs(slices[0].end - 0.5) < 0.01
        assert abs(slices[1].start - 0.5) < 0.01
        assert abs(slices[1].end - 1.0) < 0.01

    def test_export_slices(self):
        """Test exporting slices (placeholder)."""
        slicer = AudioSlicer(AMEN_WAV_PATH, method="grid")
        slices = slicer.detect_slices(divisions=4)

        # Should not raise an error
        output_paths = slicer.export_slices(
            output_dir="/tmp/test_slices", name_template="slice_{index:03d}.wav"
        )

        assert isinstance(output_paths, list)
        assert len(output_paths) == 4


class TestSliceCollection:
    """Tests for SliceCollection class."""

    def test_create_collection(self):
        """Test creating a SliceCollection."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.random.randn(4800),
                sample_rate=48000.0,
                index=i,
            )
            for i in range(5)
        ]
        collection = SliceCollection(slices)

        assert len(collection.slices) == 5
        assert collection.slices == slices

    def test_collection_len(self):
        """Test __len__ method."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.random.randn(4800),
                sample_rate=48000.0,
                index=i,
            )
            for i in range(3)
        ]
        collection = SliceCollection(slices)

        assert len(collection) == 3

    def test_collection_getitem(self):
        """Test __getitem__ method."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.random.randn(4800),
                sample_rate=48000.0,
                index=i,
            )
            for i in range(4)
        ]
        collection = SliceCollection(slices)

        assert collection[0] == slices[0]
        assert collection[2] == slices[2]
        assert collection[-1] == slices[-1]

    def test_collection_iter(self):
        """Test __iter__ method."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.random.randn(4800),
                sample_rate=48000.0,
                index=i,
            )
            for i in range(3)
        ]
        collection = SliceCollection(slices)

        iterated = list(collection)
        assert iterated == slices

    def test_shuffle(self):
        """Test shuffle method."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.full(4800, i, dtype=np.float32),
                sample_rate=48000.0,
                index=i,
            )
            for i in range(10)
        ]
        collection = SliceCollection(slices)

        shuffled = collection.shuffle()

        assert isinstance(shuffled, SliceCollection)
        assert len(shuffled) == len(collection)
        # With 10 slices, probability of same order is very low
        assert shuffled.slices != collection.slices

    def test_reverse(self):
        """Test reverse method."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.full(4800, i, dtype=np.float32),
                sample_rate=48000.0,
                index=i,
            )
            for i in range(5)
        ]
        collection = SliceCollection(slices)

        reversed_collection = collection.reverse()

        assert isinstance(reversed_collection, SliceCollection)
        assert len(reversed_collection) == len(collection)
        assert reversed_collection[0].index == 4
        assert reversed_collection[4].index == 0

    def test_repeat(self):
        """Test repeat method."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.random.randn(4800),
                sample_rate=48000.0,
                index=i,
            )
            for i in range(3)
        ]
        collection = SliceCollection(slices)

        repeated = collection.repeat(3)

        assert isinstance(repeated, SliceCollection)
        assert len(repeated) == 9
        # Check pattern repeats
        for i in range(9):
            assert repeated[i].index == slices[i % 3].index

    def test_filter(self):
        """Test filter method."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.random.randn(4800),
                sample_rate=48000.0,
                index=i,
                confidence=0.5 + i * 0.1,
            )
            for i in range(5)
        ]
        collection = SliceCollection(slices)

        filtered = collection.filter(lambda s: s.confidence > 0.7)

        assert isinstance(filtered, SliceCollection)
        assert len(filtered) < len(collection)
        assert all(s.confidence > 0.7 for s in filtered)

    def test_sort_by_duration(self):
        """Test sort_by_duration method."""
        slices = [
            Slice(
                start=0.0,
                end=duration,
                data=np.random.randn(int(48000 * duration)),
                sample_rate=48000.0,
                index=i,
            )
            for i, duration in enumerate([0.2, 0.1, 0.3, 0.15])
        ]
        collection = SliceCollection(slices)

        sorted_collection = collection.sort_by_duration()

        durations = [s.duration for s in sorted_collection]
        assert durations == sorted(durations)

    def test_sort_by_duration_reverse(self):
        """Test sort_by_duration with reverse."""
        slices = [
            Slice(
                start=0.0,
                end=duration,
                data=np.random.randn(int(48000 * duration)),
                sample_rate=48000.0,
                index=i,
            )
            for i, duration in enumerate([0.2, 0.1, 0.3, 0.15])
        ]
        collection = SliceCollection(slices)

        sorted_collection = collection.sort_by_duration(reverse=True)

        durations = [s.duration for s in sorted_collection]
        assert durations == sorted(durations, reverse=True)

    def test_select(self):
        """Test select method."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.random.randn(4800),
                sample_rate=48000.0,
                index=i,
            )
            for i in range(10)
        ]
        collection = SliceCollection(slices)

        selected = collection.select([0, 2, 5, 9])

        assert isinstance(selected, SliceCollection)
        assert len(selected) == 4
        assert selected[0].index == 0
        assert selected[1].index == 2
        assert selected[2].index == 5
        assert selected[3].index == 9

    def test_apply_pattern(self):
        """Test apply_pattern method."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.full(4800, i, dtype=np.float32),
                sample_rate=48000.0,
                index=i,
            )
            for i in range(4)
        ]
        collection = SliceCollection(slices)

        patterned = collection.apply_pattern([0, 1, 2, 1, 0, 3])

        assert isinstance(patterned, SliceCollection)
        assert len(patterned) == 6
        assert patterned[0].index == 0
        assert patterned[1].index == 1
        assert patterned[2].index == 2
        assert patterned[3].index == 1
        assert patterned[4].index == 0
        assert patterned[5].index == 3

    def test_chaining_operations(self):
        """Test chaining multiple operations."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.random.randn(4800),
                sample_rate=48000.0,
                index=i,
            )
            for i in range(10)
        ]
        collection = SliceCollection(slices)

        # Chain: select -> repeat -> reverse
        result = collection.select([0, 2, 4]).repeat(2).reverse()

        assert isinstance(result, SliceCollection)
        assert len(result) == 6  # 3 selected * 2 repeated


class TestSliceRecombinator:
    """Tests for SliceRecombinator class."""

    def test_create_recombinator(self):
        """Test creating a SliceRecombinator."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.random.randn(4800),
                sample_rate=48000.0,
                index=i,
            )
            for i in range(5)
        ]
        collection = SliceCollection(slices)
        recombinator = SliceRecombinator(collection)

        # SliceRecombinator extracts slices list from collection
        assert recombinator.slices == slices

    def test_recombine_original(self):
        """Test recombining in original order."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.full(4800, i, dtype=np.float32),
                sample_rate=48000.0,
                index=i,
            )
            for i in range(5)
        ]
        collection = SliceCollection(slices)
        recombinator = SliceRecombinator(collection)

        result = recombinator.recombine(method="original", crossfade_duration=0.0)

        assert isinstance(result, np.ndarray)
        # Should be approximately sum of slice durations
        expected_samples = sum(s.num_samples for s in slices)
        assert abs(len(result) - expected_samples) < 1000

    def test_recombine_with_crossfade(self):
        """Test recombining with crossfade."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.random.randn(4800),
                sample_rate=48000.0,
                index=i,
            )
            for i in range(3)
        ]
        collection = SliceCollection(slices)
        recombinator = SliceRecombinator(collection)

        result = recombinator.recombine(method="original", crossfade_duration=0.01)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_recombine_random(self):
        """Test random recombination."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.full(4800, i, dtype=np.float32),
                sample_rate=48000.0,
                index=i,
            )
            for i in range(5)
        ]
        collection = SliceCollection(slices)
        recombinator = SliceRecombinator(collection)

        result = recombinator.recombine(method="random", crossfade_duration=0.0)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_recombine_random_with_num_slices(self):
        """Test random recombination with specified number of slices."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.random.randn(4800),
                sample_rate=48000.0,
                index=i,
            )
            for i in range(10)
        ]
        collection = SliceCollection(slices)
        recombinator = SliceRecombinator(collection)

        result = recombinator.recombine(
            method="random", crossfade_duration=0.0, num_slices=3
        )

        assert isinstance(result, np.ndarray)
        # Should use approximately 3 slices
        expected_samples = 3 * 4800
        assert abs(len(result) - expected_samples) < 5000

    def test_recombine_reverse(self):
        """Test reverse recombination."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.full(4800, i, dtype=np.float32),
                sample_rate=48000.0,
                index=i,
            )
            for i in range(4)
        ]
        collection = SliceCollection(slices)
        recombinator = SliceRecombinator(collection)

        result = recombinator.recombine(method="reverse", crossfade_duration=0.0)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_recombine_pattern(self):
        """Test pattern-based recombination."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.full(4800, i, dtype=np.float32),
                sample_rate=48000.0,
                index=i,
            )
            for i in range(4)
        ]
        collection = SliceCollection(slices)
        recombinator = SliceRecombinator(collection)

        result = recombinator.recombine(
            method="pattern", crossfade_duration=0.0, pattern=[0, 2, 1, 3, 0]
        )

        assert isinstance(result, np.ndarray)
        # Should use 5 slices based on pattern
        expected_samples = 5 * 4800
        assert abs(len(result) - expected_samples) < 1000

    def test_recombine_custom(self):
        """Test custom recombination with order function."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.random.randn(4800),
                sample_rate=48000.0,
                index=i,
                confidence=0.5 + i * 0.1,
            )
            for i in range(5)
        ]
        collection = SliceCollection(slices)
        recombinator = SliceRecombinator(collection)

        # Custom function: sort by confidence descending
        def by_confidence(slices_list):
            return sorted(slices_list, key=lambda s: s.confidence, reverse=True)

        result = recombinator.recombine(
            method="custom", crossfade_duration=0.0, order_func=by_confidence
        )

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_recombine_with_normalization(self):
        """Test recombination with normalization."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.random.randn(4800) * 2.0,  # Louder data
                sample_rate=48000.0,
                index=i,
            )
            for i in range(3)
        ]
        collection = SliceCollection(slices)
        recombinator = SliceRecombinator(collection)

        result = recombinator.recombine(
            method="original", crossfade_duration=0.0, normalize=True
        )

        # Should be normalized to -1 to 1 range
        assert np.max(np.abs(result)) <= 1.0
        # Should actually reach near max (within some margin)
        assert np.max(np.abs(result)) > 0.9

    def test_recombine_without_normalization(self):
        """Test recombination without normalization."""
        # Use seeded random for reproducibility
        rng = np.random.default_rng(42)
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=rng.standard_normal(4800) * 0.1,  # Quieter data
                sample_rate=48000.0,
                index=i,
            )
            for i in range(3)
        ]
        collection = SliceCollection(slices)
        recombinator = SliceRecombinator(collection)

        result = recombinator.recombine(
            method="original", crossfade_duration=0.0, normalize=False
        )

        # Should maintain original amplitude (with tolerance for random variation)
        assert np.max(np.abs(result)) < 1.0

    def test_export_placeholder(self):
        """Test export method (placeholder)."""
        slices = [
            Slice(
                start=i * 0.1,
                end=(i + 1) * 0.1,
                data=np.random.randn(4800),
                sample_rate=48000.0,
                index=i,
            )
            for i in range(3)
        ]
        collection = SliceCollection(slices)
        recombinator = SliceRecombinator(collection)

        # Should not raise an error
        recombinator.export("/tmp/test_recombined.wav", method="original")


class TestIntegrationWorkflows:
    """Integration tests demonstrating complete workflows."""

    def test_onset_slice_and_shuffle_workflow(self):
        """Test: onset slice audio and shuffle slices."""
        # Slice using onset detection
        slicer = AudioSlicer(AMEN_WAV_PATH, method="onset", sensitivity=0.5)
        slices = slicer.detect_slices(min_slice_duration=0.05)

        # Create collection and shuffle
        collection = SliceCollection(slices)
        shuffled = collection.shuffle()

        # Recombine
        recombinator = SliceRecombinator(shuffled)
        result = recombinator.recombine(crossfade_duration=0.005)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_grid_slice_and_pattern_workflow(self):
        """Test: grid slice and apply pattern."""
        # Slice into regular grid
        slicer = AudioSlicer(AMEN_WAV_PATH, method="grid")
        slices = slicer.detect_slices(divisions=16)

        # Apply pattern
        collection = SliceCollection(slices)
        pattern = [0, 4, 8, 12, 1, 5, 9, 13]
        patterned = collection.apply_pattern(pattern)

        # Recombine
        recombinator = SliceRecombinator(patterned)
        result = recombinator.recombine(crossfade_duration=0.003)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_transient_slice_filter_repeat_workflow(self):
        """Test: transient slice, filter, and repeat."""
        # Slice using transient detection
        slicer = AudioSlicer(AMEN_WAV_PATH, method="transient", sensitivity=0.6)
        slices = slicer.detect_slices()

        # Filter and repeat
        collection = SliceCollection(slices)
        filtered = collection.filter(lambda s: s.duration > 0.05)
        repeated = filtered.select([0, 2]).repeat(4)

        # Recombine
        recombinator = SliceRecombinator(repeated)
        result = recombinator.recombine(crossfade_duration=0.01, normalize=True)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        assert np.max(np.abs(result)) <= 1.0

    def test_manual_slice_reverse_workflow(self):
        """Test: manual slice and reverse."""
        # Manual slicing at specific points
        slicer = AudioSlicer(AMEN_WAV_PATH, method="manual")
        slice_points = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        slices = slicer.detect_slices(slice_points=slice_points)

        # Reverse
        collection = SliceCollection(slices)
        reversed_collection = collection.reverse()

        # Recombine
        recombinator = SliceRecombinator(reversed_collection)
        result = recombinator.recombine(crossfade_duration=0.01)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_zero_crossing_custom_recombine_workflow(self):
        """Test: zero-crossing slice with custom recombination."""
        # Slice at zero crossings
        slicer = AudioSlicer(AMEN_WAV_PATH, method="zero_crossing")
        slices = slicer.detect_slices(target_slices=8, snap_to_zero=True)

        # Custom recombination: sort by duration
        collection = SliceCollection(slices)
        recombinator = SliceRecombinator(collection)

        def by_duration(slices_list):
            return sorted(slices_list, key=lambda s: s.duration)

        result = recombinator.recombine(
            method="custom", order_func=by_duration, crossfade_duration=0.005
        )

        assert isinstance(result, np.ndarray)
        assert len(result) > 0


# ============================================================================
# Audio File Generation Tests
# ============================================================================


class TestAudioFileGeneration:
    """Integration tests that generate audio files to build/audio_files/.

    Each test saves both pre-transformed and post-transformed audio files
    for comparison. Files are named with the transformation applied.
    """

    @pytest.fixture
    def output_dir(self):
        """Create output directory for audio files."""
        path = Path("build/audio_files/slicing_tests")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @pytest.fixture
    def source_audio(self):
        """Load source audio file."""
        import warnings
        from scipy.io import wavfile
        from scipy.io.wavfile import WavFileWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", WavFileWarning)
            sample_rate, data = wavfile.read(AMEN_WAV_PATH)
        # Convert to float32 normalized
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        return sample_rate, data

    def _save_audio(self, path: Path, sample_rate: int, data: np.ndarray) -> None:
        """Save audio data to WAV file."""
        from scipy.io import wavfile
        # Convert back to int16 for saving
        if data.dtype == np.float32 or data.dtype == np.float64:
            data_int16 = np.clip(data * 32767, -32768, 32767).astype(np.int16)
        else:
            data_int16 = data.astype(np.int16)
        wavfile.write(str(path), sample_rate, data_int16)

    def test_generate_shuffled(self, output_dir, source_audio):
        """Generate shuffled audio - slices reordered randomly."""
        sample_rate, data = source_audio

        # Save pre-transformed
        pre_path = output_dir / "shuffled_pre.wav"
        self._save_audio(pre_path, sample_rate, data)

        # Slice and shuffle
        slicer = AudioSlicer(AMEN_WAV_PATH, method="onset", sensitivity=0.5)
        slices = slicer.detect_slices(min_slice_duration=0.05)
        collection = SliceCollection(slices)
        shuffled = collection.shuffle()  # Shuffle slices

        # Recombine
        recombinator = SliceRecombinator(shuffled)
        result = recombinator.recombine(crossfade_duration=0.005)

        # Save post-transformed
        post_path = output_dir / "shuffled_post.wav"
        self._save_audio(post_path, sample_rate, result)

        assert pre_path.exists()
        assert post_path.exists()

    def test_generate_reversed(self, output_dir, source_audio):
        """Generate reversed audio - slices in reverse order."""
        sample_rate, data = source_audio

        # Save pre-transformed
        pre_path = output_dir / "reversed_pre.wav"
        self._save_audio(pre_path, sample_rate, data)

        # Slice and reverse
        slicer = AudioSlicer(AMEN_WAV_PATH, method="grid")
        slices = slicer.detect_slices(divisions=8)
        collection = SliceCollection(slices)
        reversed_coll = collection.reverse()

        # Recombine
        recombinator = SliceRecombinator(reversed_coll)
        result = recombinator.recombine(crossfade_duration=0.005)

        # Save post-transformed
        post_path = output_dir / "reversed_post.wav"
        self._save_audio(post_path, sample_rate, result)

        assert pre_path.exists()
        assert post_path.exists()

    def test_generate_pattern(self, output_dir, source_audio):
        """Generate patterned audio - slices reordered by pattern."""
        sample_rate, data = source_audio

        # Save pre-transformed
        pre_path = output_dir / "pattern_pre.wav"
        self._save_audio(pre_path, sample_rate, data)

        # Slice and apply pattern
        slicer = AudioSlicer(AMEN_WAV_PATH, method="grid")
        slices = slicer.detect_slices(divisions=16)
        collection = SliceCollection(slices)
        # Classic breakbeat pattern: kick-snare rearrangement
        pattern = [0, 0, 4, 4, 8, 8, 12, 12, 2, 2, 6, 6, 10, 10, 14, 14]
        patterned = collection.apply_pattern(pattern)

        # Recombine
        recombinator = SliceRecombinator(patterned)
        result = recombinator.recombine(crossfade_duration=0.003)

        # Save post-transformed
        post_path = output_dir / "pattern_post.wav"
        self._save_audio(post_path, sample_rate, result)

        assert pre_path.exists()
        assert post_path.exists()

    def test_generate_repeated(self, output_dir, source_audio):
        """Generate repeated audio - selected slices repeated."""
        sample_rate, data = source_audio

        # Save pre-transformed
        pre_path = output_dir / "repeated_pre.wav"
        self._save_audio(pre_path, sample_rate, data)

        # Slice, select, and repeat
        slicer = AudioSlicer(AMEN_WAV_PATH, method="transient", sensitivity=0.6)
        slices = slicer.detect_slices()
        collection = SliceCollection(slices)
        # Select first 4 slices and repeat 4 times
        selected = collection.select([0, 1, 2, 3])
        repeated = selected.repeat(4)

        # Recombine
        recombinator = SliceRecombinator(repeated)
        result = recombinator.recombine(crossfade_duration=0.005)

        # Save post-transformed
        post_path = output_dir / "repeated_post.wav"
        self._save_audio(post_path, sample_rate, result)

        assert pre_path.exists()
        assert post_path.exists()

    def test_generate_filtered(self, output_dir, source_audio):
        """Generate filtered audio - only slices matching criteria."""
        sample_rate, data = source_audio

        # Save pre-transformed
        pre_path = output_dir / "filtered_pre.wav"
        self._save_audio(pre_path, sample_rate, data)

        # Slice and filter
        slicer = AudioSlicer(AMEN_WAV_PATH, method="onset", sensitivity=0.5)
        slices = slicer.detect_slices()
        collection = SliceCollection(slices)
        # Keep only slices longer than 0.1 seconds
        filtered = collection.filter(lambda s: s.duration > 0.1)

        # Recombine
        recombinator = SliceRecombinator(filtered)
        result = recombinator.recombine(crossfade_duration=0.01)

        # Save post-transformed
        post_path = output_dir / "filtered_post.wav"
        self._save_audio(post_path, sample_rate, result)

        assert pre_path.exists()
        assert post_path.exists()

    def test_generate_sorted_by_duration(self, output_dir, source_audio):
        """Generate audio with slices sorted by duration."""
        sample_rate, data = source_audio

        # Save pre-transformed
        pre_path = output_dir / "sorted_duration_pre.wav"
        self._save_audio(pre_path, sample_rate, data)

        # Slice and sort by duration
        slicer = AudioSlicer(AMEN_WAV_PATH, method="zero_crossing")
        slices = slicer.detect_slices(target_slices=12)
        collection = SliceCollection(slices)

        # Custom sort by duration
        recombinator = SliceRecombinator(collection)
        result = recombinator.recombine(
            method="custom",
            order_func=lambda slices: sorted(slices, key=lambda s: s.duration),
            crossfade_duration=0.005,
        )

        # Save post-transformed
        post_path = output_dir / "sorted_duration_post.wav"
        self._save_audio(post_path, sample_rate, result)

        assert pre_path.exists()
        assert post_path.exists()

    def test_generate_normalized(self, output_dir, source_audio):
        """Generate normalized audio - peak normalized to 1.0."""
        sample_rate, data = source_audio

        # Save pre-transformed
        pre_path = output_dir / "normalized_pre.wav"
        self._save_audio(pre_path, sample_rate, data)

        # Slice and recombine with normalization
        slicer = AudioSlicer(AMEN_WAV_PATH, method="grid")
        slices = slicer.detect_slices(divisions=8)
        collection = SliceCollection(slices)

        recombinator = SliceRecombinator(collection)
        result = recombinator.recombine(normalize=True)

        # Save post-transformed
        post_path = output_dir / "normalized_post.wav"
        self._save_audio(post_path, sample_rate, result)

        assert pre_path.exists()
        assert post_path.exists()
        # Verify normalization
        assert np.max(np.abs(result)) <= 1.0
