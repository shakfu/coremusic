#!/usr/bin/env python3
"""Tests for MIDI transformation pipeline.

Tests cover all transformers and the pipeline API.
"""

import os
import tempfile
from pathlib import Path

import pytest

from coremusic.midi.utilities import MIDIEvent, MIDISequence, MIDIStatus, MIDITrack
from coremusic.midi.transform import (
    # Base classes
    MIDITransformer,
    Pipeline,
    # Pitch transformers
    Transpose,
    Invert,
    Harmonize,
    # Time transformers
    Quantize,
    TimeStretch,
    TimeShift,
    Reverse,
    # Velocity transformers
    VelocityScale,
    VelocityCurve,
    Humanize,
    # Filter transformers
    NoteFilter,
    ScaleFilter,
    EventTypeFilter,
    # Track transformers
    ChannelRemap,
    TrackMerge,
    # Arpeggio
    Arpeggiate,
    # Convenience functions
    transpose,
    quantize,
    humanize,
    reverse,
    scale_velocity,
    filter_to_scale,
)
from coremusic.music.theory import Note, Scale, ScaleType


# ============================================================================
# Test Fixtures
# ============================================================================


def make_simple_sequence(track_name: str = "Test") -> MIDISequence:
    """Create a simple test sequence with one track.

    Args:
        track_name: Name for the track (visible in DAWs like Ableton Live)
    """
    seq = MIDISequence(tempo=120.0)
    track = seq.add_track(track_name)
    # Add a C major scale
    notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
    for i, note in enumerate(notes):
        track.add_note(i * 0.5, note, 100, 0.4)
    return seq


def make_chord_sequence(track_name: str = "Chords") -> MIDISequence:
    """Create a sequence with chords.

    Args:
        track_name: Name for the track (visible in DAWs like Ableton Live)
    """
    seq = MIDISequence(tempo=120.0)
    track = seq.add_track(track_name)
    # C major chord at time 0
    for note in [60, 64, 67]:
        track.events.append(MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, note, 100))
    for note in [60, 64, 67]:
        track.events.append(MIDIEvent(1.0, MIDIStatus.NOTE_OFF, 0, note, 0))
    # G major chord at time 1
    for note in [55, 59, 62]:
        track.events.append(MIDIEvent(1.0, MIDIStatus.NOTE_ON, 0, note, 80))
    for note in [55, 59, 62]:
        track.events.append(MIDIEvent(2.0, MIDIStatus.NOTE_OFF, 0, note, 0))
    track.events.sort(key=lambda e: e.time)
    return seq


def make_multi_channel_sequence(track_name: str = "Multi") -> MIDISequence:
    """Create a sequence with events on multiple channels.

    Args:
        track_name: Name for the track (visible in DAWs like Ableton Live)
    """
    seq = MIDISequence(tempo=120.0)
    track = seq.add_track(track_name)
    # Channel 0: melody
    track.add_note(0.0, 60, 100, 0.5, channel=0)
    track.add_note(0.5, 64, 100, 0.5, channel=0)
    # Channel 9: drums
    track.add_note(0.0, 36, 120, 0.25, channel=9)
    track.add_note(0.5, 38, 110, 0.25, channel=9)
    return seq


@pytest.fixture
def simple_sequence():
    """Create a simple test sequence with one track (default name)."""
    return make_simple_sequence()


@pytest.fixture
def chord_sequence():
    """Create a sequence with chords (default name)."""
    return make_chord_sequence()


@pytest.fixture
def multi_channel_sequence():
    """Create a sequence with events on multiple channels (default name)."""
    return make_multi_channel_sequence()


@pytest.fixture
def output_dir():
    """Create temporary output directory for MIDI files."""
    path = Path("build/midi_files/transform_tests")
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================
# Pipeline Tests
# ============================================================================


class TestPipeline:
    """Tests for Pipeline class."""

    def test_empty_pipeline(self, simple_sequence):
        """Empty pipeline returns equivalent sequence."""
        pipeline = Pipeline()
        result = pipeline.apply(simple_sequence)
        assert len(result.tracks) == len(simple_sequence.tracks)
        assert len(result.tracks[0].events) == len(simple_sequence.tracks[0].events)

    def test_single_transformer(self, simple_sequence):
        """Pipeline with single transformer."""
        pipeline = Pipeline([Transpose(12)])
        result = pipeline.apply(simple_sequence)
        # All notes should be transposed up an octave
        for orig, trans in zip(simple_sequence.tracks[0].events, result.tracks[0].events):
            if orig.status == MIDIStatus.NOTE_ON:
                assert trans.data1 == orig.data1 + 12

    def test_chained_transformers(self, simple_sequence):
        """Pipeline chains transformers in order."""
        pipeline = Pipeline([
            Transpose(12),
            VelocityScale(factor=0.5),
        ])
        result = pipeline.apply(simple_sequence)
        for orig, trans in zip(simple_sequence.tracks[0].events, result.tracks[0].events):
            if orig.is_note_on:
                assert trans.data1 == orig.data1 + 12
                assert trans.data2 == orig.data2 // 2

    def test_add_method(self, simple_sequence):
        """Pipeline.add() returns self for chaining."""
        pipeline = Pipeline()
        result = pipeline.add(Transpose(5)).add(VelocityScale(factor=0.8))
        assert result is pipeline
        assert len(pipeline) == 2

    def test_callable(self, simple_sequence):
        """Pipeline can be called directly."""
        pipeline = Pipeline([Transpose(5)])
        result = pipeline(simple_sequence)
        assert result.tracks[0].events[0].data1 == simple_sequence.tracks[0].events[0].data1 + 5

    def test_repr(self):
        """Pipeline repr shows transformer names."""
        pipeline = Pipeline([Transpose(5), Quantize(0.125)])
        repr_str = repr(pipeline)
        assert "Transpose" in repr_str
        assert "Quantize" in repr_str


# ============================================================================
# Pitch Transformer Tests
# ============================================================================


class TestTranspose:
    """Tests for Transpose transformer."""

    def test_transpose_up(self, simple_sequence):
        """Transpose notes up."""
        result = Transpose(5).transform(simple_sequence)
        for orig, trans in zip(simple_sequence.tracks[0].events, result.tracks[0].events):
            if orig.status in (MIDIStatus.NOTE_ON, MIDIStatus.NOTE_OFF):
                assert trans.data1 == orig.data1 + 5

    def test_transpose_down(self, simple_sequence):
        """Transpose notes down."""
        result = Transpose(-7).transform(simple_sequence)
        for orig, trans in zip(simple_sequence.tracks[0].events, result.tracks[0].events):
            if orig.status in (MIDIStatus.NOTE_ON, MIDIStatus.NOTE_OFF):
                assert trans.data1 == orig.data1 - 7

    def test_transpose_clamps_high(self):
        """Transpose clamps notes at 127."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.add_note(0.0, 120, 100, 0.5)
        result = Transpose(20).transform(seq)
        # Note should be clamped to 127
        note_on = [e for e in result.tracks[0].events if e.is_note_on][0]
        assert note_on.data1 == 127

    def test_transpose_clamps_low(self):
        """Transpose clamps notes at 0."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.add_note(0.0, 10, 100, 0.5)
        result = Transpose(-20).transform(seq)
        note_on = [e for e in result.tracks[0].events if e.is_note_on][0]
        assert note_on.data1 == 0

    def test_transpose_preserves_non_note_events(self):
        """Transpose doesn't affect non-note events."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.add_note(0.0, 60, 100, 0.5)
        track.add_control_change(0.25, 7, 100)  # Volume
        result = Transpose(12).transform(seq)
        cc_events = [e for e in result.tracks[0].events if e.status == MIDIStatus.CONTROL_CHANGE]
        assert len(cc_events) == 1
        assert cc_events[0].data1 == 7  # Controller unchanged
        assert cc_events[0].data2 == 100  # Value unchanged


class TestInvert:
    """Tests for Invert transformer."""

    def test_invert_around_c4(self):
        """Invert melody around middle C."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.add_note(0.0, 64, 100, 0.5)  # E4 (4 above C4)
        result = Invert(60).transform(seq)
        note_on = [e for e in result.tracks[0].events if e.is_note_on][0]
        # E4 (64) inverted around C4 (60) = G#3 (56)
        assert note_on.data1 == 56

    def test_invert_symmetric(self):
        """Inverting twice returns original."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.add_note(0.0, 67, 100, 0.5)  # G4
        result = Invert(60).transform(Invert(60).transform(seq))
        note_on = [e for e in result.tracks[0].events if e.is_note_on][0]
        assert note_on.data1 == 67


class TestHarmonize:
    """Tests for Harmonize transformer."""

    def test_harmonize_major_third(self, simple_sequence):
        """Add major third harmony."""
        result = Harmonize([4]).transform(simple_sequence)
        # Should have double the note events
        orig_notes = len([e for e in simple_sequence.tracks[0].events if e.is_note_on])
        result_notes = len([e for e in result.tracks[0].events if e.is_note_on])
        assert result_notes == orig_notes * 2

    def test_harmonize_triad(self):
        """Add third and fifth for triad."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.add_note(0.0, 60, 100, 0.5)  # C4
        result = Harmonize([4, 7]).transform(seq)
        note_ons = [e for e in result.tracks[0].events if e.is_note_on]
        notes = sorted([e.data1 for e in note_ons])
        assert notes == [60, 64, 67]  # C, E, G

    def test_harmonize_velocity_scale(self):
        """Harmony notes have scaled velocity."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.add_note(0.0, 60, 100, 0.5)
        result = Harmonize([4], velocity_scale=0.5).transform(seq)
        note_ons = [e for e in result.tracks[0].events if e.is_note_on]
        velocities = [e.data2 for e in note_ons]
        assert 100 in velocities  # Original
        assert 50 in velocities   # Harmony (50% of 100)


# ============================================================================
# Time Transformer Tests
# ============================================================================


class TestQuantize:
    """Tests for Quantize transformer."""

    def test_quantize_full_strength(self):
        """Full quantization snaps to grid."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.events.append(MIDIEvent(0.13, MIDIStatus.NOTE_ON, 0, 60, 100))
        track.events.append(MIDIEvent(0.63, MIDIStatus.NOTE_OFF, 0, 60, 0))
        result = Quantize(grid=0.25, strength=1.0).transform(seq)
        # 0.13 should snap to 0.25
        assert result.tracks[0].events[0].time == pytest.approx(0.25, abs=0.01)

    def test_quantize_partial_strength(self):
        """Partial quantization blends original and grid."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.events.append(MIDIEvent(0.1, MIDIStatus.NOTE_ON, 0, 60, 100))
        result = Quantize(grid=0.25, strength=0.5).transform(seq)
        # 0.1 with 50% strength toward 0.0 grid point
        # Expected: 0.1 + (0.0 - 0.1) * 0.5 = 0.05
        assert result.tracks[0].events[0].time == pytest.approx(0.05, abs=0.01)

    def test_quantize_with_swing(self):
        """Swing shifts odd grid positions."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.events.append(MIDIEvent(0.25, MIDIStatus.NOTE_ON, 0, 60, 100))  # Odd grid
        result = Quantize(grid=0.25, swing=0.5).transform(seq)
        # With swing, odd grid positions shift forward
        assert result.tracks[0].events[0].time > 0.25

    def test_quantize_invalid_grid(self):
        """Raise error for invalid grid."""
        with pytest.raises(ValueError, match="Grid must be > 0"):
            Quantize(grid=0)

    def test_quantize_invalid_strength(self):
        """Raise error for invalid strength."""
        with pytest.raises(ValueError, match="Strength must be 0.0-1.0"):
            Quantize(grid=0.25, strength=1.5)


class TestTimeStretch:
    """Tests for TimeStretch transformer."""

    def test_stretch_double(self, simple_sequence):
        """Double the duration."""
        result = TimeStretch(2.0).transform(simple_sequence)
        for orig, trans in zip(simple_sequence.tracks[0].events, result.tracks[0].events):
            assert trans.time == pytest.approx(orig.time * 2.0)

    def test_stretch_half(self, simple_sequence):
        """Halve the duration."""
        result = TimeStretch(0.5).transform(simple_sequence)
        for orig, trans in zip(simple_sequence.tracks[0].events, result.tracks[0].events):
            assert trans.time == pytest.approx(orig.time * 0.5)

    def test_stretch_invalid_factor(self):
        """Raise error for invalid factor."""
        with pytest.raises(ValueError, match="Factor must be > 0"):
            TimeStretch(0)


class TestTimeShift:
    """Tests for TimeShift transformer."""

    def test_shift_forward(self, simple_sequence):
        """Shift events forward in time."""
        result = TimeShift(1.0).transform(simple_sequence)
        for orig, trans in zip(simple_sequence.tracks[0].events, result.tracks[0].events):
            assert trans.time == pytest.approx(orig.time + 1.0)

    def test_shift_backward_clamps(self):
        """Shift backward clamps at 0."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.events.append(MIDIEvent(0.5, MIDIStatus.NOTE_ON, 0, 60, 100))
        result = TimeShift(-1.0).transform(seq)
        assert result.tracks[0].events[0].time == 0.0


class TestReverse:
    """Tests for Reverse transformer."""

    def test_reverse_note_order(self):
        """Reverse changes note order."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.add_note(0.0, 60, 100, 0.5)  # C4 first
        track.add_note(1.0, 64, 100, 0.5)  # E4 second
        result = Reverse().transform(seq)
        note_ons = sorted([e for e in result.tracks[0].events if e.is_note_on], key=lambda e: e.time)
        # E4 should now come first (it was at the end)
        assert note_ons[0].data1 == 64
        assert note_ons[1].data1 == 60

    def test_reverse_preserves_duration(self):
        """Reverse preserves note durations."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.add_note(0.0, 60, 100, 0.3)
        result = Reverse().transform(seq)
        note_on = [e for e in result.tracks[0].events if e.is_note_on][0]
        note_off = [e for e in result.tracks[0].events if e.is_note_off][0]
        duration = note_off.time - note_on.time
        assert duration == pytest.approx(0.3, abs=0.01)


# ============================================================================
# Velocity Transformer Tests
# ============================================================================


class TestVelocityScale:
    """Tests for VelocityScale transformer."""

    def test_scale_by_factor(self, simple_sequence):
        """Scale velocity by factor."""
        result = VelocityScale(factor=0.5).transform(simple_sequence)
        for orig, trans in zip(simple_sequence.tracks[0].events, result.tracks[0].events):
            if orig.is_note_on:
                assert trans.data2 == orig.data2 // 2

    def test_scale_to_range(self):
        """Scale velocity to min/max range."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.add_note(0.0, 60, 50, 0.5)   # Low velocity
        track.add_note(0.5, 62, 100, 0.5)  # High velocity
        result = VelocityScale(min_vel=60, max_vel=80).transform(seq)
        note_ons = [e for e in result.tracks[0].events if e.is_note_on]
        velocities = [e.data2 for e in note_ons]
        assert min(velocities) >= 60
        assert max(velocities) <= 80

    def test_velocity_clamps(self):
        """Velocity clamps to 1-127."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.add_note(0.0, 60, 10, 0.5)
        result = VelocityScale(factor=20).transform(seq)  # Would be 200
        note_on = [e for e in result.tracks[0].events if e.is_note_on][0]
        assert note_on.data2 == 127


class TestVelocityCurve:
    """Tests for VelocityCurve transformer."""

    def test_log_curve(self):
        """Logarithmic curve softens dynamics."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.add_note(0.0, 60, 64, 0.5)  # Half velocity
        result = VelocityCurve(curve='log').transform(seq)
        note_on = [e for e in result.tracks[0].events if e.is_note_on][0]
        # Log curve: sqrt(0.5) * 127 = ~90
        assert note_on.data2 > 64  # Should be higher

    def test_exp_curve(self):
        """Exponential curve increases dynamics."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.add_note(0.0, 60, 64, 0.5)
        result = VelocityCurve(curve='exp').transform(seq)
        note_on = [e for e in result.tracks[0].events if e.is_note_on][0]
        # Exp curve: (0.5)^2 * 127 = ~32
        assert note_on.data2 < 64  # Should be lower

    def test_custom_curve(self):
        """Custom curve function."""
        result = VelocityCurve(curve=lambda x: 1.0 - x).transform(
            MIDISequence().add_track("Test") or MIDISequence()
        )
        # Just test it doesn't error

    def test_invalid_curve_name(self):
        """Raise error for unknown curve name."""
        with pytest.raises(ValueError, match="Unknown curve"):
            VelocityCurve(curve='invalid')


class TestHumanize:
    """Tests for Humanize transformer."""

    def test_humanize_adds_variation(self, simple_sequence):
        """Humanize adds timing and velocity variation."""
        result = Humanize(timing=0.02, velocity=10, seed=42).transform(simple_sequence)
        # Check that at least some values differ
        timing_differs = False
        velocity_differs = False
        for orig, trans in zip(simple_sequence.tracks[0].events, result.tracks[0].events):
            if abs(orig.time - trans.time) > 0.001:
                timing_differs = True
            if orig.is_note_on and orig.data2 != trans.data2:
                velocity_differs = True
        assert timing_differs
        assert velocity_differs

    def test_humanize_reproducible(self, simple_sequence):
        """Same seed produces same results."""
        result1 = Humanize(timing=0.02, velocity=10, seed=42).transform(simple_sequence)
        result2 = Humanize(timing=0.02, velocity=10, seed=42).transform(simple_sequence)
        for e1, e2 in zip(result1.tracks[0].events, result2.tracks[0].events):
            assert e1.time == e2.time
            assert e1.data2 == e2.data2

    def test_humanize_invalid_timing(self):
        """Raise error for negative timing."""
        with pytest.raises(ValueError, match="Timing must be >= 0"):
            Humanize(timing=-0.01)


# ============================================================================
# Filter Transformer Tests
# ============================================================================


class TestNoteFilter:
    """Tests for NoteFilter transformer."""

    def test_filter_by_pitch_range(self, simple_sequence):
        """Filter notes by pitch range."""
        result = NoteFilter(min_note=64, max_note=68).transform(simple_sequence)
        note_ons = [e for e in result.tracks[0].events if e.is_note_on]
        for note in note_ons:
            assert 64 <= note.data1 <= 68

    def test_filter_by_velocity(self):
        """Filter notes by velocity."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.add_note(0.0, 60, 50, 0.5)   # Soft
        track.add_note(0.5, 62, 100, 0.5)  # Loud
        result = NoteFilter(min_velocity=80).transform(seq)
        note_ons = [e for e in result.tracks[0].events if e.is_note_on]
        assert len(note_ons) == 1
        assert note_ons[0].data1 == 62

    def test_filter_by_channel(self, multi_channel_sequence):
        """Filter notes by channel."""
        result = NoteFilter(channels={0}).transform(multi_channel_sequence)
        for event in result.tracks[0].events:
            if event.status in (MIDIStatus.NOTE_ON, MIDIStatus.NOTE_OFF):
                assert event.channel == 0

    def test_filter_invert(self):
        """Invert filter removes matching notes."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.add_note(0.0, 60, 100, 0.5)  # C4
        track.add_note(0.5, 72, 100, 0.5)  # C5
        result = NoteFilter(min_note=70, invert=True).transform(seq)
        note_ons = [e for e in result.tracks[0].events if e.is_note_on]
        # Should only have C4 (C5 matched and was removed)
        assert len(note_ons) == 1
        assert note_ons[0].data1 == 60


class TestScaleFilter:
    """Tests for ScaleFilter transformer."""

    def test_filter_to_c_major(self):
        """Filter notes to C major scale."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        # C major scale notes: C, D, E, F, G, A, B (0, 2, 4, 5, 7, 9, 11)
        # Non-scale notes: C#, D#, F#, G#, A# (1, 3, 6, 8, 10)
        track.add_note(0.0, 60, 100, 0.5)  # C4 - in scale
        track.add_note(0.5, 61, 100, 0.5)  # C#4 - NOT in scale
        track.add_note(1.0, 62, 100, 0.5)  # D4 - in scale
        track.add_note(1.5, 63, 100, 0.5)  # D#4 - NOT in scale
        track.add_note(2.0, 64, 100, 0.5)  # E4 - in scale

        c_major = Scale(Note('C', 4), ScaleType.MAJOR)
        result = ScaleFilter(c_major).transform(seq)

        note_ons = [e for e in result.tracks[0].events if e.is_note_on]
        assert len(note_ons) == 3
        assert [n.data1 for n in note_ons] == [60, 62, 64]

    def test_filter_to_a_minor_pentatonic(self):
        """Filter notes to A minor pentatonic scale."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        # A minor pentatonic: A, C, D, E, G (9, 0, 2, 4, 7)
        track.add_note(0.0, 57, 100, 0.5)  # A3 - in scale
        track.add_note(0.5, 59, 100, 0.5)  # B3 - NOT in scale
        track.add_note(1.0, 60, 100, 0.5)  # C4 - in scale
        track.add_note(1.5, 61, 100, 0.5)  # C#4 - NOT in scale
        track.add_note(2.0, 62, 100, 0.5)  # D4 - in scale

        a_pent = Scale(Note('A', 3), ScaleType.MINOR_PENTATONIC)
        result = ScaleFilter(a_pent).transform(seq)

        note_ons = [e for e in result.tracks[0].events if e.is_note_on]
        assert len(note_ons) == 3
        assert [n.data1 for n in note_ons] == [57, 60, 62]

    def test_filter_preserves_note_offs(self):
        """Ensure note-offs are kept for filtered notes."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.add_note(0.0, 60, 100, 0.5)  # C4 - in scale
        track.add_note(0.5, 61, 100, 0.5)  # C#4 - NOT in scale

        c_major = Scale(Note('C', 4), ScaleType.MAJOR)
        result = ScaleFilter(c_major).transform(seq)

        # Should have note on and note off for C4 only
        events = result.tracks[0].events
        note_ons = [e for e in events if e.is_note_on]
        note_offs = [e for e in events if e.is_note_off]

        assert len(note_ons) == 1
        assert len(note_offs) == 1
        assert note_ons[0].data1 == 60
        assert note_offs[0].data1 == 60

    def test_filter_preserves_non_note_events(self):
        """Control changes and other events are preserved."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.add_note(0.0, 60, 100, 0.5)  # C4 - in scale
        track.add_note(0.5, 61, 100, 0.5)  # C#4 - NOT in scale
        track.add_control_change(0.25, 7, 100)  # Volume CC

        c_major = Scale(Note('C', 4), ScaleType.MAJOR)
        result = ScaleFilter(c_major).transform(seq)

        cc_events = [e for e in result.tracks[0].events
                     if e.status == MIDIStatus.CONTROL_CHANGE]
        assert len(cc_events) == 1

    def test_filter_across_octaves(self):
        """Scale filtering works across multiple octaves."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        # C notes in different octaves
        track.add_note(0.0, 36, 100, 0.5)  # C2 - in scale
        track.add_note(0.5, 48, 100, 0.5)  # C3 - in scale
        track.add_note(1.0, 60, 100, 0.5)  # C4 - in scale
        track.add_note(1.5, 61, 100, 0.5)  # C#4 - NOT in scale
        track.add_note(2.0, 72, 100, 0.5)  # C5 - in scale

        c_major = Scale(Note('C', 4), ScaleType.MAJOR)
        result = ScaleFilter(c_major).transform(seq)

        note_ons = [e for e in result.tracks[0].events if e.is_note_on]
        assert len(note_ons) == 4
        assert [n.data1 for n in note_ons] == [36, 48, 60, 72]

    def test_filter_chromatic_scale_keeps_all(self):
        """Chromatic scale filter keeps all notes."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        for note in range(60, 72):  # All 12 notes
            track.add_note(note - 60, note, 100, 0.5)

        chromatic = Scale(Note('C', 4), ScaleType.CHROMATIC)
        result = ScaleFilter(chromatic).transform(seq)

        note_ons = [e for e in result.tracks[0].events if e.is_note_on]
        assert len(note_ons) == 12

    def test_filter_blues_scale(self):
        """Filter to blues scale."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        # Blues scale: C, Eb, F, F#, G, Bb (0, 3, 5, 6, 7, 10)
        track.add_note(0.0, 60, 100, 0.5)  # C - in scale
        track.add_note(0.5, 62, 100, 0.5)  # D - NOT in scale
        track.add_note(1.0, 63, 100, 0.5)  # Eb - in scale
        track.add_note(1.5, 64, 100, 0.5)  # E - NOT in scale
        track.add_note(2.0, 65, 100, 0.5)  # F - in scale

        blues = Scale(Note('C', 4), ScaleType.BLUES)
        result = ScaleFilter(blues).transform(seq)

        note_ons = [e for e in result.tracks[0].events if e.is_note_on]
        assert len(note_ons) == 3
        assert [n.data1 for n in note_ons] == [60, 63, 65]

    def test_filter_invalid_scale_type(self):
        """Raise error for non-Scale input."""
        with pytest.raises(TypeError):
            ScaleFilter("not a scale")

    def test_filter_convenience_function(self):
        """Test filter_to_scale convenience function."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.add_note(0.0, 60, 100, 0.5)  # C4 - in scale
        track.add_note(0.5, 61, 100, 0.5)  # C#4 - NOT in scale

        c_major = Scale(Note('C', 4), ScaleType.MAJOR)
        result = filter_to_scale(seq, c_major)

        note_ons = [e for e in result.tracks[0].events if e.is_note_on]
        assert len(note_ons) == 1
        assert note_ons[0].data1 == 60

    def test_filter_repr(self):
        """Test string representation."""
        c_major = Scale(Note('C', 4), ScaleType.MAJOR)
        filt = ScaleFilter(c_major)
        assert "ScaleFilter" in repr(filt)

    def test_filter_multi_track(self):
        """Filter works across multiple tracks."""
        seq = MIDISequence()
        track1 = seq.add_track("Track1")
        track2 = seq.add_track("Track2")

        track1.add_note(0.0, 60, 100, 0.5)  # C4 - in scale
        track1.add_note(0.5, 61, 100, 0.5)  # C#4 - NOT in scale

        track2.add_note(0.0, 62, 100, 0.5)  # D4 - in scale
        track2.add_note(0.5, 63, 100, 0.5)  # D#4 - NOT in scale

        c_major = Scale(Note('C', 4), ScaleType.MAJOR)
        result = ScaleFilter(c_major).transform(seq)

        track1_notes = [e for e in result.tracks[0].events if e.is_note_on]
        track2_notes = [e for e in result.tracks[1].events if e.is_note_on]

        assert len(track1_notes) == 1
        assert track1_notes[0].data1 == 60
        assert len(track2_notes) == 1
        assert track2_notes[0].data1 == 62


class TestEventTypeFilter:
    """Tests for EventTypeFilter transformer."""

    def test_keep_notes_only(self):
        """Keep only note events."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.add_note(0.0, 60, 100, 0.5)
        track.add_control_change(0.25, 7, 100)
        result = EventTypeFilter(keep=[MIDIStatus.NOTE_ON, MIDIStatus.NOTE_OFF]).transform(seq)
        for event in result.tracks[0].events:
            assert event.status in (MIDIStatus.NOTE_ON, MIDIStatus.NOTE_OFF)

    def test_remove_control_changes(self):
        """Remove control change events."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        track.add_note(0.0, 60, 100, 0.5)
        track.add_control_change(0.25, 7, 100)
        result = EventTypeFilter(remove=[MIDIStatus.CONTROL_CHANGE]).transform(seq)
        cc_events = [e for e in result.tracks[0].events if e.status == MIDIStatus.CONTROL_CHANGE]
        assert len(cc_events) == 0


# ============================================================================
# Track Transformer Tests
# ============================================================================


class TestChannelRemap:
    """Tests for ChannelRemap transformer."""

    def test_remap_channel(self, multi_channel_sequence):
        """Remap MIDI channel."""
        result = ChannelRemap({0: 1}).transform(multi_channel_sequence)
        for event in result.tracks[0].events:
            if event.status in (MIDIStatus.NOTE_ON, MIDIStatus.NOTE_OFF):
                assert event.channel in (1, 9)  # 0->1, 9 unchanged

    def test_remap_invalid_channel(self):
        """Raise error for invalid channel."""
        with pytest.raises(ValueError, match="channel must be 0-15"):
            ChannelRemap({16: 0})


class TestTrackMerge:
    """Tests for TrackMerge transformer."""

    def test_merge_tracks(self):
        """Merge multiple tracks into one."""
        seq = MIDISequence()
        track1 = seq.add_track("Track 1")
        track1.add_note(0.0, 60, 100, 0.5)
        track2 = seq.add_track("Track 2")
        track2.add_note(0.0, 64, 100, 0.5)
        result = TrackMerge(name="Merged").transform(seq)
        assert len(result.tracks) == 1
        assert result.tracks[0].name == "Merged"
        note_ons = [e for e in result.tracks[0].events if e.is_note_on]
        assert len(note_ons) == 2


# ============================================================================
# Arpeggiate Transformer Tests
# ============================================================================


class TestArpeggiate:
    """Tests for Arpeggiate transformer."""

    def test_arpeggiate_chord(self, chord_sequence):
        """Arpeggiate chords."""
        result = Arpeggiate(pattern='up', note_duration=0.1).transform(chord_sequence)
        # Check that notes are now sequential, not simultaneous
        note_ons = sorted([e for e in result.tracks[0].events if e.is_note_on], key=lambda e: e.time)
        times = [e.time for e in note_ons]
        # First chord should be arpeggiated
        assert times[1] > times[0]
        assert times[2] > times[1]

    def test_arpeggiate_up_pattern(self):
        """Up pattern plays low to high."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        for note in [60, 64, 67]:
            track.events.append(MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, note, 100))
        for note in [60, 64, 67]:
            track.events.append(MIDIEvent(1.0, MIDIStatus.NOTE_OFF, 0, note, 0))
        result = Arpeggiate(pattern='up', note_duration=0.1).transform(seq)
        note_ons = sorted([e for e in result.tracks[0].events if e.is_note_on], key=lambda e: e.time)
        notes = [e.data1 for e in note_ons]
        assert notes == [60, 64, 67]  # Low to high

    def test_arpeggiate_down_pattern(self):
        """Down pattern plays high to low."""
        seq = MIDISequence()
        track = seq.add_track("Test")
        for note in [60, 64, 67]:
            track.events.append(MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, note, 100))
        for note in [60, 64, 67]:
            track.events.append(MIDIEvent(1.0, MIDIStatus.NOTE_OFF, 0, note, 0))
        result = Arpeggiate(pattern='down', note_duration=0.1).transform(seq)
        note_ons = sorted([e for e in result.tracks[0].events if e.is_note_on], key=lambda e: e.time)
        notes = [e.data1 for e in note_ons]
        assert notes == [67, 64, 60]  # High to low

    def test_arpeggiate_invalid_pattern(self):
        """Raise error for invalid pattern."""
        with pytest.raises(ValueError, match="Unknown pattern"):
            Arpeggiate(pattern='invalid')


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_transpose_function(self, simple_sequence):
        """transpose() convenience function."""
        result = transpose(simple_sequence, 5)
        assert result.tracks[0].events[0].data1 == simple_sequence.tracks[0].events[0].data1 + 5

    def test_quantize_function(self, simple_sequence):
        """quantize() convenience function."""
        result = quantize(simple_sequence, 0.25)
        # All times should be on grid
        for event in result.tracks[0].events:
            assert event.time % 0.25 == pytest.approx(0.0, abs=0.01) or \
                   event.time % 0.25 == pytest.approx(0.25, abs=0.01)

    def test_humanize_function(self, simple_sequence):
        """humanize() convenience function."""
        result = humanize(simple_sequence, timing=0.01, velocity=5)
        # Should produce some variation
        assert result is not simple_sequence

    def test_reverse_function(self, simple_sequence):
        """reverse() convenience function."""
        result = reverse(simple_sequence)
        assert result is not simple_sequence

    def test_scale_velocity_function(self, simple_sequence):
        """scale_velocity() convenience function."""
        result = scale_velocity(simple_sequence, factor=0.5)
        for orig, trans in zip(simple_sequence.tracks[0].events, result.tracks[0].events):
            if orig.is_note_on:
                assert trans.data2 == orig.data2 // 2


# ============================================================================
# Integration Tests - MIDI File Generation
# ============================================================================


class TestMIDIFileGeneration:
    """Integration tests that generate actual MIDI files.

    Each test saves both pre-transformed (pre_<name>.mid) and
    post-transformed (post_<name>.mid) files for comparison.
    Track names match the filename (minus .mid) for easy identification in DAWs.
    """

    def _save_with_track_name(self, seq: MIDISequence, path: Path) -> None:
        """Save sequence with track name matching the filename."""
        track_name = path.stem  # filename without .mid
        # Find the first track with events (skip tempo track if present)
        for track in seq.tracks:
            if track.events:
                track.name = track_name
                break
        seq.save(str(path))

    def test_generate_transposed(self, output_dir):
        """Generate transposed MIDI file."""
        seq = make_simple_sequence()

        # Save pre-transformed
        pre_path = output_dir / "transposed_fifth_pre.mid"
        self._save_with_track_name(seq, pre_path)

        # Transform and save post-transformed
        result = Transpose(7).transform(seq)  # Perfect fifth
        post_path = output_dir / "transposed_fifth_post.mid"
        self._save_with_track_name(result, post_path)

        assert pre_path.exists()
        assert post_path.exists()

    def test_generate_quantized(self, output_dir):
        """Generate quantized MIDI file."""
        seq = MIDISequence(tempo=120.0)
        track = seq.add_track("temp")
        # Add notes with timing variations
        for i in range(8):
            track.add_note(i * 0.5 + (i % 2) * 0.08, 60 + i, 100, 0.4)

        # Save pre-transformed
        pre_path = output_dir / "quantized_pre.mid"
        self._save_with_track_name(seq, pre_path)

        # Transform and save post-transformed
        result = Quantize(grid=0.5, strength=1.0).transform(seq)
        post_path = output_dir / "quantized_post.mid"
        self._save_with_track_name(result, post_path)

        assert pre_path.exists()
        assert post_path.exists()

    def test_generate_humanized(self, output_dir):
        """Generate humanized MIDI file."""
        seq = make_simple_sequence()

        # Save pre-transformed
        pre_path = output_dir / "humanized_pre.mid"
        self._save_with_track_name(seq, pre_path)

        # Transform and save post-transformed
        result = Humanize(timing=0.02, velocity=15, seed=42).transform(seq)
        post_path = output_dir / "humanized_post.mid"
        self._save_with_track_name(result, post_path)

        assert pre_path.exists()
        assert post_path.exists()

    def test_generate_harmonized(self, output_dir):
        """Generate harmonized MIDI file."""
        seq = make_simple_sequence()

        # Save pre-transformed
        pre_path = output_dir / "harmonized_triads_pre.mid"
        self._save_with_track_name(seq, pre_path)

        # Transform and save post-transformed
        result = Harmonize([4, 7], velocity_scale=0.7).transform(seq)
        post_path = output_dir / "harmonized_triads_post.mid"
        self._save_with_track_name(result, post_path)

        assert pre_path.exists()
        assert post_path.exists()

    def test_generate_arpeggiated(self, output_dir):
        """Generate arpeggiated MIDI file."""
        seq = make_chord_sequence()

        # Save pre-transformed
        pre_path = output_dir / "arpeggiated_pre.mid"
        self._save_with_track_name(seq, pre_path)

        # Transform and save post-transformed
        result = Arpeggiate(pattern='up_down', note_duration=0.1).transform(seq)
        post_path = output_dir / "arpeggiated_post.mid"
        self._save_with_track_name(result, post_path)

        assert pre_path.exists()
        assert post_path.exists()

    def test_generate_pipeline(self, output_dir):
        """Generate MIDI file with full pipeline."""
        seq = make_simple_sequence()

        # Save pre-transformed
        pre_path = output_dir / "pipeline_full_pre.mid"
        self._save_with_track_name(seq, pre_path)

        # Transform and save post-transformed
        pipeline = Pipeline([
            Transpose(5),
            Quantize(grid=0.25, strength=0.8),
            VelocityScale(min_vel=50, max_vel=110),
            Humanize(timing=0.01, velocity=5, seed=42),
        ])
        result = pipeline.apply(seq)
        post_path = output_dir / "pipeline_full_post.mid"
        self._save_with_track_name(result, post_path)

        assert pre_path.exists()
        assert post_path.exists()

    def test_generate_reversed(self, output_dir):
        """Generate reversed MIDI file."""
        seq = make_simple_sequence()

        # Save pre-transformed
        pre_path = output_dir / "reversed_pre.mid"
        self._save_with_track_name(seq, pre_path)

        # Transform and save post-transformed
        result = Reverse().transform(seq)
        post_path = output_dir / "reversed_post.mid"
        self._save_with_track_name(result, post_path)

        assert pre_path.exists()
        assert post_path.exists()

    def test_generate_inverted(self, output_dir):
        """Generate inverted MIDI file."""
        seq = make_simple_sequence()

        # Save pre-transformed
        pre_path = output_dir / "inverted_pre.mid"
        self._save_with_track_name(seq, pre_path)

        # Transform and save post-transformed
        result = Invert(60).transform(seq)
        post_path = output_dir / "inverted_post.mid"
        self._save_with_track_name(result, post_path)

        assert pre_path.exists()
        assert post_path.exists()

    def test_generate_velocity_curved(self, output_dir):
        """Generate velocity curved MIDI file."""
        seq = make_simple_sequence()

        # Save pre-transformed
        pre_path = output_dir / "velocity_soft_pre.mid"
        self._save_with_track_name(seq, pre_path)

        # Transform and save post-transformed
        result = VelocityCurve(curve='soft').transform(seq)
        post_path = output_dir / "velocity_soft_post.mid"
        self._save_with_track_name(result, post_path)

        assert pre_path.exists()
        assert post_path.exists()

    def test_generate_time_stretched(self, output_dir):
        """Generate time stretched MIDI file."""
        seq = make_simple_sequence()

        # Save pre-transformed
        pre_path = output_dir / "double_tempo_pre.mid"
        self._save_with_track_name(seq, pre_path)

        # Transform and save post-transformed
        result = TimeStretch(0.5).transform(seq)  # Double tempo
        post_path = output_dir / "double_tempo_post.mid"
        self._save_with_track_name(result, post_path)

        assert pre_path.exists()
        assert post_path.exists()


# ============================================================================
# Load and Transform Tests
# ============================================================================


class TestLoadAndTransform:
    """Test loading MIDI files and transforming them."""

    def _save_with_track_name(self, seq: MIDISequence, path: Path) -> None:
        """Save sequence with track name matching the filename."""
        track_name = path.stem  # filename without .mid
        # Find the first track with events (skip tempo track if present)
        for track in seq.tracks:
            if track.events:
                track.name = track_name
                break
        seq.save(str(path))

    def test_load_transform_save(self, output_dir):
        """Load, transform, and save MIDI file."""
        seq = make_simple_sequence()

        # Save pre-transformed (original)
        pre_path = output_dir / "loaded_transformed_pre.mid"
        self._save_with_track_name(seq, pre_path)

        # Load, transform, save post-transformed
        loaded = MIDISequence.load(str(pre_path))
        transformed = Pipeline([
            Transpose(12),
            VelocityScale(factor=0.8),
        ]).apply(loaded)
        post_path = output_dir / "loaded_transformed_post.mid"
        self._save_with_track_name(transformed, post_path)

        assert pre_path.exists()
        assert post_path.exists()

    def test_roundtrip_preserves_data(self, output_dir):
        """Roundtrip through file preserves essential data.

        Note: This test verifies file I/O works without errors.
        Detailed MIDI file parsing is tested in test_midi_utilities.py.
        """
        seq = make_simple_sequence()
        path = output_dir / "roundtrip.mid"
        self._save_with_track_name(seq, path)
        loaded = MIDISequence.load(str(path))

        # Check track count
        assert len(loaded.tracks) >= 1

        # Verify file was created and can be loaded
        assert loaded.tempo > 0
        assert loaded.ppq > 0
