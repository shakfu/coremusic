#!/usr/bin/env python3
"""Tests for MIDI utilities module."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import tempfile
import os

from coremusic.midi import (
    MIDIEvent,
    MIDITrack,
    MIDISequence,
    MIDIRouter,
    MIDIFileFormat,
    MIDIStatus,
    transpose_transform,
    velocity_scale_transform,
    channel_remap_transform,
    quantize_transform,
)


class TestMIDIEvent:
    """Tests for MIDIEvent class."""

    def test_create_event(self):
        """Test creating MIDI event."""
        event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100)

        assert event.time == 0.0
        assert event.status == MIDIStatus.NOTE_ON
        assert event.channel == 0
        assert event.data1 == 60
        assert event.data2 == 100

    def test_is_note_on(self):
        """Test note on detection."""
        event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100)
        assert event.is_note_on is True

        # Zero velocity is note off
        event2 = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 0)
        assert event2.is_note_on is False

    def test_is_note_off(self):
        """Test note off detection."""
        event = MIDIEvent(0.0, MIDIStatus.NOTE_OFF, 0, 60, 0)
        assert event.is_note_off is True

        # Note on with zero velocity is also note off
        event2 = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 0)
        assert event2.is_note_off is True

    def test_is_control_change(self):
        """Test control change detection."""
        event = MIDIEvent(0.0, MIDIStatus.CONTROL_CHANGE, 0, 7, 100)
        assert event.is_control_change is True

    def test_is_program_change(self):
        """Test program change detection."""
        event = MIDIEvent(0.0, MIDIStatus.PROGRAM_CHANGE, 0, 1, 0)
        assert event.is_program_change is True

    def test_to_bytes(self):
        """Test converting event to bytes."""
        event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100)
        data = event.to_bytes()

        assert len(data) == 3
        assert data[0] == 0x90  # Note on, channel 0
        assert data[1] == 60
        assert data[2] == 100

    def test_to_bytes_program_change(self):
        """Test converting program change to bytes."""
        event = MIDIEvent(0.0, MIDIStatus.PROGRAM_CHANGE, 0, 5, 0)
        data = event.to_bytes()

        assert len(data) == 2
        assert data[0] == 0xC0  # Program change, channel 0
        assert data[1] == 5

    def test_from_bytes(self):
        """Test creating event from bytes."""
        data = bytes([0x90, 60, 100])
        event = MIDIEvent.from_bytes(data, 1.0)

        assert event.time == 1.0
        assert event.status == MIDIStatus.NOTE_ON
        assert event.channel == 0
        assert event.data1 == 60
        assert event.data2 == 100

    def test_from_bytes_with_channel(self):
        """Test creating event from bytes with channel."""
        data = bytes([0x93, 60, 100])  # Channel 3
        event = MIDIEvent.from_bytes(data)

        assert event.channel == 3


class TestMIDITrack:
    """Tests for MIDITrack class."""

    def test_create_track(self):
        """Test creating MIDI track."""
        track = MIDITrack("Test Track")

        assert track.name == "Test Track"
        assert len(track.events) == 0
        assert track.program == 0
        assert track.channel == 0

    def test_add_note(self):
        """Test adding note to track."""
        track = MIDITrack()
        track.add_note(0.0, 60, 100, 0.5)

        assert len(track.events) == 2  # Note on + note off
        assert track.events[0].is_note_on
        assert track.events[1].is_note_off
        assert track.events[0].time == 0.0
        assert track.events[1].time == 0.5

    def test_add_note_with_channel(self):
        """Test adding note with specific channel."""
        track = MIDITrack()
        track.add_note(0.0, 60, 100, 0.5, channel=5)

        assert track.events[0].channel == 5
        assert track.events[1].channel == 5

    def test_add_note_validation(self):
        """Test note parameter validation."""
        track = MIDITrack()

        # Invalid note
        with pytest.raises(ValueError, match="Note must be 0-127"):
            track.add_note(0.0, 128, 100, 0.5)

        # Invalid velocity
        with pytest.raises(ValueError, match="Velocity must be 0-127"):
            track.add_note(0.0, 60, 128, 0.5)

        # Invalid channel
        with pytest.raises(ValueError, match="Channel must be 0-15"):
            track.add_note(0.0, 60, 100, 0.5, channel=16)

        # Invalid duration
        with pytest.raises(ValueError, match="Duration must be >= 0"):
            track.add_note(0.0, 60, 100, -1.0)

    def test_add_control_change(self):
        """Test adding control change."""
        track = MIDITrack()
        track.add_control_change(0.0, 7, 100)  # Volume

        assert len(track.events) == 1
        assert track.events[0].is_control_change
        assert track.events[0].data1 == 7
        assert track.events[0].data2 == 100

    def test_add_program_change(self):
        """Test adding program change."""
        track = MIDITrack()
        track.add_program_change(0.0, 42)

        assert len(track.events) == 1
        assert track.events[0].is_program_change
        assert track.events[0].data1 == 42
        assert track.program == 42

    def test_add_pitch_bend(self):
        """Test adding pitch bend."""
        track = MIDITrack()
        track.add_pitch_bend(0.0, 8192)  # Center

        assert len(track.events) == 1
        assert track.events[0].status == MIDIStatus.PITCH_BEND
        # 8192 = 0x2000 -> LSB=0, MSB=64
        assert track.events[0].data1 == 0
        assert track.events[0].data2 == 64

    def test_events_sorted_by_time(self):
        """Test that events are kept sorted by time."""
        track = MIDITrack()
        track.add_note(1.0, 60, 100, 0.5)
        track.add_note(0.0, 64, 100, 0.5)
        track.add_note(0.5, 67, 100, 0.5)

        # Events should be sorted by time
        times = [e.time for e in track.events]
        assert times == sorted(times)

    def test_track_duration(self):
        """Test track duration calculation."""
        track = MIDITrack()
        track.add_note(0.0, 60, 100, 0.5)
        track.add_note(1.0, 64, 100, 0.5)

        assert track.duration == 1.5

    def test_track_duration_empty(self):
        """Test empty track duration."""
        track = MIDITrack()
        assert track.duration == 0.0

    def test_clear_track(self):
        """Test clearing track events."""
        track = MIDITrack()
        track.add_note(0.0, 60, 100, 0.5)
        track.clear()

        assert len(track.events) == 0

    def test_track_len(self):
        """Test track length."""
        track = MIDITrack()
        track.add_note(0.0, 60, 100, 0.5)

        assert len(track) == 2  # Note on + note off

    def test_track_repr(self):
        """Test track string representation."""
        track = MIDITrack("Piano")
        track.add_note(0.0, 60, 100, 0.5)

        repr_str = repr(track)
        assert "Piano" in repr_str
        assert "events=2" in repr_str


class TestMIDISequence:
    """Tests for MIDISequence class."""

    def test_create_sequence(self):
        """Test creating MIDI sequence."""
        seq = MIDISequence(tempo=120.0)

        assert seq.tempo == 120.0
        assert seq.time_signature == (4, 4)
        assert len(seq.tracks) == 0
        assert seq.ppq == 480

    def test_create_sequence_with_time_signature(self):
        """Test creating sequence with custom time signature."""
        seq = MIDISequence(tempo=90.0, time_signature=(3, 4))

        assert seq.tempo == 90.0
        assert seq.time_signature == (3, 4)

    def test_sequence_validation(self):
        """Test sequence parameter validation."""
        # Invalid tempo
        with pytest.raises(ValueError, match="Tempo must be > 0"):
            MIDISequence(tempo=0.0)

        # Invalid time signature
        with pytest.raises(ValueError, match="Invalid time signature"):
            MIDISequence(time_signature=(0, 4))

    def test_add_track(self):
        """Test adding track to sequence."""
        seq = MIDISequence()
        track = seq.add_track("Test")

        assert len(seq.tracks) == 1
        assert track.name == "Test"
        assert track in seq.tracks

    def test_sequence_duration(self):
        """Test sequence duration calculation."""
        seq = MIDISequence()

        track1 = seq.add_track()
        track1.add_note(0.0, 60, 100, 1.0)

        track2 = seq.add_track()
        track2.add_note(0.0, 64, 100, 2.0)

        assert seq.duration == 2.0

    def test_sequence_duration_empty(self):
        """Test empty sequence duration."""
        seq = MIDISequence()
        assert seq.duration == 0.0

    def test_save_and_load_midi_file(self, tmp_path):
        """Test saving and loading MIDI file."""
        # Create sequence
        seq = MIDISequence(tempo=120.0, time_signature=(4, 4))

        track = seq.add_track("Melody")
        track.add_note(0.0, 60, 100, 0.5)
        track.add_note(0.5, 64, 100, 0.5)
        track.add_note(1.0, 67, 100, 0.5)

        # Save to file
        output_path = tmp_path / "test.mid"
        seq.save(str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Load from file
        loaded_seq = MIDISequence.load(str(output_path))

        assert loaded_seq.tempo == pytest.approx(120.0, rel=0.1)
        assert len(loaded_seq.tracks) > 0

    def test_save_single_track_format(self, tmp_path):
        """Test saving with single track format."""
        seq = MIDISequence()
        track = seq.add_track()
        track.add_note(0.0, 60, 100, 0.5)

        output_path = tmp_path / "single_track.mid"
        seq.save(str(output_path), format=MIDIFileFormat.SINGLE_TRACK)

        assert output_path.exists()

    def test_save_multi_track_format(self, tmp_path):
        """Test saving with multi track format."""
        seq = MIDISequence()
        track1 = seq.add_track("Track 1")
        track1.add_note(0.0, 60, 100, 0.5)
        track2 = seq.add_track("Track 2")
        track2.add_note(0.0, 64, 100, 0.5)

        output_path = tmp_path / "multi_track.mid"
        seq.save(str(output_path), format=MIDIFileFormat.MULTI_TRACK)

        assert output_path.exists()

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            MIDISequence.load("nonexistent.mid")

    def test_sequence_repr(self):
        """Test sequence string representation."""
        seq = MIDISequence(tempo=120.0)
        track = seq.add_track()
        track.add_note(0.0, 60, 100, 1.0)

        repr_str = repr(seq)
        assert "120.0" in repr_str
        assert "tracks=1" in repr_str


class TestMIDIRouter:
    """Tests for MIDIRouter class."""

    def test_create_router(self):
        """Test creating MIDI router."""
        router = MIDIRouter()

        assert len(router.routes) == 0
        assert len(router.transforms) == 0

    def test_add_route(self):
        """Test adding route."""
        router = MIDIRouter()
        router.add_route("keyboard", "synth")

        assert len(router.routes) == 1
        assert router.routes[0].source == "keyboard"
        assert router.routes[0].destination == "synth"

    def test_add_route_with_channel_map(self):
        """Test adding route with channel mapping."""
        router = MIDIRouter()
        router.add_route("keyboard", "synth", channel_map={0: 1, 1: 2})

        assert router.routes[0].channel_map == {0: 1, 1: 2}

    def test_add_transform(self):
        """Test adding transform."""
        router = MIDIRouter()
        router.add_transform("transpose", transpose_transform(12))

        assert "transpose" in router.transforms

    def test_remove_route(self):
        """Test removing route."""
        router = MIDIRouter()
        router.add_route("keyboard", "synth")
        router.add_route("keyboard", "drum")

        result = router.remove_route("keyboard", "synth")
        assert result is True
        assert len(router.routes) == 1

        result = router.remove_route("keyboard", "nonexistent")
        assert result is False

    def test_process_event_basic(self):
        """Test processing event through router."""
        router = MIDIRouter()
        router.add_route("keyboard", "synth")

        event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100)
        results = router.process_event("keyboard", event)

        assert len(results) == 1
        assert results[0][0] == "synth"
        assert results[0][1].data1 == 60

    def test_process_event_with_transform(self):
        """Test processing event with transform."""
        router = MIDIRouter()
        router.add_transform("transpose", transpose_transform(12))
        router.add_route("keyboard", "synth", transform="transpose")

        event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100)
        results = router.process_event("keyboard", event)

        assert len(results) == 1
        assert results[0][1].data1 == 72  # Transposed up by 12

    def test_process_event_with_channel_map(self):
        """Test processing event with channel mapping."""
        router = MIDIRouter()
        router.add_route("keyboard", "synth", channel_map={0: 5})

        event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100)
        results = router.process_event("keyboard", event)

        assert results[0][1].channel == 5

    def test_process_event_with_filter(self):
        """Test processing event with filter."""
        router = MIDIRouter()

        # Only pass note events
        def note_filter(e):
            return e.is_note_on or e.is_note_off

        router.add_route("keyboard", "synth", filter_func=note_filter)

        # Note event should pass
        note_event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100)
        results = router.process_event("keyboard", note_event)
        assert len(results) == 1

        # CC event should be filtered
        cc_event = MIDIEvent(0.0, MIDIStatus.CONTROL_CHANGE, 0, 7, 100)
        results = router.process_event("keyboard", cc_event)
        assert len(results) == 0

    def test_process_event_no_matching_routes(self):
        """Test processing event with no matching routes."""
        router = MIDIRouter()
        router.add_route("keyboard", "synth")

        event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100)
        results = router.process_event("other_source", event)

        assert len(results) == 0

    def test_clear_routes(self):
        """Test clearing all routes."""
        router = MIDIRouter()
        router.add_route("keyboard", "synth")
        router.add_route("keyboard", "drum")
        router.clear()

        assert len(router.routes) == 0

    def test_router_repr(self):
        """Test router string representation."""
        router = MIDIRouter()
        router.add_route("keyboard", "synth")
        router.add_transform("transpose", transpose_transform(12))

        repr_str = repr(router)
        assert "routes=1" in repr_str
        assert "transforms=1" in repr_str


class TestTransforms:
    """Tests for transform functions."""

    def test_transpose_transform_up(self):
        """Test transpose transform (upward)."""
        transform = transpose_transform(12)
        event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100)

        transformed = transform(event)
        assert transformed.data1 == 72

    def test_transpose_transform_down(self):
        """Test transpose transform (downward)."""
        transform = transpose_transform(-12)
        event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100)

        transformed = transform(event)
        assert transformed.data1 == 48

    def test_transpose_transform_clipping(self):
        """Test transpose transform clipping at boundaries."""
        transform = transpose_transform(100)
        event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100)

        transformed = transform(event)
        assert transformed.data1 == 127  # Clipped to max

    def test_transpose_transform_non_note(self):
        """Test transpose transform doesn't affect non-note events."""
        transform = transpose_transform(12)
        event = MIDIEvent(0.0, MIDIStatus.CONTROL_CHANGE, 0, 7, 100)

        transformed = transform(event)
        assert transformed.data1 == 7  # Unchanged

    def test_velocity_scale_transform(self):
        """Test velocity scale transform."""
        transform = velocity_scale_transform(0.5)
        event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100)

        transformed = transform(event)
        assert transformed.data2 == 50

    def test_velocity_scale_transform_clipping(self):
        """Test velocity scale transform clipping."""
        transform = velocity_scale_transform(2.0)
        event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100)

        transformed = transform(event)
        assert transformed.data2 == 127  # Clipped to max

    def test_velocity_scale_transform_minimum(self):
        """Test velocity scale transform maintains minimum."""
        transform = velocity_scale_transform(0.01)
        event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 10)

        transformed = transform(event)
        assert transformed.data2 >= 1  # At least 1

    def test_channel_remap_transform(self):
        """Test channel remap transform."""
        transform = channel_remap_transform({0: 5, 1: 6})
        event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100)

        transformed = transform(event)
        assert transformed.channel == 5

    def test_channel_remap_transform_no_mapping(self):
        """Test channel remap when no mapping exists."""
        transform = channel_remap_transform({0: 5})
        event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 1, 60, 100)

        transformed = transform(event)
        assert transformed.channel == 1  # Unchanged

    def test_quantize_transform(self):
        """Test time quantization transform."""
        transform = quantize_transform(0.25)

        # Event at 0.1s should quantize to 0.0s
        event1 = MIDIEvent(0.1, MIDIStatus.NOTE_ON, 0, 60, 100)
        transformed1 = transform(event1)
        assert transformed1.time == 0.0

        # Event at 0.4s should quantize to 0.5s
        event2 = MIDIEvent(0.4, MIDIStatus.NOTE_ON, 0, 60, 100)
        transformed2 = transform(event2)
        assert transformed2.time == 0.5


class TestIntegration:
    """Integration tests for MIDI utilities."""

    def test_create_and_save_multi_track_sequence(self, tmp_path):
        """Test creating and saving a multi-track sequence."""
        seq = MIDISequence(tempo=120.0, time_signature=(4, 4))

        # Create melody track
        melody = seq.add_track("Melody")
        melody.channel = 0
        melody.add_program_change(0.0, 0)  # Piano
        for i, note in enumerate([60, 62, 64, 65]):
            melody.add_note(i * 0.5, note, 100, 0.4)

        # Create bass track
        bass = seq.add_track("Bass")
        bass.channel = 1
        bass.add_program_change(0.0, 32)  # Bass
        bass.add_note(0.0, 48, 100, 2.0)

        # Save
        output_path = tmp_path / "composition.mid"
        seq.save(str(output_path))

        assert output_path.exists()
        assert len(seq.tracks) == 2
        assert seq.duration == 2.0

    def test_router_with_multiple_transforms(self):
        """Test router with multiple transforms."""
        router = MIDIRouter()

        # Add transforms
        router.add_transform("transpose_octave", transpose_transform(12))
        router.add_transform("softer", velocity_scale_transform(0.7))

        # Add routes with different transforms
        router.add_route("keyboard", "synth1", transform="transpose_octave")
        router.add_route("keyboard", "synth2", transform="softer")

        # Process event
        event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100)
        results = router.process_event("keyboard", event)

        assert len(results) == 2
        assert results[0][0] == "synth1"
        assert results[0][1].data1 == 72  # Transposed
        assert results[1][0] == "synth2"
        assert results[1][1].data2 == 70  # Velocity scaled
