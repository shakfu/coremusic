"""Tests for MusicPlayer, MusicSequence, and MusicTrack OO API."""

import logging
import os
import pytest
import coremusic as cm

logger = logging.getLogger(__name__)


class TestMusicTrack:
    """Test MusicTrack OO API"""

    def test_add_midi_note(self):
        """Test adding MIDI notes to track"""
        sequence = cm.MusicSequence()
        try:
            track = sequence.new_track()

            # Add middle C
            track.add_midi_note(0.0, 0, 60, 100, 64, 1.0)

            # Add E
            track.add_midi_note(1.0, 0, 64, 100, 64, 1.0)

            # Add G
            track.add_midi_note(2.0, 0, 67, 100, 64, 1.0)

            # Should not raise exceptions
        finally:
            sequence.dispose()

    def test_add_midi_channel_event(self):
        """Test adding MIDI channel events"""
        sequence = cm.MusicSequence()
        try:
            track = sequence.new_track()

            # Add program change
            track.add_midi_channel_event(0.0, 0xC0, 0)  # Piano

            # Add volume control
            track.add_midi_channel_event(0.0, 0xB0, 7, 100)

            # Should not raise exceptions
        finally:
            sequence.dispose()

    def test_add_tempo_event(self):
        """Test adding tempo events"""
        sequence = cm.MusicSequence()
        try:
            tempo_track = sequence.tempo_track

            # Set initial tempo
            tempo_track.add_tempo_event(0.0, 120.0)

            # Change tempo
            tempo_track.add_tempo_event(4.0, 140.0)

            # Should not raise exceptions
        finally:
            sequence.dispose()

    def test_repr(self):
        """Test track string representation"""
        sequence = cm.MusicSequence()
        try:
            track = sequence.new_track()
            repr_str = repr(track)
            assert "MusicTrack" in repr_str
            assert "id=" in repr_str
        finally:
            sequence.dispose()


class TestMusicSequence:
    """Test MusicSequence OO API"""

    def test_creation(self):
        """Test sequence creation"""
        sequence = cm.MusicSequence()
        try:
            assert not sequence.is_disposed
            assert sequence.track_count == 0
        finally:
            sequence.dispose()

    def test_new_track(self):
        """Test creating new tracks"""
        sequence = cm.MusicSequence()
        try:
            assert sequence.track_count == 0

            track1 = sequence.new_track()
            assert isinstance(track1, cm.MusicTrack)
            assert sequence.track_count == 1

            track2 = sequence.new_track()
            assert isinstance(track2, cm.MusicTrack)
            assert sequence.track_count == 2

            # Tracks should be different
            assert track1.object_id != track2.object_id
        finally:
            sequence.dispose()

    def test_dispose_track(self):
        """Test removing tracks"""
        sequence = cm.MusicSequence()
        try:
            track1 = sequence.new_track()
            track2 = sequence.new_track()
            assert sequence.track_count == 2

            sequence.dispose_track(track1)
            assert sequence.track_count == 1
            assert track1.is_disposed
        finally:
            sequence.dispose()

    def test_get_track(self):
        """Test getting track by index"""
        sequence = cm.MusicSequence()
        try:
            track1 = sequence.new_track()
            track2 = sequence.new_track()

            # Get first track
            retrieved = sequence.get_track(0)
            assert isinstance(retrieved, cm.MusicTrack)

            # Get second track
            retrieved = sequence.get_track(1)
            assert isinstance(retrieved, cm.MusicTrack)
        finally:
            sequence.dispose()

    def test_get_track_invalid_index(self):
        """Test getting track with invalid index"""
        sequence = cm.MusicSequence()
        try:
            sequence.new_track()

            # Index 1 should fail (only 1 track at index 0)
            # Raises IndexError for out-of-range, ValueError for negative
            with pytest.raises((IndexError, ValueError)):
                sequence.get_track(1)

            # Negative index should raise ValueError
            with pytest.raises(ValueError):
                sequence.get_track(-1)
        finally:
            sequence.dispose()

    def test_tempo_track(self):
        """Test getting tempo track"""
        sequence = cm.MusicSequence()
        try:
            tempo_track = sequence.tempo_track
            assert isinstance(tempo_track, cm.MusicTrack)

            # Should be the same object if accessed again
            tempo_track2 = sequence.tempo_track
            assert tempo_track is tempo_track2

            # Tempo track doesn't count in track_count
            assert sequence.track_count == 0
        finally:
            sequence.dispose()

    def test_sequence_type(self):
        """Test getting and setting sequence type"""
        sequence = cm.MusicSequence()
        try:
            # Default should be beats
            seq_type = sequence.sequence_type
            assert seq_type == cm.capi.get_music_sequence_type_beats()

            # Change to seconds
            sequence.sequence_type = cm.capi.get_music_sequence_type_seconds()
            assert sequence.sequence_type == cm.capi.get_music_sequence_type_seconds()

            # Change back to beats
            sequence.sequence_type = cm.capi.get_music_sequence_type_beats()
            assert sequence.sequence_type == cm.capi.get_music_sequence_type_beats()
        finally:
            sequence.dispose()

    def test_load_from_file_invalid(self):
        """Test loading from nonexistent file"""
        sequence = cm.MusicSequence()
        try:
            with pytest.raises(cm.MusicPlayerError):
                sequence.load_from_file("/nonexistent/file.mid")
        finally:
            sequence.dispose()

    def test_dispose(self):
        """Test sequence disposal"""
        sequence = cm.MusicSequence()
        track1 = sequence.new_track()
        track2 = sequence.new_track()

        assert not sequence.is_disposed
        assert not track1.is_disposed
        assert not track2.is_disposed

        sequence.dispose()

        assert sequence.is_disposed
        # Note: tracks are also disposed when sequence is disposed

    def test_repr(self):
        """Test sequence string representation"""
        sequence = cm.MusicSequence()
        try:
            repr_str = repr(sequence)
            assert "MusicSequence" in repr_str
            assert "tracks=0" in repr_str

            sequence.new_track()
            repr_str = repr(sequence)
            assert "tracks=1" in repr_str
        finally:
            sequence.dispose()

    def test_repr_disposed(self):
        """Test repr of disposed sequence"""
        sequence = cm.MusicSequence()
        sequence.dispose()
        repr_str = repr(sequence)
        assert "disposed" in repr_str


class TestMusicPlayer:
    """Test MusicPlayer OO API"""

    def test_creation(self):
        """Test player creation"""
        player = cm.MusicPlayer()
        try:
            assert not player.is_disposed
            assert player.sequence is None
            assert not player.is_playing
        finally:
            player.dispose()

    def test_sequence_assignment(self):
        """Test assigning sequence to player"""
        player = cm.MusicPlayer()
        sequence = cm.MusicSequence()
        try:
            # Assign sequence
            player.sequence = sequence
            assert player.sequence is sequence

            # Clear sequence
            player.sequence = None
            assert player.sequence is None
        finally:
            sequence.dispose()
            player.dispose()

    def test_time_operations(self):
        """Test getting and setting time"""
        player = cm.MusicPlayer()
        sequence = cm.MusicSequence()
        try:
            player.sequence = sequence

            # Set time
            player.time = 10.5
            assert abs(player.time - 10.5) < 0.01

            # Reset time
            player.time = 0.0
            assert player.time == 0.0
        finally:
            player.sequence = None
            sequence.dispose()
            player.dispose()

    def test_play_rate(self):
        """Test getting and setting play rate"""
        player = cm.MusicPlayer()
        try:
            # Default rate should be 1.0
            assert player.play_rate == 1.0

            # Set to double speed
            player.play_rate = 2.0
            assert player.play_rate == 2.0

            # Set to half speed
            player.play_rate = 0.5
            assert player.play_rate == 0.5
        finally:
            player.dispose()

    def test_play_rate_invalid(self):
        """Test that invalid play rates are rejected"""
        player = cm.MusicPlayer()
        try:
            with pytest.raises(ValueError):
                player.play_rate = -1.0

            with pytest.raises(ValueError):
                player.play_rate = 0.0
        finally:
            player.dispose()

    def test_playback_control(self):
        """Test start/stop playback"""
        player = cm.MusicPlayer()
        sequence = cm.MusicSequence()
        try:
            # Create a simple sequence
            track = sequence.new_track()
            tempo_track = sequence.tempo_track
            tempo_track.add_tempo_event(0.0, 120.0)
            track.add_midi_note(0.0, 0, 60, 100, 64, 1.0)

            # Assign to player
            player.sequence = sequence

            # Initial state should be stopped
            assert not player.is_playing

            # Preroll and start
            player.preroll()
            try:
                player.start()
                # Note: may or may not be playing depending on system state
                # Just ensure no exception
            except Exception as e:
                logger.warning(f"Playback start skipped (system state): {e}")

            # Stop
            player.stop()
            assert not player.is_playing
        finally:
            player.sequence = None
            sequence.dispose()
            player.dispose()

    def test_context_manager(self):
        """Test player as context manager"""
        with cm.MusicPlayer() as player:
            assert not player.is_disposed
            assert player.sequence is None

        # Should be disposed after exiting context
        assert player.is_disposed

    def test_dispose(self):
        """Test player disposal"""
        player = cm.MusicPlayer()
        sequence = cm.MusicSequence()

        player.sequence = sequence
        assert not player.is_disposed

        player.dispose()
        assert player.is_disposed

        sequence.dispose()

    def test_repr(self):
        """Test player string representation"""
        player = cm.MusicPlayer()
        try:
            repr_str = repr(player)
            assert "MusicPlayer" in repr_str
            assert "stopped" in repr_str or "time=" in repr_str
        finally:
            player.dispose()

    def test_repr_disposed(self):
        """Test repr of disposed player"""
        player = cm.MusicPlayer()
        player.dispose()
        repr_str = repr(player)
        assert "disposed" in repr_str


class TestMusicPlayerIntegration:
    """Test integration between MusicPlayer, MusicSequence, and MusicTrack"""

    def test_complete_workflow(self):
        """Test complete sequence creation and playback setup"""
        player = cm.MusicPlayer()
        sequence = cm.MusicSequence()
        try:
            # Create melody track
            melody = sequence.new_track()
            melody.add_midi_note(0.0, 0, 60, 100, 64, 1.0)  # C
            melody.add_midi_note(1.0, 0, 64, 100, 64, 1.0)  # E
            melody.add_midi_note(2.0, 0, 67, 100, 64, 1.0)  # G
            melody.add_midi_note(3.0, 0, 72, 100, 64, 1.0)  # C

            # Set tempo
            tempo_track = sequence.tempo_track
            tempo_track.add_tempo_event(0.0, 120.0)

            # Assign to player
            player.sequence = sequence
            assert player.sequence is sequence

            # Prepare for playback
            player.preroll()

            # Set position
            player.time = 2.5
            assert abs(player.time - 2.5) < 0.01

            # Set playback rate
            player.play_rate = 0.8
            assert player.play_rate == 0.8

            # Try to start (may fail on some systems)
            try:
                player.start()
            except Exception as e:
                logger.warning(f"Playback start skipped (system state): {e}")

            # Stop
            player.stop()
            assert not player.is_playing
        finally:
            player.sequence = None
            sequence.dispose()
            player.dispose()

    def test_multiple_tracks(self):
        """Test creating multiple tracks with different parts"""
        sequence = cm.MusicSequence()
        try:
            # Melody track
            melody = sequence.new_track()
            melody.add_midi_channel_event(0.0, 0xC0, 0)  # Piano
            melody.add_midi_note(0.0, 0, 60, 100)

            # Bass track
            bass = sequence.new_track()
            bass.add_midi_channel_event(0.0, 0xC1, 32)  # Bass
            bass.add_midi_note(0.0, 1, 36, 100)

            # Verify both tracks exist
            assert sequence.track_count == 2

            # Verify we can get them back
            track0 = sequence.get_track(0)
            track1 = sequence.get_track(1)
            assert track0.object_id != track1.object_id
        finally:
            sequence.dispose()

    def test_switching_sequences(self):
        """Test switching between sequences on a player"""
        player = cm.MusicPlayer()
        sequence1 = cm.MusicSequence()
        sequence2 = cm.MusicSequence()
        try:
            # Add content to both sequences
            track1 = sequence1.new_track()
            track1.add_midi_note(0.0, 0, 60, 100)

            track2 = sequence2.new_track()
            track2.add_midi_note(0.0, 0, 64, 100)

            # Switch between sequences
            player.sequence = sequence1
            assert player.sequence is sequence1

            player.sequence = sequence2
            assert player.sequence is sequence2

            # Clear sequence
            player.sequence = None
            assert player.sequence is None
        finally:
            player.sequence = None
            sequence1.dispose()
            sequence2.dispose()
            player.dispose()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
