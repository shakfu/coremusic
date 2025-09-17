#!/usr/bin/env python3
"""pytest test suite for MusicPlayer functionality."""

import os
import pytest
import tempfile
import coreaudio as ca


class TestMusicPlayerConstants:
    """Test MusicPlayer constants access"""

    def test_music_event_type_constants(self):
        """Test music event type constants"""
        assert ca.get_music_event_type_null() == 0
        assert ca.get_music_event_type_extended_note() == 1
        assert ca.get_music_event_type_extended_tempo() == 3
        assert ca.get_music_event_type_user() == 4
        assert ca.get_music_event_type_meta() == 5
        assert ca.get_music_event_type_midi_note_message() == 6
        assert ca.get_music_event_type_midi_channel_message() == 7
        assert ca.get_music_event_type_midi_raw_data() == 8
        assert ca.get_music_event_type_parameter() == 9
        assert ca.get_music_event_type_au_preset() == 10

    def test_music_sequence_type_constants(self):
        """Test music sequence type constants"""
        # These are fourcc values for 'beat', 'secs', 'samp'
        assert ca.get_music_sequence_type_beats() == 1650811252
        assert ca.get_music_sequence_type_seconds() == 1936024435
        assert ca.get_music_sequence_type_samples() == 1935764848

    def test_music_sequence_file_type_constants(self):
        """Test music sequence file type constants"""
        assert ca.get_music_sequence_file_any_type() == 0
        assert ca.get_music_sequence_file_midi_type() == 1835623529  # 'midi'
        assert ca.get_music_sequence_file_imelody_type() == 1768777068  # 'imel'

    def test_sequence_track_property_constants(self):
        """Test sequence track property constants"""
        assert ca.get_sequence_track_property_loop_info() == 0
        assert ca.get_sequence_track_property_offset_time() == 1
        assert ca.get_sequence_track_property_mute_status() == 2
        assert ca.get_sequence_track_property_solo_status() == 3
        assert ca.get_sequence_track_property_automated_parameters() == 4
        assert ca.get_sequence_track_property_track_length() == 5
        assert ca.get_sequence_track_property_time_resolution() == 6


class TestMusicPlayerHelpers:
    """Test MusicPlayer helper functions"""

    def test_create_midi_note_message(self):
        """Test creating MIDI note message"""
        msg = ca.create_midi_note_message(0, 60, 127, 64, 2.0)
        assert msg['channel'] == 0
        assert msg['note'] == 60
        assert msg['velocity'] == 127
        assert msg['release_velocity'] == 64
        assert msg['duration'] == 2.0

        # Test defaults
        msg = ca.create_midi_note_message(1, 72, 100)
        assert msg['channel'] == 1
        assert msg['note'] == 72
        assert msg['velocity'] == 100
        assert msg['release_velocity'] == 0
        assert msg['duration'] == 1.0

    def test_create_midi_channel_message(self):
        """Test creating MIDI channel message"""
        msg = ca.create_midi_channel_message(0x90, 60, 127)
        assert msg['status'] == 0x90
        assert msg['data1'] == 60
        assert msg['data2'] == 127

        # Test with default data2
        msg = ca.create_midi_channel_message(0xC0, 10)
        assert msg['status'] == 0xC0
        assert msg['data1'] == 10
        assert msg['data2'] == 0

    def test_midi_message_bounds_checking(self):
        """Test that helper functions respect bounds"""
        # Test channel bounds (should mask to 0-15)
        msg = ca.create_midi_note_message(16, 60, 127)
        assert msg['channel'] == 0  # 16 & 0x0F = 0

        # Test note bounds (should mask to 0-127)
        msg = ca.create_midi_note_message(0, 128, 127)
        assert msg['note'] == 0  # 128 & 0x7F = 0

        # Test velocity bounds
        msg = ca.create_midi_note_message(0, 60, 255)
        assert msg['velocity'] == 127  # 255 & 0x7F = 127


class TestMusicPlayerBasicOperations:
    """Test basic MusicPlayer operations"""

    def test_music_player_creation_and_disposal(self):
        """Test creating and disposing a music player"""
        player = ca.new_music_player()
        assert isinstance(player, int)
        assert player != 0

        # Dispose the player
        result = ca.dispose_music_player(player)
        assert result == 0

    def test_music_sequence_creation_and_disposal(self):
        """Test creating and disposing a music sequence"""
        sequence = ca.new_music_sequence()
        assert isinstance(sequence, int)
        assert sequence != 0

        # Dispose the sequence
        result = ca.dispose_music_sequence(sequence)
        assert result == 0

    def test_music_player_sequence_assignment(self):
        """Test assigning sequence to player"""
        player = ca.new_music_player()
        sequence = ca.new_music_sequence()

        try:
            # Set sequence on player
            result = ca.music_player_set_sequence(player, sequence)
            assert result == 0

            # Get sequence from player
            retrieved_sequence = ca.music_player_get_sequence(player)
            assert retrieved_sequence == sequence

            # Clear sequence (set to 0/NULL)
            result = ca.music_player_set_sequence(player, 0)
            assert result == 0

        finally:
            ca.dispose_music_sequence(sequence)
            ca.dispose_music_player(player)

    def test_music_player_time_operations(self):
        """Test player time get/set operations"""
        player = ca.new_music_player()
        sequence = ca.new_music_sequence()

        try:
            # Set sequence on player first
            ca.music_player_set_sequence(player, sequence)

            # Set time to 10.5 beats
            result = ca.music_player_set_time(player, 10.5)
            assert result == 0

            # Get time back
            time = ca.music_player_get_time(player)
            assert abs(time - 10.5) < 0.01  # Allow for small floating point differences

            # Set time to 0
            result = ca.music_player_set_time(player, 0.0)
            assert result == 0

            time = ca.music_player_get_time(player)
            assert time == 0.0

        finally:
            # Clear sequence from player before disposing
            try:
                ca.music_player_set_sequence(player, 0)
            except:
                pass
            ca.dispose_music_sequence(sequence)
            ca.dispose_music_player(player)

    def test_music_player_playback_state(self):
        """Test player playback state operations"""
        player = ca.new_music_player()
        sequence = ca.new_music_sequence()

        try:
            # Initially should not be playing
            is_playing = ca.music_player_is_playing(player)
            assert is_playing is False

            # Set sequence first
            ca.music_player_set_sequence(player, sequence)

            # Preroll should work with sequence
            result = ca.music_player_preroll(player)
            assert result == 0

            # Note: Start may fail without an AUGraph, which is expected
            # We'll test the state operations that should work
            try:
                result = ca.music_player_start(player)
                if result == 0:
                    # If start succeeded, test the playing state
                    is_playing = ca.music_player_is_playing(player)
                    # Stop the player
                    ca.music_player_stop(player)

                    # Should no longer be playing
                    is_playing = ca.music_player_is_playing(player)
                    assert is_playing is False
                else:
                    # Start failed, which is expected without AUGraph
                    # Just verify we can still check the playing state
                    is_playing = ca.music_player_is_playing(player)
                    assert is_playing is False
            except RuntimeError as e:
                # Expected if no AUGraph is set - this is normal behavior
                # Verify we can still check the playing state
                is_playing = ca.music_player_is_playing(player)
                assert is_playing is False

        finally:
            # Clear sequence from player before disposing
            try:
                ca.music_player_stop(player)
                ca.music_player_set_sequence(player, 0)
            except:
                pass
            ca.dispose_music_sequence(sequence)
            ca.dispose_music_player(player)

    def test_music_player_rate_scalar(self):
        """Test player rate scalar operations"""
        player = ca.new_music_player()

        try:
            # Default rate should be 1.0
            rate = ca.music_player_get_play_rate_scalar(player)
            assert rate == 1.0

            # Set to double speed
            result = ca.music_player_set_play_rate_scalar(player, 2.0)
            assert result == 0

            rate = ca.music_player_get_play_rate_scalar(player)
            assert rate == 2.0

            # Set to half speed
            result = ca.music_player_set_play_rate_scalar(player, 0.5)
            assert result == 0

            rate = ca.music_player_get_play_rate_scalar(player)
            assert rate == 0.5

        finally:
            ca.dispose_music_player(player)

    def test_music_player_invalid_rate_scalar(self):
        """Test player rate scalar validation"""
        player = ca.new_music_player()

        try:
            # Negative rate should raise ValueError
            with pytest.raises(ValueError):
                ca.music_player_set_play_rate_scalar(player, -1.0)

            # Zero rate should raise ValueError
            with pytest.raises(ValueError):
                ca.music_player_set_play_rate_scalar(player, 0.0)

        finally:
            ca.dispose_music_player(player)


class TestMusicSequenceOperations:
    """Test MusicSequence operations"""

    def test_music_sequence_track_operations(self):
        """Test sequence track creation and management"""
        sequence = ca.new_music_sequence()

        try:
            # New sequence should have 0 tracks (tempo track is separate)
            count = ca.music_sequence_get_track_count(sequence)
            assert count == 0

            # Add a track
            track1 = ca.music_sequence_new_track(sequence)
            assert isinstance(track1, int)
            assert track1 != 0

            # Should now have 1 track
            count = ca.music_sequence_get_track_count(sequence)
            assert count == 1

            # Get track by index
            retrieved_track = ca.music_sequence_get_ind_track(sequence, 0)
            assert retrieved_track == track1

            # Add another track
            track2 = ca.music_sequence_new_track(sequence)
            assert isinstance(track2, int)
            assert track2 != 0
            assert track2 != track1

            # Should now have 2 tracks
            count = ca.music_sequence_get_track_count(sequence)
            assert count == 2

            # Dispose a track
            result = ca.music_sequence_dispose_track(sequence, track1)
            assert result == 0

            # Should now have 1 track
            count = ca.music_sequence_get_track_count(sequence)
            assert count == 1

        finally:
            ca.dispose_music_sequence(sequence)

    def test_music_sequence_tempo_track(self):
        """Test getting the tempo track"""
        sequence = ca.new_music_sequence()

        try:
            # Get tempo track
            tempo_track = ca.music_sequence_get_tempo_track(sequence)
            assert isinstance(tempo_track, int)
            assert tempo_track != 0

            # Tempo track should be separate from regular tracks
            count = ca.music_sequence_get_track_count(sequence)
            assert count == 0  # Still 0 regular tracks

        finally:
            ca.dispose_music_sequence(sequence)

    def test_music_sequence_type_operations(self):
        """Test sequence type get/set operations"""
        sequence = ca.new_music_sequence()

        try:
            # Default should be beats
            seq_type = ca.music_sequence_get_sequence_type(sequence)
            assert seq_type == ca.get_music_sequence_type_beats()

            # Set to seconds
            result = ca.music_sequence_set_sequence_type(sequence, ca.get_music_sequence_type_seconds())
            assert result == 0

            seq_type = ca.music_sequence_get_sequence_type(sequence)
            assert seq_type == ca.get_music_sequence_type_seconds()

            # Set back to beats
            result = ca.music_sequence_set_sequence_type(sequence, ca.get_music_sequence_type_beats())
            assert result == 0

            seq_type = ca.music_sequence_get_sequence_type(sequence)
            assert seq_type == ca.get_music_sequence_type_beats()

        finally:
            ca.dispose_music_sequence(sequence)

    def test_music_sequence_file_load_invalid_path(self):
        """Test loading non-existent file"""
        sequence = ca.new_music_sequence()

        try:
            # Try to load a non-existent file
            with pytest.raises(RuntimeError):
                ca.music_sequence_file_load(sequence, "/path/that/does/not/exist.mid")

        finally:
            ca.dispose_music_sequence(sequence)


class TestMusicTrackOperations:
    """Test MusicTrack operations"""

    def test_music_track_midi_note_events(self):
        """Test adding MIDI note events to tracks"""
        sequence = ca.new_music_sequence()
        track = ca.music_sequence_new_track(sequence)

        try:
            # Add a MIDI note event
            result = ca.music_track_new_midi_note_event(
                track, 0.0,  # timestamp
                0,           # channel
                60,          # note (middle C)
                127,         # velocity
                64,          # release velocity
                1.0          # duration
            )
            assert result == 0

            # Add another note
            result = ca.music_track_new_midi_note_event(
                track, 1.0,  # timestamp
                1,           # channel
                67,          # note (G)
                100,         # velocity
                50,          # release velocity
                0.5          # duration
            )
            assert result == 0

        finally:
            ca.dispose_music_sequence(sequence)

    def test_music_track_midi_channel_events(self):
        """Test adding MIDI channel events to tracks"""
        sequence = ca.new_music_sequence()
        track = ca.music_sequence_new_track(sequence)

        try:
            # Add a program change event
            result = ca.music_track_new_midi_channel_event(
                track, 0.0,  # timestamp
                0xC0,        # program change on channel 0
                10,          # program number (electric piano)
                0            # unused for program change
            )
            assert result == 0

            # Add a control change event
            result = ca.music_track_new_midi_channel_event(
                track, 0.5,  # timestamp
                0xB0,        # control change on channel 0
                7,           # volume controller
                100          # volume value
            )
            assert result == 0

        finally:
            ca.dispose_music_sequence(sequence)

    def test_music_track_tempo_events(self):
        """Test adding tempo events to tracks"""
        sequence = ca.new_music_sequence()
        tempo_track = ca.music_sequence_get_tempo_track(sequence)

        try:
            # Add a tempo event to the tempo track
            result = ca.music_track_new_extended_tempo_event(
                tempo_track, 0.0,  # timestamp
                120.0              # 120 BPM
            )
            assert result == 0

            # Add another tempo change
            result = ca.music_track_new_extended_tempo_event(
                tempo_track, 4.0,  # timestamp
                140.0              # 140 BPM
            )
            assert result == 0

        finally:
            ca.dispose_music_sequence(sequence)

    def test_music_track_invalid_bpm(self):
        """Test tempo event validation"""
        sequence = ca.new_music_sequence()
        tempo_track = ca.music_sequence_get_tempo_track(sequence)

        try:
            # Negative BPM should raise ValueError
            with pytest.raises(ValueError):
                ca.music_track_new_extended_tempo_event(tempo_track, 0.0, -120.0)

            # Zero BPM should raise ValueError
            with pytest.raises(ValueError):
                ca.music_track_new_extended_tempo_event(tempo_track, 0.0, 0.0)

        finally:
            ca.dispose_music_sequence(sequence)


class TestMusicPlayerErrorHandling:
    """Test MusicPlayer error handling"""

    def test_invalid_player_operations(self):
        """Test operations with invalid player handles"""
        invalid_player = 0

        # All operations should raise RuntimeError with invalid player
        with pytest.raises(RuntimeError):
            ca.music_player_set_sequence(invalid_player, 0)

        with pytest.raises(RuntimeError):
            ca.music_player_get_sequence(invalid_player)

        with pytest.raises(RuntimeError):
            ca.music_player_set_time(invalid_player, 0.0)

        with pytest.raises(RuntimeError):
            ca.music_player_get_time(invalid_player)

        with pytest.raises(RuntimeError):
            ca.music_player_preroll(invalid_player)

        with pytest.raises(RuntimeError):
            ca.music_player_start(invalid_player)

        with pytest.raises(RuntimeError):
            ca.music_player_stop(invalid_player)

        with pytest.raises(RuntimeError):
            ca.music_player_is_playing(invalid_player)

        with pytest.raises(RuntimeError):
            ca.music_player_set_play_rate_scalar(invalid_player, 1.0)

        with pytest.raises(RuntimeError):
            ca.music_player_get_play_rate_scalar(invalid_player)

    def test_invalid_sequence_operations(self):
        """Test operations with invalid sequence handles"""
        invalid_sequence = 0

        # All operations should raise RuntimeError with invalid sequence
        with pytest.raises(RuntimeError):
            ca.music_sequence_new_track(invalid_sequence)

        with pytest.raises(RuntimeError):
            ca.music_sequence_get_track_count(invalid_sequence)

        with pytest.raises(RuntimeError):
            ca.music_sequence_get_ind_track(invalid_sequence, 0)

        with pytest.raises(RuntimeError):
            ca.music_sequence_get_tempo_track(invalid_sequence)

        with pytest.raises(RuntimeError):
            ca.music_sequence_set_sequence_type(invalid_sequence, ca.get_music_sequence_type_beats())

        with pytest.raises(RuntimeError):
            ca.music_sequence_get_sequence_type(invalid_sequence)

    def test_invalid_track_operations(self):
        """Test operations with invalid track handles"""
        invalid_track = 0

        # All operations should raise RuntimeError with invalid track
        with pytest.raises(RuntimeError):
            ca.music_track_new_midi_note_event(invalid_track, 0.0, 0, 60, 127, 64, 1.0)

        with pytest.raises(RuntimeError):
            ca.music_track_new_midi_channel_event(invalid_track, 0.0, 0xC0, 10, 0)

        with pytest.raises(RuntimeError):
            ca.music_track_new_extended_tempo_event(invalid_track, 0.0, 120.0)

    def test_sequence_track_index_errors(self):
        """Test sequence track index error handling"""
        sequence = ca.new_music_sequence()

        try:
            # No tracks exist, so index 0 should fail
            with pytest.raises(RuntimeError):
                ca.music_sequence_get_ind_track(sequence, 0)

            # Create a track
            track = ca.music_sequence_new_track(sequence)

            # Index 0 should now work
            retrieved_track = ca.music_sequence_get_ind_track(sequence, 0)
            assert retrieved_track == track

            # Index 1 should still fail
            with pytest.raises(RuntimeError):
                ca.music_sequence_get_ind_track(sequence, 1)

        finally:
            ca.dispose_music_sequence(sequence)


class TestMusicPlayerIntegration:
    """Test MusicPlayer integration scenarios"""

    def test_complete_sequence_playback_setup(self):
        """Test setting up a complete sequence for playback"""
        player = ca.new_music_player()
        sequence = ca.new_music_sequence()

        try:
            # Create a track and add some events
            track = ca.music_sequence_new_track(sequence)

            # Add a tempo event
            tempo_track = ca.music_sequence_get_tempo_track(sequence)
            ca.music_track_new_extended_tempo_event(tempo_track, 0.0, 120.0)

            # Add some MIDI note events
            ca.music_track_new_midi_note_event(track, 0.0, 0, 60, 127, 64, 1.0)  # C
            ca.music_track_new_midi_note_event(track, 1.0, 0, 64, 127, 64, 1.0)  # E
            ca.music_track_new_midi_note_event(track, 2.0, 0, 67, 127, 64, 1.0)  # G
            ca.music_track_new_midi_note_event(track, 3.0, 0, 72, 127, 64, 1.0)  # C

            # Set sequence on player
            ca.music_player_set_sequence(player, sequence)

            # Verify sequence is set
            retrieved_sequence = ca.music_player_get_sequence(player)
            assert retrieved_sequence == sequence

            # Test player operations
            ca.music_player_preroll(player)

            # Note: Start may fail without an AUGraph, but other operations should work
            try:
                ca.music_player_start(player)
                started = True
            except RuntimeError:
                started = False

            if started:
                assert ca.music_player_is_playing(player) is True

            # Set playback rate (should work regardless of start status)
            ca.music_player_set_play_rate_scalar(player, 0.8)
            rate = ca.music_player_get_play_rate_scalar(player)
            assert rate == 0.8

            # Jump to different time (should work regardless of start status)
            ca.music_player_set_time(player, 2.5)
            time = ca.music_player_get_time(player)
            assert abs(time - 2.5) < 0.01

            # Stop (only if we started successfully)
            if started:
                ca.music_player_stop(player)
                assert ca.music_player_is_playing(player) is False

        finally:
            # Proper cleanup order
            try:
                ca.music_player_stop(player)
                ca.music_player_set_sequence(player, 0)
            except:
                pass
            ca.dispose_music_sequence(sequence)
            ca.dispose_music_player(player)

    def test_multiple_sequences_on_player(self):
        """Test switching sequences on a player"""
        player = ca.new_music_player()
        sequence1 = ca.new_music_sequence()
        sequence2 = ca.new_music_sequence()

        try:
            # Set first sequence
            ca.music_player_set_sequence(player, sequence1)
            retrieved = ca.music_player_get_sequence(player)
            assert retrieved == sequence1

            # Switch to second sequence
            ca.music_player_set_sequence(player, sequence2)
            retrieved = ca.music_player_get_sequence(player)
            assert retrieved == sequence2

            # Clear sequence
            ca.music_player_set_sequence(player, 0)

            # Should fail to get sequence now
            with pytest.raises(RuntimeError):
                ca.music_player_get_sequence(player)

        finally:
            ca.dispose_music_sequence(sequence1)
            ca.dispose_music_sequence(sequence2)
            ca.dispose_music_player(player)


class TestMusicPlayerResourceManagement:
    """Test MusicPlayer resource management"""

    def test_multiple_players(self):
        """Test creating multiple players"""
        players = []
        sequences = []

        try:
            # Create multiple players with sequences
            for i in range(5):
                player = ca.new_music_player()
                sequence = ca.new_music_sequence()
                players.append(player)
                sequences.append(sequence)
                assert isinstance(player, int)
                assert player != 0

                # Set sequence on player so time operations work
                ca.music_player_set_sequence(player, sequence)

            # All players should be unique
            assert len(set(players)) == len(players)

            # Each player should work independently
            for i, player in enumerate(players):
                ca.music_player_set_time(player, float(i))
                time = ca.music_player_get_time(player)
                assert time == float(i)

        finally:
            # Cleanup all players and sequences
            for i, (player, sequence) in enumerate(zip(players, sequences)):
                try:
                    ca.music_player_set_sequence(player, 0)
                    ca.dispose_music_sequence(sequence)
                    ca.dispose_music_player(player)
                except:
                    pass  # Ignore cleanup errors in tests

    def test_multiple_sequences(self):
        """Test creating multiple sequences"""
        sequences = []

        try:
            # Create multiple sequences
            for i in range(3):
                sequence = ca.new_music_sequence()
                sequences.append(sequence)
                assert isinstance(sequence, int)
                assert sequence != 0

            # All sequences should be unique
            assert len(set(sequences)) == len(sequences)

            # Each sequence should work independently
            for i, sequence in enumerate(sequences):
                # Add tracks to each sequence
                for j in range(i + 1):
                    track = ca.music_sequence_new_track(sequence)
                    assert isinstance(track, int)

                # Verify track count
                count = ca.music_sequence_get_track_count(sequence)
                assert count == i + 1

        finally:
            # Cleanup all sequences
            for sequence in sequences:
                try:
                    ca.dispose_music_sequence(sequence)
                except:
                    pass  # Ignore cleanup errors in tests

    def test_sequence_lifecycle(self):
        """Test complete sequence lifecycle"""
        # Create sequence and player
        sequence = ca.new_music_sequence()
        player = ca.new_music_player()

        try:
            # Add content to sequence
            track = ca.music_sequence_new_track(sequence)
            tempo_track = ca.music_sequence_get_tempo_track(sequence)

            # Add events
            ca.music_track_new_extended_tempo_event(tempo_track, 0.0, 120.0)
            ca.music_track_new_midi_note_event(track, 0.0, 0, 60, 127, 64, 4.0)

            # Use with player
            ca.music_player_set_sequence(player, sequence)
            ca.music_player_preroll(player)
            ca.music_player_start(player)

            # Verify it's working
            assert ca.music_player_is_playing(player) is True

            # Stop and cleanup
            ca.music_player_stop(player)
            ca.music_player_set_sequence(player, 0)

        finally:
            ca.dispose_music_sequence(sequence)
            ca.dispose_music_player(player)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])