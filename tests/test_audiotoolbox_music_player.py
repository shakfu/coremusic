"""pytest test suite for MusicPlayer functionality."""

import logging
import os
import pytest
import tempfile
import coremusic as cm
import coremusic.capi as capi

logger = logging.getLogger(__name__)


class TestMusicPlayerConstants:
    """Test MusicPlayer constants access"""

    def test_music_event_type_constants(self):
        """Test music event type constants"""
        assert capi.get_music_event_type_null() == 0
        assert capi.get_music_event_type_extended_note() == 1
        assert capi.get_music_event_type_extended_tempo() == 3
        assert capi.get_music_event_type_user() == 4
        assert capi.get_music_event_type_meta() == 5
        assert capi.get_music_event_type_midi_note_message() == 6
        assert capi.get_music_event_type_midi_channel_message() == 7
        assert capi.get_music_event_type_midi_raw_data() == 8
        assert capi.get_music_event_type_parameter() == 9
        assert capi.get_music_event_type_au_preset() == 10

    def test_music_sequence_type_constants(self):
        """Test music sequence type constants"""
        assert capi.get_music_sequence_type_beats() == 1650811252
        assert capi.get_music_sequence_type_seconds() == 1936024435
        assert capi.get_music_sequence_type_samples() == 1935764848

    def test_music_sequence_file_type_constants(self):
        """Test music sequence file type constants"""
        assert capi.get_music_sequence_file_any_type() == 0
        assert capi.get_music_sequence_file_midi_type() == 1835623529
        assert capi.get_music_sequence_file_imelody_type() == 1768777068

    def test_sequence_track_property_constants(self):
        """Test sequence track property constants"""
        assert capi.get_sequence_track_property_loop_info() == 0
        assert capi.get_sequence_track_property_offset_time() == 1
        assert capi.get_sequence_track_property_mute_status() == 2
        assert capi.get_sequence_track_property_solo_status() == 3
        assert capi.get_sequence_track_property_automated_parameters() == 4
        assert capi.get_sequence_track_property_track_length() == 5
        assert capi.get_sequence_track_property_time_resolution() == 6


class TestMusicPlayerHelpers:
    """Test MusicPlayer helper functions"""

    def test_create_midi_note_message(self):
        """Test creating MIDI note message"""
        msg = capi.create_midi_note_message(0, 60, 127, 64, 2.0)
        assert msg["channel"] == 0
        assert msg["note"] == 60
        assert msg["velocity"] == 127
        assert msg["release_velocity"] == 64
        assert msg["duration"] == 2.0
        msg = capi.create_midi_note_message(1, 72, 100)
        assert msg["channel"] == 1
        assert msg["note"] == 72
        assert msg["velocity"] == 100
        assert msg["release_velocity"] == 0
        assert msg["duration"] == 1.0

    def test_create_midi_channel_message(self):
        """Test creating MIDI channel message"""
        msg = capi.create_midi_channel_message(144, 60, 127)
        assert msg["status"] == 144
        assert msg["data1"] == 60
        assert msg["data2"] == 127
        msg = capi.create_midi_channel_message(192, 10)
        assert msg["status"] == 192
        assert msg["data1"] == 10
        assert msg["data2"] == 0

    def test_midi_message_bounds_checking(self):
        """Test that helper functions respect bounds"""
        msg = capi.create_midi_note_message(16, 60, 127)
        assert msg["channel"] == 0
        msg = capi.create_midi_note_message(0, 128, 127)
        assert msg["note"] == 0
        msg = capi.create_midi_note_message(0, 60, 255)
        assert msg["velocity"] == 127


class TestMusicPlayerBasicOperations:
    """Test basic MusicPlayer operations"""

    def test_music_player_creation_and_disposal(self):
        """Test creating and disposing a music player"""
        player = capi.new_music_player()
        assert isinstance(player, int)
        assert player != 0
        result = capi.dispose_music_player(player)
        assert result == 0

    def test_music_sequence_creation_and_disposal(self):
        """Test creating and disposing a music sequence"""
        sequence = capi.new_music_sequence()
        assert isinstance(sequence, int)
        assert sequence != 0
        result = capi.dispose_music_sequence(sequence)
        assert result == 0

    def test_music_player_sequence_assignment(self):
        """Test assigning sequence to player"""
        player = capi.new_music_player()
        sequence = capi.new_music_sequence()
        try:
            result = capi.music_player_set_sequence(player, sequence)
            assert result == 0
            retrieved_sequence = capi.music_player_get_sequence(player)
            assert retrieved_sequence == sequence
            result = capi.music_player_set_sequence(player, 0)
            assert result == 0
        finally:
            capi.dispose_music_sequence(sequence)
            capi.dispose_music_player(player)

    def test_music_player_time_operations(self):
        """Test player time get/set operations"""
        player = capi.new_music_player()
        sequence = capi.new_music_sequence()
        try:
            capi.music_player_set_sequence(player, sequence)
            result = capi.music_player_set_time(player, 10.5)
            assert result == 0
            time = capi.music_player_get_time(player)
            assert abs(time - 10.5) < 0.01
            result = capi.music_player_set_time(player, 0.0)
            assert result == 0
            time = capi.music_player_get_time(player)
            assert time == 0.0
        finally:
            try:
                capi.music_player_set_sequence(player, 0)
            except Exception as e:
                logger.warning(f"Cleanup failed (set_sequence): {e}")
            capi.dispose_music_sequence(sequence)
            capi.dispose_music_player(player)

    def test_music_player_playback_state(self):
        """Test player playback state operations"""
        player = capi.new_music_player()
        sequence = capi.new_music_sequence()
        try:
            is_playing = capi.music_player_is_playing(player)
            assert is_playing is False
            capi.music_player_set_sequence(player, sequence)
            result = capi.music_player_preroll(player)
            assert result == 0
            try:
                result = capi.music_player_start(player)
                if result == 0:
                    is_playing = capi.music_player_is_playing(player)
                    capi.music_player_stop(player)
                    is_playing = capi.music_player_is_playing(player)
                    assert is_playing is False
                else:
                    is_playing = capi.music_player_is_playing(player)
                    assert is_playing is False
            except RuntimeError as e:
                is_playing = capi.music_player_is_playing(player)
                assert is_playing is False
        finally:
            try:
                capi.music_player_stop(player)
                capi.music_player_set_sequence(player, 0)
            except Exception as e:
                logger.warning(f"Cleanup failed (stop/set_sequence): {e}")
            capi.dispose_music_sequence(sequence)
            capi.dispose_music_player(player)

    def test_music_player_rate_scalar(self):
        """Test player rate scalar operations"""
        player = capi.new_music_player()
        try:
            rate = capi.music_player_get_play_rate_scalar(player)
            assert rate == 1.0
            result = capi.music_player_set_play_rate_scalar(player, 2.0)
            assert result == 0
            rate = capi.music_player_get_play_rate_scalar(player)
            assert rate == 2.0
            result = capi.music_player_set_play_rate_scalar(player, 0.5)
            assert result == 0
            rate = capi.music_player_get_play_rate_scalar(player)
            assert rate == 0.5
        finally:
            capi.dispose_music_player(player)

    def test_music_player_invalid_rate_scalar(self):
        """Test player rate scalar validation"""
        player = capi.new_music_player()
        try:
            with pytest.raises(ValueError):
                capi.music_player_set_play_rate_scalar(player, -1.0)
            with pytest.raises(ValueError):
                capi.music_player_set_play_rate_scalar(player, 0.0)
        finally:
            capi.dispose_music_player(player)


class TestMusicSequenceOperations:
    """Test MusicSequence operations"""

    def test_music_sequence_track_operations(self):
        """Test sequence track creation and management"""
        sequence = capi.new_music_sequence()
        try:
            count = capi.music_sequence_get_track_count(sequence)
            assert count == 0
            track1 = capi.music_sequence_new_track(sequence)
            assert isinstance(track1, int)
            assert track1 != 0
            count = capi.music_sequence_get_track_count(sequence)
            assert count == 1
            retrieved_track = capi.music_sequence_get_ind_track(sequence, 0)
            assert retrieved_track == track1
            track2 = capi.music_sequence_new_track(sequence)
            assert isinstance(track2, int)
            assert track2 != 0
            assert track2 != track1
            count = capi.music_sequence_get_track_count(sequence)
            assert count == 2
            result = capi.music_sequence_dispose_track(sequence, track1)
            assert result == 0
            count = capi.music_sequence_get_track_count(sequence)
            assert count == 1
        finally:
            capi.dispose_music_sequence(sequence)

    def test_music_sequence_tempo_track(self):
        """Test getting the tempo track"""
        sequence = capi.new_music_sequence()
        try:
            tempo_track = capi.music_sequence_get_tempo_track(sequence)
            assert isinstance(tempo_track, int)
            assert tempo_track != 0
            count = capi.music_sequence_get_track_count(sequence)
            assert count == 0
        finally:
            capi.dispose_music_sequence(sequence)

    def test_music_sequence_type_operations(self):
        """Test sequence type get/set operations"""
        sequence = capi.new_music_sequence()
        try:
            seq_type = capi.music_sequence_get_sequence_type(sequence)
            assert seq_type == capi.get_music_sequence_type_beats()
            result = capi.music_sequence_set_sequence_type(
                sequence, capi.get_music_sequence_type_seconds()
            )
            assert result == 0
            seq_type = capi.music_sequence_get_sequence_type(sequence)
            assert seq_type == capi.get_music_sequence_type_seconds()
            result = capi.music_sequence_set_sequence_type(
                sequence, capi.get_music_sequence_type_beats()
            )
            assert result == 0
            seq_type = capi.music_sequence_get_sequence_type(sequence)
            assert seq_type == capi.get_music_sequence_type_beats()
        finally:
            capi.dispose_music_sequence(sequence)

    def test_music_sequence_file_load_invalid_path(self):
        """Test loading non-existent file"""
        sequence = capi.new_music_sequence()
        try:
            with pytest.raises(RuntimeError):
                capi.music_sequence_file_load(sequence, "/path/that/does/not/exist.mid")
        finally:
            capi.dispose_music_sequence(sequence)


class TestMusicTrackOperations:
    """Test MusicTrack operations"""

    def test_music_track_midi_note_events(self):
        """Test adding MIDI note events to tracks"""
        sequence = capi.new_music_sequence()
        track = capi.music_sequence_new_track(sequence)
        try:
            result = capi.music_track_new_midi_note_event(
                track, 0.0, 0, 60, 127, 64, 1.0
            )
            assert result == 0
            result = capi.music_track_new_midi_note_event(
                track, 1.0, 1, 67, 100, 50, 0.5
            )
            assert result == 0
        finally:
            capi.dispose_music_sequence(sequence)

    def test_music_track_midi_channel_events(self):
        """Test adding MIDI channel events to tracks"""
        sequence = capi.new_music_sequence()
        track = capi.music_sequence_new_track(sequence)
        try:
            result = capi.music_track_new_midi_channel_event(track, 0.0, 192, 10, 0)
            assert result == 0
            result = capi.music_track_new_midi_channel_event(track, 0.5, 176, 7, 100)
            assert result == 0
        finally:
            capi.dispose_music_sequence(sequence)

    def test_music_track_tempo_events(self):
        """Test adding tempo events to tracks"""
        sequence = capi.new_music_sequence()
        tempo_track = capi.music_sequence_get_tempo_track(sequence)
        try:
            result = capi.music_track_new_extended_tempo_event(tempo_track, 0.0, 120.0)
            assert result == 0
            result = capi.music_track_new_extended_tempo_event(tempo_track, 4.0, 140.0)
            assert result == 0
        finally:
            capi.dispose_music_sequence(sequence)

    def test_music_track_invalid_bpm(self):
        """Test tempo event validation"""
        sequence = capi.new_music_sequence()
        tempo_track = capi.music_sequence_get_tempo_track(sequence)
        try:
            with pytest.raises(ValueError):
                capi.music_track_new_extended_tempo_event(tempo_track, 0.0, -120.0)
            with pytest.raises(ValueError):
                capi.music_track_new_extended_tempo_event(tempo_track, 0.0, 0.0)
        finally:
            capi.dispose_music_sequence(sequence)


class TestMusicPlayerErrorHandling:
    """Test MusicPlayer error handling"""

    def test_invalid_player_operations(self):
        """Test operations with invalid player handles"""
        invalid_player = 0
        with pytest.raises(RuntimeError):
            capi.music_player_set_sequence(invalid_player, 0)
        with pytest.raises(RuntimeError):
            capi.music_player_get_sequence(invalid_player)
        with pytest.raises(RuntimeError):
            capi.music_player_set_time(invalid_player, 0.0)
        with pytest.raises(RuntimeError):
            capi.music_player_get_time(invalid_player)
        with pytest.raises(RuntimeError):
            capi.music_player_preroll(invalid_player)
        with pytest.raises(RuntimeError):
            capi.music_player_start(invalid_player)
        with pytest.raises(RuntimeError):
            capi.music_player_stop(invalid_player)
        with pytest.raises(RuntimeError):
            capi.music_player_is_playing(invalid_player)
        with pytest.raises(RuntimeError):
            capi.music_player_set_play_rate_scalar(invalid_player, 1.0)
        with pytest.raises(RuntimeError):
            capi.music_player_get_play_rate_scalar(invalid_player)

    def test_invalid_sequence_operations(self):
        """Test operations with invalid sequence handles"""
        invalid_sequence = 0
        with pytest.raises(RuntimeError):
            capi.music_sequence_new_track(invalid_sequence)
        with pytest.raises(RuntimeError):
            capi.music_sequence_get_track_count(invalid_sequence)
        with pytest.raises(RuntimeError):
            capi.music_sequence_get_ind_track(invalid_sequence, 0)
        with pytest.raises(RuntimeError):
            capi.music_sequence_get_tempo_track(invalid_sequence)
        with pytest.raises(RuntimeError):
            capi.music_sequence_set_sequence_type(
                invalid_sequence, capi.get_music_sequence_type_beats()
            )
        with pytest.raises(RuntimeError):
            capi.music_sequence_get_sequence_type(invalid_sequence)

    def test_invalid_track_operations(self):
        """Test operations with invalid track handles"""
        invalid_track = 0
        with pytest.raises(RuntimeError):
            capi.music_track_new_midi_note_event(
                invalid_track, 0.0, 0, 60, 127, 64, 1.0
            )
        with pytest.raises(RuntimeError):
            capi.music_track_new_midi_channel_event(invalid_track, 0.0, 192, 10, 0)
        with pytest.raises(RuntimeError):
            capi.music_track_new_extended_tempo_event(invalid_track, 0.0, 120.0)

    def test_sequence_track_index_errors(self):
        """Test sequence track index error handling"""
        sequence = capi.new_music_sequence()
        try:
            with pytest.raises(RuntimeError):
                capi.music_sequence_get_ind_track(sequence, 0)
            track = capi.music_sequence_new_track(sequence)
            retrieved_track = capi.music_sequence_get_ind_track(sequence, 0)
            assert retrieved_track == track
            with pytest.raises(RuntimeError):
                capi.music_sequence_get_ind_track(sequence, 1)
        finally:
            capi.dispose_music_sequence(sequence)


class TestMusicPlayerIntegration:
    """Test MusicPlayer integration scenarios"""

    def test_complete_sequence_playback_setup(self):
        """Test setting up a complete sequence for playback"""
        player = capi.new_music_player()
        sequence = capi.new_music_sequence()
        try:
            track = capi.music_sequence_new_track(sequence)
            tempo_track = capi.music_sequence_get_tempo_track(sequence)
            capi.music_track_new_extended_tempo_event(tempo_track, 0.0, 120.0)
            capi.music_track_new_midi_note_event(track, 0.0, 0, 60, 127, 64, 1.0)
            capi.music_track_new_midi_note_event(track, 1.0, 0, 64, 127, 64, 1.0)
            capi.music_track_new_midi_note_event(track, 2.0, 0, 67, 127, 64, 1.0)
            capi.music_track_new_midi_note_event(track, 3.0, 0, 72, 127, 64, 1.0)
            capi.music_player_set_sequence(player, sequence)
            retrieved_sequence = capi.music_player_get_sequence(player)
            assert retrieved_sequence == sequence
            capi.music_player_preroll(player)
            try:
                capi.music_player_start(player)
                started = True
            except RuntimeError:
                started = False
            if started:
                assert capi.music_player_is_playing(player) is True
            capi.music_player_set_play_rate_scalar(player, 0.8)
            rate = capi.music_player_get_play_rate_scalar(player)
            assert rate == 0.8
            capi.music_player_set_time(player, 2.5)
            time = capi.music_player_get_time(player)
            assert abs(time - 2.5) < 0.01
            if started:
                capi.music_player_stop(player)
                assert capi.music_player_is_playing(player) is False
        finally:
            try:
                capi.music_player_stop(player)
                capi.music_player_set_sequence(player, 0)
            except Exception as e:
                logger.warning(f"Cleanup failed (stop/set_sequence): {e}")
            capi.dispose_music_sequence(sequence)
            capi.dispose_music_player(player)

    def test_multiple_sequences_on_player(self):
        """Test switching sequences on a player"""
        player = capi.new_music_player()
        sequence1 = capi.new_music_sequence()
        sequence2 = capi.new_music_sequence()
        try:
            capi.music_player_set_sequence(player, sequence1)
            retrieved = capi.music_player_get_sequence(player)
            assert retrieved == sequence1
            capi.music_player_set_sequence(player, sequence2)
            retrieved = capi.music_player_get_sequence(player)
            assert retrieved == sequence2
            capi.music_player_set_sequence(player, 0)
            with pytest.raises(RuntimeError):
                capi.music_player_get_sequence(player)
        finally:
            capi.dispose_music_sequence(sequence1)
            capi.dispose_music_sequence(sequence2)
            capi.dispose_music_player(player)


class TestMusicPlayerResourceManagement:
    """Test MusicPlayer resource management"""

    def test_multiple_players(self):
        """Test creating multiple players"""
        players = []
        sequences = []
        try:
            for i in range(5):
                player = capi.new_music_player()
                sequence = capi.new_music_sequence()
                players.append(player)
                sequences.append(sequence)
                assert isinstance(player, int)
                assert player != 0
                capi.music_player_set_sequence(player, sequence)
            assert len(set(players)) == len(players)
            for i, player in enumerate(players):
                capi.music_player_set_time(player, float(i))
                time = capi.music_player_get_time(player)
                assert time == float(i)
        finally:
            for i, (player, sequence) in enumerate(zip(players, sequences)):
                try:
                    capi.music_player_set_sequence(player, 0)
                    capi.dispose_music_sequence(sequence)
                    capi.dispose_music_player(player)
                except Exception as e:
                    logger.warning(f"Cleanup failed for player/sequence {i}: {e}")

    def test_multiple_sequences(self):
        """Test creating multiple sequences"""
        sequences = []
        try:
            for i in range(3):
                sequence = capi.new_music_sequence()
                sequences.append(sequence)
                assert isinstance(sequence, int)
                assert sequence != 0
            assert len(set(sequences)) == len(sequences)
            for i, sequence in enumerate(sequences):
                for j in range(i + 1):
                    track = capi.music_sequence_new_track(sequence)
                    assert isinstance(track, int)
                count = capi.music_sequence_get_track_count(sequence)
                assert count == i + 1
        finally:
            for i, sequence in enumerate(sequences):
                try:
                    capi.dispose_music_sequence(sequence)
                except Exception as e:
                    logger.warning(f"Cleanup failed for sequence {i}: {e}")

    def test_sequence_lifecycle(self):
        """Test complete sequence lifecycle"""
        sequence = capi.new_music_sequence()
        player = capi.new_music_player()
        try:
            track = capi.music_sequence_new_track(sequence)
            tempo_track = capi.music_sequence_get_tempo_track(sequence)
            capi.music_track_new_extended_tempo_event(tempo_track, 0.0, 120.0)
            capi.music_track_new_midi_note_event(track, 0.0, 0, 60, 127, 64, 4.0)
            capi.music_player_set_sequence(player, sequence)
            capi.music_player_preroll(player)
            capi.music_player_start(player)
            assert capi.music_player_is_playing(player) is True
            capi.music_player_stop(player)
            capi.music_player_set_sequence(player, 0)
        finally:
            capi.dispose_music_sequence(sequence)
            capi.dispose_music_player(player)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
