"""Tests for Link integration with AudioPlayer

Tests the integration between Ableton Link and AudioPlayer for tempo-synchronized
audio playback.
"""

import pytest
import time

from coremusic import link
from coremusic import AudioPlayer


class TestLinkAudioPlayerIntegration:
    """Test Link integration with AudioPlayer"""

    def test_audio_player_creation_with_link(self):
        """Test creating an AudioPlayer with Link session"""
        session = link.LinkSession(bpm=120.0)
        player = AudioPlayer(link_session=session)

        assert player is not None
        assert player.link_session is session

    def test_audio_player_creation_without_link(self):
        """Test creating an AudioPlayer without Link session"""
        player = AudioPlayer()

        assert player is not None
        assert player.link_session is None

    def test_get_link_timing_without_session(self):
        """Test get_link_timing returns None when no Link session"""
        player = AudioPlayer()
        timing = player.get_link_timing()

        assert timing is None

    def test_get_link_timing_with_session(self):
        """Test get_link_timing returns timing info when Link session attached"""
        session = link.LinkSession(bpm=120.0)
        session.enabled = True

        player = AudioPlayer(link_session=session)
        timing = player.get_link_timing(quantum=4.0)

        assert timing is not None
        assert 'tempo' in timing
        assert 'beat' in timing
        assert 'phase' in timing
        assert 'is_playing' in timing

        # Check values are reasonable
        assert timing['tempo'] == pytest.approx(120.0, abs=0.1)
        assert timing['beat'] >= 0.0
        assert 0.0 <= timing['phase'] < 4.0
        assert timing['is_playing'] == False

    def test_link_timing_updates(self):
        """Test that Link timing values update over time"""
        session = link.LinkSession(bpm=120.0)
        session.enabled = True

        player = AudioPlayer(link_session=session)

        # Get initial timing
        timing1 = player.get_link_timing(quantum=4.0)
        beat1 = timing1['beat']

        # Wait a bit
        time.sleep(0.1)

        # Get updated timing
        timing2 = player.get_link_timing(quantum=4.0)
        beat2 = timing2['beat']

        # Beat should have advanced
        # At 120 BPM, we have 2 beats per second
        # In 0.1s, beat advances by ~0.2
        assert beat2 > beat1
        assert abs(beat2 - beat1 - 0.2) < 0.05

    def test_link_tempo_changes_visible_in_player(self):
        """Test that tempo changes in Link are visible through AudioPlayer"""
        session = link.LinkSession(bpm=120.0)
        session.enabled = True

        player = AudioPlayer(link_session=session)

        # Get initial timing
        timing1 = player.get_link_timing()
        assert timing1['tempo'] == pytest.approx(120.0, abs=0.1)

        # Change tempo in Link session
        state = session.capture_app_session_state()
        current_time = session.clock.micros()
        state.set_tempo(140.0, current_time)
        session.commit_app_session_state(state)

        time.sleep(0.1)

        # Get updated timing through player
        timing2 = player.get_link_timing()
        assert timing2['tempo'] == pytest.approx(140.0, abs=0.1)

    def test_link_transport_state_visible_in_player(self):
        """Test that transport state changes are visible through AudioPlayer"""
        session = link.LinkSession(bpm=120.0)
        session.enabled = True
        session.start_stop_sync_enabled = True

        player = AudioPlayer(link_session=session)

        # Check initial state
        timing1 = player.get_link_timing()
        assert timing1['is_playing'] == False

        # Start transport
        state = session.capture_app_session_state()
        current_time = session.clock.micros()
        state.set_is_playing(True, current_time)
        session.commit_app_session_state(state)

        time.sleep(0.1)

        # Check transport is playing
        timing2 = player.get_link_timing()
        assert timing2['is_playing'] == True

        # Stop transport
        state = session.capture_app_session_state()
        current_time = session.clock.micros()
        state.set_is_playing(False, current_time)
        session.commit_app_session_state(state)

        time.sleep(0.1)

        # Check transport stopped
        timing3 = player.get_link_timing()
        assert timing3['is_playing'] == False

    def test_multiple_players_sharing_link_session(self):
        """Test multiple AudioPlayers can share the same Link session"""
        session = link.LinkSession(bpm=120.0)
        session.enabled = True

        player1 = AudioPlayer(link_session=session)
        player2 = AudioPlayer(link_session=session)

        # Both should see the same Link session
        assert player1.link_session is player2.link_session

        # Get timing from both
        timing1 = player1.get_link_timing(quantum=4.0)
        timing2 = player2.get_link_timing(quantum=4.0)

        # Should have very similar timing (within a few microseconds)
        assert timing1['tempo'] == pytest.approx(timing2['tempo'], abs=0.01)
        assert timing1['beat'] == pytest.approx(timing2['beat'], abs=0.01)

    def test_link_session_reference_kept_alive(self):
        """Test that AudioPlayer keeps Link session alive"""
        def create_player():
            # Create Link session in local scope
            session = link.LinkSession(bpm=120.0)
            session.enabled = True
            return AudioPlayer(link_session=session)

        player = create_player()
        # Link session should still be alive through player reference

        timing = player.get_link_timing()
        assert timing is not None
        assert timing['tempo'] == pytest.approx(120.0, abs=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
