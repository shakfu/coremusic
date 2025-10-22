"""Tests for Ableton Link integration

Tests the basic functionality of the Link wrapper including session management,
clock operations, and session state manipulation.
"""

import pytest
import time

# Import Link module directly (not through main coremusic package to avoid circular import)
from coremusic import link


class TestClock:
    """Test Link Clock operations"""

    def test_clock_creation(self):
        """Test creating a Clock instance"""
        clock = link.Clock()
        assert clock is not None

    def test_clock_micros(self):
        """Test getting current time in microseconds"""
        clock = link.Clock()
        t1 = clock.micros()
        assert t1 > 0
        assert isinstance(t1, int)

        # Time should advance
        time.sleep(0.01)
        t2 = clock.micros()
        assert t2 > t1

    def test_clock_ticks(self):
        """Test getting current time in system ticks"""
        clock = link.Clock()
        t1 = clock.ticks()
        assert t1 > 0
        assert isinstance(t1, int)

        # Ticks should advance
        time.sleep(0.01)
        t2 = clock.ticks()
        assert t2 > t1

    def test_ticks_to_micros_conversion(self):
        """Test converting ticks to microseconds"""
        clock = link.Clock()
        ticks = clock.ticks()
        micros = clock.ticks_to_micros(ticks)

        assert isinstance(micros, int)
        assert micros > 0

    def test_micros_to_ticks_conversion(self):
        """Test converting microseconds to ticks"""
        clock = link.Clock()
        micros = clock.micros()
        ticks = clock.micros_to_ticks(micros)

        assert isinstance(ticks, int)
        assert ticks > 0

    def test_round_trip_conversion(self):
        """Test round-trip time conversion"""
        clock = link.Clock()
        micros1 = clock.micros()
        ticks = clock.micros_to_ticks(micros1)
        micros2 = clock.ticks_to_micros(ticks)

        # Should be approximately equal (within a small margin)
        assert abs(micros2 - micros1) < 1000  # Within 1ms


class TestLinkSession:
    """Test Link session management"""

    def test_link_creation(self):
        """Test creating a Link session"""
        session = link.LinkSession(bpm=120.0)
        assert session is not None

    def test_link_creation_default_bpm(self):
        """Test creating a Link session with default BPM"""
        session = link.LinkSession()
        assert session is not None

    def test_link_enabled_property(self):
        """Test Link enabled property"""
        session = link.LinkSession(bpm=120.0)

        # Should start disabled
        assert session.enabled == False

        # Enable
        session.enabled = True
        assert session.enabled == True

        # Disable
        session.enabled = False
        assert session.enabled == False

    def test_link_num_peers_initially_zero(self):
        """Test that num_peers is initially zero"""
        session = link.LinkSession(bpm=120.0)
        assert session.num_peers == 0

    def test_link_start_stop_sync_property(self):
        """Test start/stop sync property"""
        session = link.LinkSession(bpm=120.0)

        # Should start disabled
        assert session.start_stop_sync_enabled == False

        # Enable
        session.start_stop_sync_enabled = True
        assert session.start_stop_sync_enabled == True

        # Disable
        session.start_stop_sync_enabled = False
        assert session.start_stop_sync_enabled == False

    def test_link_clock_access(self):
        """Test accessing Link's clock"""
        session = link.LinkSession(bpm=120.0)
        clock = session.clock

        assert clock is not None
        assert isinstance(clock, link.Clock)

    def test_link_repr(self):
        """Test Link string representation"""
        session = link.LinkSession(bpm=120.0)
        repr_str = repr(session)

        assert "LinkSession" in repr_str
        assert "disabled" in repr_str
        assert "0 peers" in repr_str

        # Enable and check again
        session.enabled = True
        repr_str = repr(session)
        assert "enabled" in repr_str


class TestSessionState:
    """Test Link SessionState operations"""

    def test_capture_app_session_state(self):
        """Test capturing session state from app thread"""
        session = link.LinkSession(bpm=120.0)
        state = session.capture_app_session_state()

        assert state is not None
        assert isinstance(state, link.SessionState)

    def test_session_state_tempo(self):
        """Test reading tempo from session state"""
        session = link.LinkSession(bpm=120.0)
        state = session.capture_app_session_state()

        tempo = state.tempo
        assert isinstance(tempo, float)
        assert tempo == pytest.approx(120.0, abs=0.1)

    def test_session_state_set_tempo(self):
        """Test setting tempo in session state"""
        session = link.LinkSession(bpm=120.0)
        session.enabled = True

        state = session.capture_app_session_state()
        current_time = session.clock.micros()

        # Set new tempo
        state.set_tempo(140.0, current_time)
        session.commit_app_session_state(state)

        # Verify tempo changed
        time.sleep(0.1)  # Give it time to propagate
        state2 = session.capture_app_session_state()
        assert state2.tempo == pytest.approx(140.0, abs=0.1)

    def test_session_state_beat_at_time(self):
        """Test getting beat at time"""
        session = link.LinkSession(bpm=120.0)
        session.enabled = True

        state = session.capture_app_session_state()
        current_time = session.clock.micros()

        beat = state.beat_at_time(current_time, quantum=4.0)
        assert isinstance(beat, float)
        assert beat >= 0.0

    def test_session_state_phase_at_time(self):
        """Test getting phase at time"""
        session = link.LinkSession(bpm=120.0)
        session.enabled = True

        state = session.capture_app_session_state()
        current_time = session.clock.micros()

        phase = state.phase_at_time(current_time, quantum=4.0)
        assert isinstance(phase, float)
        assert 0.0 <= phase < 4.0

    def test_session_state_time_at_beat(self):
        """Test getting time at beat"""
        session = link.LinkSession(bpm=120.0)
        session.enabled = True

        state = session.capture_app_session_state()

        time_micros = state.time_at_beat(10.0, quantum=4.0)
        assert isinstance(time_micros, int)
        assert time_micros > 0

    def test_session_state_is_playing(self):
        """Test transport playing state"""
        session = link.LinkSession(bpm=120.0)
        session.enabled = True

        state = session.capture_app_session_state()

        # Should start as not playing
        assert state.is_playing == False

    def test_session_state_set_is_playing(self):
        """Test setting transport playing state"""
        session = link.LinkSession(bpm=120.0)
        session.enabled = True

        state = session.capture_app_session_state()
        current_time = session.clock.micros()

        # Start playing
        state.set_is_playing(True, current_time)
        session.commit_app_session_state(state)

        # Verify
        time.sleep(0.1)
        state2 = session.capture_app_session_state()
        assert state2.is_playing == True

        # Stop playing
        current_time2 = session.clock.micros()
        state2.set_is_playing(False, current_time2)
        session.commit_app_session_state(state2)

        # Verify
        time.sleep(0.1)
        state3 = session.capture_app_session_state()
        assert state3.is_playing == False

    def test_session_state_time_for_is_playing(self):
        """Test getting time when transport state changed"""
        session = link.LinkSession(bpm=120.0)
        session.enabled = True

        state = session.capture_app_session_state()
        current_time = session.clock.micros()

        state.set_is_playing(True, current_time)
        session.commit_app_session_state(state)

        time.sleep(0.1)
        state2 = session.capture_app_session_state()
        change_time = state2.time_for_is_playing()

        assert isinstance(change_time, int)
        assert change_time > 0

    def test_session_state_request_beat_at_time(self):
        """Test requesting beat at time"""
        session = link.LinkSession(bpm=120.0)
        session.enabled = True

        state = session.capture_app_session_state()
        current_time = session.clock.micros()

        # Request beat 0 at current time
        state.request_beat_at_time(0.0, current_time, quantum=4.0)
        session.commit_app_session_state(state)

        # Verify beat is near 0
        time.sleep(0.01)
        state2 = session.capture_app_session_state()
        current_time2 = session.clock.micros()
        beat = state2.beat_at_time(current_time2, quantum=4.0)

        # Beat should be close to 0 (allowing for some time passage)
        assert beat >= 0.0 and beat < 1.0


class TestTwoLinkSessions:
    """Test interaction between two Link sessions"""

    def test_two_sessions_discover_each_other(self):
        """Test that two Link sessions can discover each other"""
        session1 = link.LinkSession(bpm=120.0)
        session2 = link.LinkSession(bpm=120.0)

        session1.enabled = True
        session2.enabled = True

        # Give them time to discover each other
        time.sleep(1.0)

        # Should see each other
        assert session1.num_peers >= 1
        assert session2.num_peers >= 1

        # Cleanup
        session1.enabled = False
        session2.enabled = False

    def test_two_sessions_sync_tempo(self):
        """Test that two sessions synchronize tempo"""
        session1 = link.LinkSession(bpm=120.0)
        session2 = link.LinkSession(bpm=120.0)

        session1.enabled = True
        session2.enabled = True

        # Wait for discovery
        time.sleep(1.0)

        # Change tempo on session1
        state1 = session1.capture_app_session_state()
        current_time = session1.clock.micros()
        state1.set_tempo(150.0, current_time)
        session1.commit_app_session_state(state1)

        # Wait for sync
        time.sleep(0.5)

        # session2 should see the new tempo
        state2 = session2.capture_app_session_state()
        assert state2.tempo == pytest.approx(150.0, abs=1.0)

        # Cleanup
        session1.enabled = False
        session2.enabled = False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
