"""Tests for Link high-level Python API (Phase 3)

Tests context manager support, module exports, and Pythonic API patterns.
"""

import pytest
import time
import coremusic as cm


class TestLinkContextManager:
    """Test LinkSession context manager protocol"""

    def test_context_manager_enables_and_disables(self):
        """Test context manager automatically enables/disables Link"""
        session = cm.link.LinkSession(bpm=120.0)

        # Should start disabled
        assert session.enabled == False

        # Enter context - should enable
        with session:
            assert session.enabled == True

        # Exit context - should disable
        assert session.enabled == False

    def test_context_manager_returns_session(self):
        """Test context manager returns the session instance"""
        session = cm.link.LinkSession(bpm=120.0)

        with session as link:
            assert link is session
            assert isinstance(link, cm.link.LinkSession)

    def test_context_manager_with_operations(self):
        """Test performing operations within context manager"""
        with cm.link.LinkSession(bpm=120.0) as session:
            # Should be enabled
            assert session.enabled == True

            # Should be able to capture state
            state = session.capture_app_session_state()
            assert state.tempo == pytest.approx(120.0, abs=0.1)

            # Should be able to modify state
            current_time = session.clock.micros()
            state.set_tempo(140.0, current_time)
            session.commit_app_session_state(state)

            time.sleep(0.1)

            # Verify tempo changed
            state2 = session.capture_app_session_state()
            assert state2.tempo == pytest.approx(140.0, abs=0.1)

        # After exit, should be disabled
        assert session.enabled == False

    def test_context_manager_exception_still_disables(self):
        """Test context manager disables Link even on exception"""
        session = cm.link.LinkSession(bpm=120.0)

        try:
            with session:
                assert session.enabled == True
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still be disabled after exception
        assert session.enabled == False

    def test_nested_context_managers(self):
        """Test using multiple Link sessions with context managers"""
        session1 = cm.link.LinkSession(bpm=120.0)
        session2 = cm.link.LinkSession(bpm=130.0)

        with session1:
            assert session1.enabled == True
            assert session2.enabled == False

            with session2:
                assert session1.enabled == True
                assert session2.enabled == True

            assert session1.enabled == True
            assert session2.enabled == False

        assert session1.enabled == False
        assert session2.enabled == False


class TestLinkModuleExports:
    """Test Link module is properly exported from main package"""

    def test_link_module_accessible(self):
        """Test link module is accessible from coremusic"""
        assert hasattr(cm, 'link')
        assert cm.link is not None

    def test_link_classes_accessible(self):
        """Test Link classes are accessible from link module"""
        assert hasattr(cm.link, 'LinkSession')
        assert hasattr(cm.link, 'SessionState')
        assert hasattr(cm.link, 'Clock')

    def test_can_create_instances_from_main_module(self):
        """Test creating Link instances via main module"""
        session = cm.link.LinkSession(bpm=120.0)
        assert isinstance(session, cm.link.LinkSession)

        clock = cm.link.Clock()
        assert isinstance(clock, cm.link.Clock)

    def test_link_in_all_exports(self):
        """Test link is in __all__ exports"""
        assert 'link' in cm.__all__


class TestLinkPythonicPatterns:
    """Test Pythonic API patterns and conventions"""

    def test_properties_are_readable(self):
        """Test Link properties are readable"""
        session = cm.link.LinkSession(bpm=120.0)

        # All properties should be readable
        _ = session.enabled
        _ = session.num_peers
        _ = session.start_stop_sync_enabled
        _ = session.clock

    def test_properties_are_writable(self):
        """Test Link properties are writable where appropriate"""
        session = cm.link.LinkSession(bpm=120.0)

        # enabled property should be writable
        session.enabled = True
        assert session.enabled == True

        session.enabled = False
        assert session.enabled == False

        # start_stop_sync_enabled should be writable
        session.start_stop_sync_enabled = True
        assert session.start_stop_sync_enabled == True

        session.start_stop_sync_enabled = False
        assert session.start_stop_sync_enabled == False

    def test_repr_is_informative(self):
        """Test __repr__ provides useful information"""
        session = cm.link.LinkSession(bpm=120.0)

        repr_str = repr(session)
        assert 'LinkSession' in repr_str
        assert 'disabled' in repr_str
        assert 'peers' in repr_str

        session.enabled = True
        repr_str = repr(session)
        assert 'enabled' in repr_str

    def test_session_state_properties(self):
        """Test SessionState has readable properties"""
        session = cm.link.LinkSession(bpm=120.0)
        session.enabled = True

        state = session.capture_app_session_state()

        # Should have readable properties
        _ = state.tempo
        _ = state.is_playing

    def test_clock_methods_return_int(self):
        """Test Clock methods return proper integer types"""
        clock = cm.link.Clock()

        micros = clock.micros()
        assert isinstance(micros, int)
        assert micros > 0

        ticks = clock.ticks()
        assert isinstance(ticks, int)
        assert ticks > 0

    def test_method_signatures_accept_named_args(self):
        """Test methods accept named arguments (Pythonic)"""
        session = cm.link.LinkSession(bpm=120.0)
        session.enabled = True

        state = session.capture_app_session_state()
        current_time = session.clock.micros()

        # Should accept named arguments
        state.set_tempo(bpm=140.0, time_micros=current_time)
        session.commit_app_session_state(state=state)

        # Should accept positional arguments
        state.set_tempo(130.0, current_time)
        session.commit_app_session_state(state)


class TestLinkUsagePatterns:
    """Test common usage patterns work as expected"""

    def test_simple_tempo_monitoring(self):
        """Test simple tempo monitoring pattern"""
        with cm.link.LinkSession(bpm=120.0) as session:
            for _ in range(5):
                state = session.capture_app_session_state()
                tempo = state.tempo
                assert isinstance(tempo, float)
                assert tempo > 0
                time.sleep(0.01)

    def test_beat_tracking_pattern(self):
        """Test beat tracking pattern"""
        with cm.link.LinkSession(bpm=120.0) as session:
            clock = session.clock

            for _ in range(5):
                state = session.capture_app_session_state()
                current_time = clock.micros()
                beat = state.beat_at_time(current_time, quantum=4.0)

                assert isinstance(beat, float)
                assert beat >= 0.0

                time.sleep(0.01)

    def test_transport_control_pattern(self):
        """Test transport control pattern"""
        with cm.link.LinkSession(bpm=120.0) as session:
            session.start_stop_sync_enabled = True

            # Start transport
            state = session.capture_app_session_state()
            current_time = session.clock.micros()
            state.set_is_playing(True, current_time)
            session.commit_app_session_state(state)

            time.sleep(0.1)

            # Verify playing
            state = session.capture_app_session_state()
            assert state.is_playing == True

            # Stop transport
            current_time = session.clock.micros()
            state.set_is_playing(False, current_time)
            session.commit_app_session_state(state)

            time.sleep(0.1)

            # Verify stopped
            state = session.capture_app_session_state()
            assert state.is_playing == False

    def test_tempo_change_pattern(self):
        """Test tempo change pattern"""
        with cm.link.LinkSession(bpm=120.0) as session:
            # Change tempo
            state = session.capture_app_session_state()
            current_time = session.clock.micros()
            state.set_tempo(150.0, current_time)
            session.commit_app_session_state(state)

            time.sleep(0.1)

            # Verify tempo changed
            state = session.capture_app_session_state()
            assert state.tempo == pytest.approx(150.0, abs=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
