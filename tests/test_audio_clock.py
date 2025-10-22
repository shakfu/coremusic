"""Tests for CoreAudioClock functionality"""

import pytest
import time
import coremusic as cm
import coremusic.capi as capi


class TestClockTimeFormat:
    """Test ClockTimeFormat constants"""

    def test_time_format_constants_exist(self):
        """Test that all time format constants are defined"""
        assert hasattr(cm.ClockTimeFormat, "HOST_TIME")
        assert hasattr(cm.ClockTimeFormat, "SAMPLES")
        assert hasattr(cm.ClockTimeFormat, "BEATS")
        assert hasattr(cm.ClockTimeFormat, "SECONDS")
        assert hasattr(cm.ClockTimeFormat, "SMPTE_TIME")

    def test_time_format_values(self):
        """Test that time format constants are valid integers"""
        assert isinstance(cm.ClockTimeFormat.HOST_TIME, int)
        assert isinstance(cm.ClockTimeFormat.BEATS, int)
        assert isinstance(cm.ClockTimeFormat.SECONDS, int)


class TestAudioClockLowLevel:
    """Test low-level clock API (capi module)"""

    def test_clock_create_dispose(self):
        """Test clock creation and disposal"""
        clock_id = capi.ca_clock_new()
        assert isinstance(clock_id, int)
        assert clock_id > 0
        capi.ca_clock_dispose(clock_id)

    def test_clock_start_stop(self):
        """Test starting and stopping clock"""
        clock_id = capi.ca_clock_new()
        try:
            capi.ca_clock_start(clock_id)
            capi.ca_clock_stop(clock_id)
        finally:
            capi.ca_clock_dispose(clock_id)

    def test_clock_play_rate(self):
        """Test getting and setting play rate"""
        clock_id = capi.ca_clock_new()
        try:
            # Default rate should be 1.0
            rate = capi.ca_clock_get_play_rate(clock_id)
            assert isinstance(rate, float)

            # Set different rates
            capi.ca_clock_set_play_rate(clock_id, 0.5)
            assert abs(capi.ca_clock_get_play_rate(clock_id) - 0.5) < 0.01

            capi.ca_clock_set_play_rate(clock_id, 2.0)
            assert abs(capi.ca_clock_get_play_rate(clock_id) - 2.0) < 0.01
        finally:
            capi.ca_clock_dispose(clock_id)

    def test_clock_get_time_seconds(self):
        """Test getting current time in seconds"""
        clock_id = capi.ca_clock_new()
        try:
            fmt = capi.get_ca_clock_time_format_seconds()
            time_info = capi.ca_clock_get_current_time(clock_id, fmt)

            assert isinstance(time_info, dict)
            assert "format" in time_info
            assert "value" in time_info
            assert isinstance(time_info["value"], float)
        finally:
            capi.ca_clock_dispose(clock_id)

    def test_clock_get_time_beats(self):
        """Test getting current time in beats"""
        clock_id = capi.ca_clock_new()
        try:
            fmt = capi.get_ca_clock_time_format_beats()
            time_info = capi.ca_clock_get_current_time(clock_id, fmt)

            assert isinstance(time_info, dict)
            assert "value" in time_info
            assert isinstance(time_info["value"], float)
        finally:
            capi.ca_clock_dispose(clock_id)


class TestAudioClockHighLevel:
    """Test high-level AudioClock class"""

    def test_clock_creation(self):
        """Test AudioClock creation"""
        clock = cm.AudioClock()
        clock.create()
        assert not clock.is_disposed
        clock.dispose()
        assert clock.is_disposed

    def test_clock_context_manager(self):
        """Test AudioClock as context manager"""
        with cm.AudioClock() as clock:
            assert not clock.is_disposed
        # Should be disposed after exiting context

    def test_clock_start_stop(self):
        """Test starting and stopping clock"""
        with cm.AudioClock() as clock:
            assert not clock.is_running
            clock.start()
            assert clock.is_running
            clock.stop()
            assert not clock.is_running

    def test_clock_play_rate_property(self):
        """Test play rate property"""
        with cm.AudioClock() as clock:
            # Default should be around 1.0
            initial_rate = clock.play_rate
            assert isinstance(initial_rate, float)

            # Set new rate
            clock.play_rate = 0.5
            assert abs(clock.play_rate - 0.5) < 0.01

            clock.play_rate = 2.0
            assert abs(clock.play_rate - 2.0) < 0.01

    def test_get_time_seconds(self):
        """Test getting time in seconds"""
        with cm.AudioClock() as clock:
            seconds = clock.get_time_seconds()
            assert isinstance(seconds, float)
            assert seconds >= 0.0

    def test_get_time_beats(self):
        """Test getting time in beats"""
        with cm.AudioClock() as clock:
            beats = clock.get_time_beats()
            assert isinstance(beats, float)
            assert beats >= 0.0

    def test_get_time_samples(self):
        """Test getting time in samples"""
        with cm.AudioClock() as clock:
            samples = clock.get_time_samples()
            assert isinstance(samples, float)

    def test_get_time_host(self):
        """Test getting host time"""
        with cm.AudioClock() as clock:
            host_time = clock.get_time_host()
            assert isinstance(host_time, int)

    def test_clock_advances_at_normal_speed(self):
        """Test that clock advances at approximately 1:1 with real time"""
        with cm.AudioClock() as clock:
            clock.play_rate = 1.0
            clock.start()

            time1 = clock.get_time_seconds()
            time.sleep(0.1)  # Wait 100ms
            time2 = clock.get_time_seconds()

            clock.stop()

            # Clock should have advanced approximately 0.1 seconds
            delta = time2 - time1
            assert 0.08 < delta < 0.12, f"Expected ~0.1s, got {delta:.4f}s"

    def test_clock_advances_at_half_speed(self):
        """Test that clock advances at half speed"""
        with cm.AudioClock() as clock:
            clock.play_rate = 0.5
            clock.start()

            time1 = clock.get_time_seconds()
            time.sleep(0.1)  # Wait 100ms
            time2 = clock.get_time_seconds()

            clock.stop()

            # Clock should have advanced approximately 0.05 seconds (half of real time)
            delta = time2 - time1
            assert 0.04 < delta < 0.06, f"Expected ~0.05s at 0.5x, got {delta:.4f}s"

    def test_clock_repr(self):
        """Test string representation"""
        clock = cm.AudioClock()
        repr_str = repr(clock)
        assert "AudioClock" in repr_str

        clock.create()
        repr_str = repr(clock)
        assert "created" in repr_str

        clock.start()
        repr_str = repr(clock)
        assert "running" in repr_str

        clock.dispose()
        repr_str = repr(clock)
        assert "disposed" in repr_str

    def test_multiple_clocks(self):
        """Test creating multiple clocks simultaneously"""
        clock1 = cm.AudioClock()
        clock2 = cm.AudioClock()

        try:
            clock1.create()
            clock2.create()

            clock1.play_rate = 1.0
            clock2.play_rate = 2.0

            assert abs(clock1.play_rate - 1.0) < 0.01
            assert abs(clock2.play_rate - 2.0) < 0.01
        finally:
            clock1.dispose()
            clock2.dispose()


class TestAudioClockErrorHandling:
    """Test error handling in AudioClock"""

    def test_operations_on_disposed_clock(self):
        """Test that operations on disposed clock raise errors"""
        clock = cm.AudioClock()
        clock.create()
        clock.dispose()

        with pytest.raises(RuntimeError):
            clock.start()

    def test_get_time_formats(self):
        """Test get_current_time with different formats"""
        with cm.AudioClock() as clock:
            # Test seconds format
            time_info = clock.get_current_time(cm.ClockTimeFormat.SECONDS)
            assert isinstance(time_info, dict)
            assert "value" in time_info

            # Test beats format
            time_info = clock.get_current_time(cm.ClockTimeFormat.BEATS)
            assert isinstance(time_info, dict)
            assert "value" in time_info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
