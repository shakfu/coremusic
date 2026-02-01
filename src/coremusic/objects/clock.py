"""Audio clock classes for coremusic.

This module provides classes for audio/MIDI synchronization and timing:
- ClockTimeFormat: Time format constants
- AudioClock: High-level CoreAudioClock for synchronization
"""

from __future__ import annotations

from typing import Any, Dict

from .. import capi

__all__ = [
    "ClockTimeFormat",
    "AudioClock",
]


class ClockTimeFormat:
    """Time format constants for CoreAudioClock"""
    HOST_TIME = capi.get_ca_clock_time_format_host_time()
    SAMPLES = capi.get_ca_clock_time_format_samples()
    BEATS = capi.get_ca_clock_time_format_beats()
    SECONDS = capi.get_ca_clock_time_format_seconds()
    SMPTE_TIME = capi.get_ca_clock_time_format_smpte_time()


class AudioClock(capi.CoreAudioObject):
    """High-level CoreAudioClock for audio/MIDI synchronization and timing

    AudioClock provides synchronization services for audio and MIDI applications,
    supporting multiple time formats and playback control.

    Supported time formats:

    - Host time (mach_absolute_time)
    - Audio samples
    - Musical beats
    - Seconds
    - SMPTE timecode

    Example::

        # Create and control a clock
        with AudioClock() as clock:
            clock.play_rate = 1.0
            clock.start()

            # Get current time in different formats
            seconds = clock.get_time_seconds()
            beats = clock.get_time_beats()

            print(f"Position: {seconds:.2f}s ({beats:.2f} beats)")

            clock.stop()

    Example with tempo and speed control::

        clock = AudioClock()
        clock.play_rate = 0.5  # Half speed
        clock.start()
        # ... use clock for synchronization
        clock.stop()
        clock.dispose()
    """

    def __init__(self):
        """Initialize a new CoreAudioClock"""
        super().__init__()
        self._is_created = False
        self._is_running = False

    def create(self) -> "AudioClock":
        """Create the underlying clock object

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If clock creation fails
        """
        if not self._is_created:
            try:
                clock_id = capi.ca_clock_new()
                self._set_object_id(clock_id)
                self._is_created = True
            except Exception as e:
                raise RuntimeError(f"Failed to create clock: {e}")
        return self

    def start(self) -> None:
        """Start the clock advancing on its timeline

        Raises:
            RuntimeError: If clock is not created or start fails
        """
        self._ensure_not_disposed()
        if not self._is_created:
            self.create()

        try:
            capi.ca_clock_start(self.object_id)
            self._is_running = True
        except Exception as e:
            raise RuntimeError(f"Failed to start clock: {e}")

    def stop(self) -> None:
        """Stop the clock

        Raises:
            RuntimeError: If stop fails
        """
        if self._is_running:
            try:
                capi.ca_clock_stop(self.object_id)
                self._is_running = False
            except Exception as e:
                raise RuntimeError(f"Failed to stop clock: {e}")

    @property
    def is_running(self) -> bool:
        """Check if the clock is currently running"""
        return self._is_running

    @property
    def play_rate(self) -> float:
        """Get or set the playback rate (1.0 = normal speed)"""
        self._ensure_not_disposed()
        if not self._is_created:
            self.create()

        try:
            return capi.ca_clock_get_play_rate(self.object_id)
        except Exception as e:
            raise RuntimeError(f"Failed to get play rate: {e}")

    @play_rate.setter
    def play_rate(self, rate: float) -> None:
        """Set the playback rate"""
        self._ensure_not_disposed()
        if not self._is_created:
            self.create()

        try:
            capi.ca_clock_set_play_rate(self.object_id, rate)
        except Exception as e:
            raise RuntimeError(f"Failed to set play rate: {e}")

    def get_current_time(self, time_format: int) -> Dict[str, Any]:
        """Get current time in specified format

        Args:
            time_format: Time format constant from ClockTimeFormat

        Returns:
            Dictionary with 'format' and 'value' keys

        Raises:
            RuntimeError: If getting time fails
        """
        self._ensure_not_disposed()
        if not self._is_created:
            self.create()

        try:
            return capi.ca_clock_get_current_time(self.object_id, time_format)
        except Exception as e:
            raise RuntimeError(f"Failed to get current time: {e}")

    def get_time_seconds(self) -> float:
        """Get current time in seconds

        Returns:
            Current time in seconds
        """
        time_info = self.get_current_time(ClockTimeFormat.SECONDS)
        return float(time_info.get("value", 0.0))

    def get_time_beats(self) -> float:
        """Get current time in musical beats

        Returns:
            Current time in beats
        """
        time_info = self.get_current_time(ClockTimeFormat.BEATS)
        return float(time_info.get("value", 0.0))

    def get_time_samples(self) -> float:
        """Get current time in audio samples

        Returns:
            Current time in samples
        """
        time_info = self.get_current_time(ClockTimeFormat.SAMPLES)
        return float(time_info.get("value", 0.0))

    def get_time_host(self) -> int:
        """Get current time as host time

        Returns:
            Current host time (mach_absolute_time)
        """
        time_info = self.get_current_time(ClockTimeFormat.HOST_TIME)
        return int(time_info.get("value", 0))

    def get_smpte_time(self) -> Dict[str, int]:
        """Get current time as SMPTE timecode

        Returns:
            Dictionary with SMPTE time components:
            - hours, minutes, seconds, frames
            - subframes, subframe_divisor
            - type, flags
        """
        time_info = self.get_current_time(ClockTimeFormat.SMPTE_TIME)
        value = time_info.get("value", {})
        if isinstance(value, dict):
            return value
        return {}

    def dispose(self) -> None:
        """Dispose the clock and free resources"""
        if not self.is_disposed and self._is_created:
            try:
                if self._is_running:
                    self.stop()
                capi.ca_clock_dispose(self.object_id)
            except Exception:
                pass  # Best effort cleanup
            finally:
                self._is_created = False
                self._is_running = False
                super().dispose()

    def __enter__(self) -> "AudioClock":
        """Enter context manager"""
        self.create()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and dispose"""
        self.dispose()

    def __repr__(self) -> str:
        status = []
        if not self.is_disposed:
            if self._is_created:
                status.append("created")
            if self._is_running:
                status.append("running")
                try:
                    rate = self.play_rate
                    status.append(f"rate={rate:.2f}")
                except Exception:
                    pass
        else:
            status.append("disposed")

        status_str = ", ".join(status) if status else "not created"
        return f"AudioClock({status_str})"
