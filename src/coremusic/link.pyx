# distutils: language = c++
# cython: language_level=3

"""Ableton Link Cython wrapper

Provides Python interface to Ableton Link tempo synchronization library.
"""

from libc.stdint cimport int64_t, uint64_t, uintptr_t
from libcpp cimport bool as cbool
from libcpp.memory cimport make_shared, shared_ptr

# Import microseconds type from declarations
from coremusic.link cimport microseconds

from . cimport link as link_cpp


cdef class Clock:
    """Platform-specific clock for Link timing

    Provides conversion between system time (mach_absolute_time on macOS)
    and microseconds used by Link.
    """
    cdef link_cpp.LinkClock _clock

    def __cinit__(self):
        """Initialize clock"""
        self._clock = link_cpp.LinkClock()

    def micros(self) -> int:
        """Get current time in microseconds

        Returns:
            Current time in microseconds
        """
        cdef microseconds us
        with nogil:
            us = self._clock.micros()
        return us.count()

    def ticks(self) -> int:
        """Get current time in system ticks (mach_absolute_time)

        Returns:
            Current time in system ticks
        """
        cdef uint64_t t
        with nogil:
            t = self._clock.ticks()
        return t

    def ticks_to_micros(self, uint64_t ticks) -> int:
        """Convert system ticks to microseconds

        Args:
            ticks: System time from mach_absolute_time() or AudioTimeStamp.mHostTime

        Returns:
            Time in microseconds
        """
        cdef microseconds us
        with nogil:
            us = self._clock.ticksToMicros(ticks)
        return us.count()

    def micros_to_ticks(self, int64_t micros_val) -> int:
        """Convert microseconds to system ticks

        Args:
            micros_val: Time in microseconds

        Returns:
            Time in system ticks
        """
        cdef microseconds us = microseconds(micros_val)
        cdef uint64_t t
        with nogil:
            t = self._clock.microsToTicks(us)
        return t


cdef class SessionState:
    """Link session state snapshot

    Represents a snapshot of the Link timeline and transport state.
    All methods are non-blocking and realtime-safe.

    Do not store SessionState objects for later use - they represent
    a snapshot at a specific moment in time. Always capture fresh
    state when needed.

    Note: This wraps a C++ SessionState by value via shared_ptr for
    proper copy semantics.
    """
    cdef shared_ptr[link_cpp.LinkSessionState] _state

    def __cinit__(self):
        """Initialize empty session state (internal use only)"""
        pass

    @property
    def tempo(self) -> float:
        """Current tempo in beats per minute

        This is a stable value appropriate for display to the user.
        Beat time progress will not necessarily match this tempo exactly
        because of clock drift compensation.
        """
        if not self._state:
            raise RuntimeError("SessionState is not initialized")
        return self._state.get().tempo()

    def set_tempo(self, double bpm, int64_t time_micros):
        """Set tempo at given time

        Args:
            bpm: New tempo in beats per minute
            time_micros: Time in microseconds when tempo change takes effect
        """
        if not self._state:
            raise RuntimeError("SessionState is not initialized")
        cdef microseconds time_us = microseconds(time_micros)
        self._state.get().setTempo(bpm, time_us)

    def beat_at_time(self, int64_t time_micros, double quantum) -> float:
        """Get beat value at given time for the given quantum

        The magnitude of the resulting beat value is unique to this Link
        instance, but its phase with respect to the provided quantum is
        shared among all session peers.

        Args:
            time_micros: Time in microseconds
            quantum: Beat quantum (e.g., 4.0 for 4/4 time)

        Returns:
            Beat value at the given time
        """
        if not self._state:
            raise RuntimeError("SessionState is not initialized")
        cdef microseconds time_us = microseconds(time_micros)
        return self._state.get().beatAtTime(time_us, quantum)

    def phase_at_time(self, int64_t time_micros, double quantum) -> float:
        """Get phase at given time for the given quantum

        The result is in the interval [0, quantum). This is equivalent to
        fmod(beatAtTime(t, q), q) for non-negative beat values, but handles
        negative beat values correctly.

        Args:
            time_micros: Time in microseconds
            quantum: Beat quantum (e.g., 4.0 for 4/4 time)

        Returns:
            Phase in range [0, quantum)
        """
        if not self._state:
            raise RuntimeError("SessionState is not initialized")
        cdef microseconds time_us = microseconds(time_micros)
        return self._state.get().phaseAtTime(time_us, quantum)

    def time_at_beat(self, double beat, double quantum) -> int:
        """Get time at which the given beat occurs for the given quantum

        The inverse of beatAtTime, assuming a constant tempo.
        beatAtTime(timeAtBeat(b, q), q) === b

        Args:
            beat: Beat value
            quantum: Beat quantum (e.g., 4.0 for 4/4 time)

        Returns:
            Time in microseconds when beat occurs
        """
        if not self._state:
            raise RuntimeError("SessionState is not initialized")
        cdef microseconds time_us = self._state.get().timeAtBeat(beat, quantum)
        return time_us.count()

    def request_beat_at_time(self, double beat, int64_t time_micros, double quantum):
        """Attempt to map the given beat to the given time in context of quantum

        This method behaves differently depending on the state of the session:

        - If no other peers are connected, this instance is free to re-map the
          beat/time relationship. In this case, beatAtTime(time, quantum) == beat
          after this method returns.

        - If there are other peers in the session, the given beat will be mapped
          to the next time value greater than the given time with the same phase
          as the given beat.

        This enables "quantized launch" - events happen immediately when alone,
        but wait for the next aligned beat when playing with others.

        Args:
            beat: Target beat value
            time_micros: Time in microseconds
            quantum: Beat quantum (e.g., 4.0 for 4/4 time)
        """
        if not self._state:
            raise RuntimeError("SessionState is not initialized")
        cdef microseconds time_us = microseconds(time_micros)
        self._state.get().requestBeatAtTime(beat, time_us, quantum)

    def force_beat_at_time(self, double beat, int64_t time_micros, double quantum):
        """Rudely re-map the beat/time relationship for all peers

        DANGER: This method should only be needed in special circumstances.
        Most applications should not use it. Unlike requestBeatAtTime, this
        does not fall back to quantizing behavior - it unconditionally maps
        the given beat to the given time.

        This is anti-social behavior and should be avoided. One legitimate use
        is synchronizing a Link session with an external clock source.

        Args:
            beat: Target beat value
            time_micros: Time in microseconds
            quantum: Beat quantum (e.g., 4.0 for 4/4 time)
        """
        if not self._state:
            raise RuntimeError("SessionState is not initialized")
        cdef microseconds time_us = microseconds(time_micros)
        self._state.get().forceBeatAtTime(beat, time_us, quantum)

    @property
    def is_playing(self) -> bool:
        """Whether transport is playing"""
        if not self._state:
            raise RuntimeError("SessionState is not initialized")
        return self._state.get().isPlaying()

    def set_is_playing(self, cbool playing, int64_t time_micros):
        """Set transport play state at given time

        Args:
            playing: True to start transport, False to stop
            time_micros: Time in microseconds when change takes effect
        """
        if not self._state:
            raise RuntimeError("SessionState is not initialized")
        cdef microseconds time_us = microseconds(time_micros)
        self._state.get().setIsPlaying(playing, time_us)

    def time_for_is_playing(self) -> int:
        """Get the time at which the current transport state took effect

        Returns:
            Time in microseconds
        """
        if not self._state:
            raise RuntimeError("SessionState is not initialized")
        cdef microseconds time_us = self._state.get().timeForIsPlaying()
        return time_us.count()

    def request_beat_at_start_playing_time(self, double beat, double quantum):
        """Convenience function to map beat to transport start time

        Attempts to map the given beat to the time when transport is starting
        to play in context of the given quantum. This evaluates to a no-op if
        isPlaying() is False.

        Args:
            beat: Target beat value
            quantum: Beat quantum (e.g., 4.0 for 4/4 time)
        """
        if not self._state:
            raise RuntimeError("SessionState is not initialized")
        self._state.get().requestBeatAtStartPlayingTime(beat, quantum)

    def set_is_playing_and_request_beat_at_time(
        self,
        cbool playing,
        int64_t time_micros,
        double beat,
        double quantum
    ):
        """Convenience function to start/stop transport and map beat

        Combines setIsPlaying and requestBeatAtTime in one call.

        Args:
            playing: True to start transport, False to stop
            time_micros: Time in microseconds when change takes effect
            beat: Target beat value
            quantum: Beat quantum (e.g., 4.0 for 4/4 time)
        """
        if not self._state:
            raise RuntimeError("SessionState is not initialized")
        cdef microseconds time_us = microseconds(time_micros)
        self._state.get().setIsPlayingAndRequestBeatAtTime(playing, time_us, beat, quantum)


cdef class LinkSession:
    """Ableton Link session for tempo synchronization

    Each LinkSession instance represents a participant in a Link session.
    When enabled, it will discover other peers on the local network and
    synchronize tempo, beat grid, and optionally transport state.

    Example:
        >>> link = LinkSession(bpm=120.0)
        >>> link.enabled = True
        >>> print(f"Connected to {link.num_peers} peers")
        >>> state = link.capture_app_session_state()
        >>> print(f"Current tempo: {state.tempo} BPM")
    """
    cdef link_cpp.Link* _link
    cdef cbool _owned
    cdef Clock _clock

    def __cinit__(self, double bpm=120.0):
        """Create a new Link session with initial tempo

        The session starts disabled (no network communication). Call
        enable=True to start networking.

        Args:
            bpm: Initial tempo in beats per minute (default: 120.0)
        """
        self._link = new link_cpp.Link(bpm)
        self._owned = True
        self._clock = Clock()

    def __dealloc__(self):
        """Clean up Link session"""
        if self._owned and self._link != NULL:
            del self._link
            self._link = NULL

    @property
    def enabled(self) -> bool:
        """Whether Link networking is enabled

        When enabled, this instance will discover and connect to peers
        on the local network. When disabled, it operates standalone.
        """
        if self._link == NULL:
            return False
        return self._link.isEnabled()

    @enabled.setter
    def enabled(self, cbool value):
        """Enable or disable Link networking

        Args:
            value: True to enable networking, False to disable
        """
        if self._link != NULL:
            self._link.enable(value)

    @property
    def num_peers(self) -> int:
        """Number of currently connected Link peers

        Returns 0 when disabled or when no other peers are on the network.
        """
        if self._link == NULL:
            return 0
        return self._link.numPeers()

    @property
    def start_stop_sync_enabled(self) -> bool:
        """Whether start/stop transport synchronization is enabled"""
        if self._link == NULL:
            return False
        return self._link.isStartStopSyncEnabled()

    @start_stop_sync_enabled.setter
    def start_stop_sync_enabled(self, cbool value):
        """Enable or disable start/stop transport synchronization

        When enabled, this instance will share and respond to transport
        start/stop state changes with other peers that also have this
        enabled.

        Args:
            value: True to enable sync, False to disable
        """
        if self._link != NULL:
            self._link.enableStartStopSync(value)

    @property
    def clock(self) -> Clock:
        """Get the platform clock used by Link

        Returns:
            Clock instance for time conversions
        """
        return self._clock

    def capture_audio_session_state(self) -> SessionState:
        """Capture session state from audio thread (realtime-safe)

        This method should ONLY be called from the audio thread.
        It does not block and does not allocate memory.

        Returns:
            SessionState snapshot of current state
        """
        if self._link == NULL:
            raise RuntimeError("Link session is not initialized")

        # Directly create SessionState with captured C++ state
        cdef SessionState result = SessionState.__new__(SessionState)
        with nogil:
            result._state = make_shared[link_cpp.LinkSessionState](
                self._link.captureAudioSessionState()
            )
        return result

    def commit_audio_session_state(self, SessionState state):
        """Commit session state from audio thread (realtime-safe)

        This method should ONLY be called from the audio thread.
        The given session state will replace the current Link state.
        Modifications will be communicated to other peers.

        Args:
            state: SessionState with modifications to commit
        """
        if self._link == NULL:
            raise RuntimeError("Link session is not initialized")
        if not state._state:
            raise RuntimeError("SessionState is not initialized")

        with nogil:
            self._link.commitAudioSessionState(state._state.get()[0])

    def capture_app_session_state(self) -> SessionState:
        """Capture session state from application thread

        This method can be called from any thread except the audio thread.
        Use this for UI updates, user interactions, etc.

        Returns:
            SessionState snapshot of current state
        """
        if self._link == NULL:
            raise RuntimeError("Link session is not initialized")

        # Directly create SessionState with captured C++ state
        cdef SessionState result = SessionState.__new__(SessionState)
        result._state = make_shared[link_cpp.LinkSessionState](
            self._link.captureAppSessionState()
        )
        return result

    def commit_app_session_state(self, SessionState state):
        """Commit session state from application thread

        This method can be called from any thread except the audio thread.
        The given session state will replace the current Link state.
        Modifications will be communicated to other peers.

        Args:
            state: SessionState with modifications to commit
        """
        if self._link == NULL:
            raise RuntimeError("Link session is not initialized")
        if not state._state:
            raise RuntimeError("SessionState is not initialized")

        self._link.commitAppSessionState(state._state.get()[0])

    def __repr__(self) -> str:
        """String representation of Link session"""
        status = "enabled" if self.enabled else "disabled"
        peers = self.num_peers
        return f"LinkSession({status}, {peers} peers)"

    def __enter__(self):
        """Enter context manager - enables Link networking

        Returns:
            self: The LinkSession instance
        """
        self.enabled = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - disables Link networking

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)

        Returns:
            False: Do not suppress exceptions
        """
        self.enabled = False
        return False

    # Internal methods for C++ pointer access (used by AudioPlayer integration)
    def _get_link_ptr_as_int(self) -> int:
        """Get internal C++ Link pointer as integer (for C integration)

        WARNING: This is for internal use only by other Cython modules.
        Returns the memory address of the C++ Link object as a Python int.
        """
        return <uintptr_t><void*>self._link

    def _get_clock_ptr_as_int(self) -> int:
        """Get internal C++ Clock pointer as integer (for C integration)

        WARNING: This is for internal use only by other Cython modules.
        Returns the memory address of the C++ Clock object as a Python int.
        """
        return <uintptr_t><void*>&self._clock._clock
