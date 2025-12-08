# distutils: language = c++
# cython: language_level=3

"""Ableton Link C++ API declarations for Cython

This file declares the C++ Link API from thirdparty/link for use in Cython.
"""

from libc.stdint cimport int64_t, uint64_t
from libcpp cimport bool as cbool
from libcpp.string cimport string


# Forward declare std::chrono types we need
cdef extern from "<chrono>" namespace "std::chrono":
    cdef cppclass microseconds:
        microseconds() except +
        microseconds(int64_t) except +
        int64_t count() nogil

# Ableton Link C++ API
cdef extern from "ableton/Link.hpp" namespace "ableton":

    # Forward declaration of SessionState (defined within Link class)
    cdef cppclass LinkSessionState "ableton::Link::SessionState":
        # Tempo
        double tempo() nogil
        void setTempo(double bpm, microseconds atTime) nogil

        # Beat/phase queries
        double beatAtTime(microseconds time, double quantum) nogil
        double phaseAtTime(microseconds time, double quantum) nogil
        microseconds timeAtBeat(double beat, double quantum) nogil

        # Beat mapping
        void requestBeatAtTime(double beat, microseconds time, double quantum) nogil
        void forceBeatAtTime(double beat, microseconds time, double quantum) nogil

        # Transport
        cbool isPlaying() nogil
        void setIsPlaying(cbool isPlaying, microseconds time) nogil
        microseconds timeForIsPlaying() nogil
        void requestBeatAtStartPlayingTime(double beat, double quantum) nogil
        void setIsPlayingAndRequestBeatAtTime(
            cbool isPlaying,
            microseconds time,
            double beat,
            double quantum
        ) nogil

    # Main Link class
    cdef cppclass Link:
        # Constructor
        Link(double bpm) except +

        # Lifecycle - cannot copy or move
        # Link(const Link&) = delete
        # Link& operator=(const Link&) = delete

        # Enable/disable
        cbool isEnabled() nogil
        void enable(cbool bEnable) nogil

        # Start/stop sync
        cbool isStartStopSyncEnabled() nogil
        void enableStartStopSync(cbool bEnable) nogil

        # Peers
        size_t numPeers() nogil

        # Session state - audio thread (realtime-safe)
        LinkSessionState captureAudioSessionState() nogil
        void commitAudioSessionState(LinkSessionState state) nogil

        # Session state - app thread (non-realtime)
        LinkSessionState captureAppSessionState() nogil
        void commitAppSessionState(LinkSessionState state) nogil

        # Clock access
        LinkClock clock() nogil

# Platform-specific clock implementation for Darwin (macOS)
cdef extern from "ableton/platforms/darwin/Clock.hpp" namespace "ableton::platforms::darwin":
    cdef cppclass LinkClock "ableton::platforms::darwin::Clock":
        LinkClock() except +
        uint64_t ticks() nogil
        microseconds micros() nogil
        microseconds ticksToMicros(uint64_t ticks) nogil
        uint64_t microsToTicks(microseconds micros) nogil
