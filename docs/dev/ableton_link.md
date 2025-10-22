# Ableton Link Integration Analysis for CoreMusic

## Overview

This document analyzes how Ableton Link can be integrated with the coremusic project to provide tempo synchronization and network music capabilities.

## What is Ableton Link?

Ableton Link is a **tempo synchronization protocol** that allows multiple music applications running on different devices to:

- Share a common **tempo (BPM)**
- Synchronize **beat grid/phase** across applications
- Share **transport state** (play/stop) optionally
- Automatically discover and connect peers on the local network

### Key Features

- **Zero-configuration networking**: Automatic peer discovery
- **Sub-millisecond synchronization**: Precise beat alignment
- **Clock drift compensation**: Maintains sync despite hardware differences
- **Realtime-safe**: Designed for audio thread usage
- **Cross-platform**: macOS, Linux, Windows, iOS, Android
- **Open source**: GPLv2+ with proprietary licensing option

## Key Components of Link

### 1. Core API (`include/ableton/Link.hpp`)

**Main Classes:**

- `Link` - Main session participant (non-copyable, non-movable)
- `SessionState` - Timeline and transport state snapshot
- `Clock` - Platform-specific timing abstraction

**Key Methods:**

```cpp
// Link instance management
Link(double bpm);                    // Construct with initial tempo
void enable(bool bEnable);           // Start/stop network communication
bool isEnabled() const;              // Check if enabled

// Peer management
std::size_t numPeers() const;        // Number of connected peers
void setNumPeersCallback(Callback);  // Register peer count callback

// Session state access
SessionState captureAudioSessionState() const;     // From audio thread
void commitAudioSessionState(SessionState);        // From audio thread
SessionState captureAppSessionState() const;       // From app thread
void commitAppSessionState(SessionState);          // From app thread

// Transport synchronization
void enableStartStopSync(bool);      // Enable transport sync
bool isStartStopSyncEnabled() const; // Check if enabled
void setStartStopCallback(Callback); // Register transport callback

// Tempo
void setTempoCallback(Callback);     // Register tempo callback

// Clock
Clock clock() const;                 // Get platform clock
```

**SessionState API:**

```cpp
// Tempo
double tempo() const;
void setTempo(double bpm, std::chrono::microseconds atTime);

// Beat/phase queries
double beatAtTime(std::chrono::microseconds time, double quantum) const;
double phaseAtTime(std::chrono::microseconds time, double quantum) const;
std::chrono::microseconds timeAtBeat(double beat, double quantum) const;

// Beat/time mapping
void requestBeatAtTime(double beat, std::chrono::microseconds time, double quantum);
void forceBeatAtTime(double beat, std::chrono::microseconds time, double quantum);

// Transport
bool isPlaying() const;
void setIsPlaying(bool isPlaying, std::chrono::microseconds time);
std::chrono::microseconds timeForIsPlaying() const;
void requestBeatAtStartPlayingTime(double beat, double quantum);
```

### 2. Platform Support

**Darwin (macOS) Clock (`include/ableton/platforms/darwin/Clock.hpp`):**

```cpp
struct Clock {
    using Ticks = std::uint64_t;
    using Micros = std::chrono::microseconds;

    Micros ticksToMicros(const Ticks ticks) const;
    Ticks microsToTicks(const Micros micros) const;
    Ticks ticks() const;  // mach_absolute_time()
    std::chrono::microseconds micros() const;
};
```

**Key Properties:**

- Header-only C++11 library
- Requires ASIO-standalone for networking (git submodule)
- No compiled dependencies
- Platform macro: `LINK_PLATFORM_MACOSX=1`

### 3. CoreAudio Integration Example

The Link repository includes a complete CoreAudio integration example in `examples/linkaudio/AudioPlatform_CoreAudio.cpp`:

**Key Integration Points:**

```cpp
// Audio callback with Link integration
OSStatus audioCallback(
    void* inRefCon,
    AudioUnitRenderActionFlags*,
    const AudioTimeStamp* inTimeStamp,
    UInt32,
    UInt32 inNumberFrames,
    AudioBufferList* ioData)
{
    AudioEngine* engine = static_cast<AudioEngine*>(inRefCon);

    // Convert AudioTimeStamp to Link time with latency compensation
    const auto bufferBeginAtOutput =
        engine->mLink.clock().ticksToMicros(inTimeStamp->mHostTime)
        + engine->mOutputLatency.load();

    // Process audio with Link timing
    engine->audioCallback(bufferBeginAtOutput, inNumberFrames);

    // ... render audio ...

    return noErr;
}
```

**Latency Compensation:**

```cpp
// Query output latency
UInt32 deviceLatency = 0;
AudioUnitGetProperty(mIoUnit,
    kAudioDevicePropertyLatency,
    kAudioUnitScope_Output,
    0, &deviceLatency, &size);

// Convert to microseconds
const double latency = static_cast<double>(deviceLatency) / sampleRate;
mOutputLatency = duration_cast<microseconds>(duration<double>{latency});
```

## How Link Fits with CoreMusic

### Perfect Synergy

CoreMusic already has the complete infrastructure needed for Link integration:

| Link Requirement | CoreMusic Has |
|-----------------|---------------|
| System time from audio callbacks | ✅ `AudioTimeStamp` handling |
| `mach_absolute_time()` timing | ✅ `CoreAudioClock` (capi.pyx:5945) |
| Sample rate information | ✅ AudioUnit property queries |
| Output latency queries | ✅ AudioDevice property access |
| Realtime-safe buffer processing | ✅ Existing render callbacks |
| AudioUnit infrastructure | ✅ Complete AudioUnit wrapping |
| C++ interop capability | ✅ Cython with C++ support |

### Value Proposition

CoreMusic could benefit from Link by providing:

1. **Multi-device synchronization** - Sync multiple audio applications
2. **Network tempo sharing** - Collaborative music creation
3. **Beat-accurate playback** - Quantized starts/loops
4. **Transport synchronization** - Shared play/stop state
5. **DAW integration** - Sync with Ableton Live, Bitwig, etc.
6. **Professional workflows** - Industry-standard sync protocol

## Integration Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────────┐
│  Python Layer (coremusic/objects.py)            │
│  - High-level Link class                        │
│  - SessionState wrapper                         │
│  - Pythonic API (context managers, callbacks)   │
│  - Exception handling                           │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  Cython Bridge (coremusic/link.pyx)             │
│  - C++ Link class wrapper                       │
│  - SessionState wrapping                        │
│  - Callback bridging (C++ ↔ Python)             │
│  - Clock integration with CoreAudio             │
│  - Memory management                            │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  C++ Link Library (thirdparty/link)             │
│  - ableton::Link template                       │
│  - darwin::Clock using mach_absolute_time()     │
│  - Network synchronization (ASIO)               │
│  - Session state management                     │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  CoreAudio Integration                          │
│  - Existing coremusic AudioUnit callbacks       │
│  - mHostTime from AudioTimeStamp                │
│  - Latency-compensated timing                   │
│  - Realtime-safe render thread                  │
└─────────────────────────────────────────────────┘
```

## Implementation Strategy

### Phase 1: Basic Link Wrapper

Create a new Cython module for Link wrapping:

**File: `src/coremusic/link.pxd`**

```cython
from libcpp cimport bool as cbool
from libcpp.string cimport string
from libc.stdint cimport uint64_t

cdef extern from "ableton/Link.hpp" namespace "ableton":
    cdef cppclass Link:
        Link(double bpm) except +

        # Lifecycle
        cbool isEnabled() nogil
        void enable(cbool bEnable) nogil

        # Peers
        size_t numPeers() nogil

        # Session state
        SessionState captureAudioSessionState() nogil
        void commitAudioSessionState(SessionState state) nogil
        SessionState captureAppSessionState() nogil
        void commitAppSessionState(SessionState state) nogil

        # Transport sync
        cbool isStartStopSyncEnabled() nogil
        void enableStartStopSync(cbool bEnable) nogil

        # Clock
        Clock clock() nogil

    cdef cppclass SessionState:
        # Tempo
        double tempo() nogil
        void setTempo(double bpm, microseconds atTime) nogil

        # Beat/phase
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

cdef extern from "ableton/platforms/darwin/Clock.hpp" namespace "ableton::platforms::darwin":
    cdef cppclass Clock:
        uint64_t ticks() nogil
        microseconds micros() nogil
        microseconds ticksToMicros(uint64_t ticks) nogil
        uint64_t microsToTicks(microseconds micros) nogil
```

**File: `src/coremusic/link.pyx`**

```cython
from . cimport link as link_cpp
from libcpp cimport bool as cbool

cdef class LinkSession:
    """Ableton Link session for tempo synchronization"""
    cdef link_cpp.Link* _link
    cdef cbool _owned

    def __cinit__(self, double bpm=120.0):
        """Create a new Link session with initial tempo"""
        self._link = new link_cpp.Link(bpm)
        self._owned = True

    def __dealloc__(self):
        if self._owned and self._link != NULL:
            del self._link
            self._link = NULL

    @property
    def enabled(self) -> bool:
        """Check if Link is enabled (network active)"""
        return self._link.isEnabled()

    @enabled.setter
    def enabled(self, cbool value):
        """Enable or disable Link networking"""
        self._link.enable(value)

    @property
    def num_peers(self) -> int:
        """Number of connected Link peers"""
        return self._link.numPeers()

    def capture_audio_session_state(self):
        """Capture session state from audio thread (realtime-safe)"""
        cdef link_cpp.SessionState state
        with nogil:
            state = self._link.captureAudioSessionState()
        return SessionState._from_cpp(state)

    def commit_audio_session_state(self, SessionState state):
        """Commit session state from audio thread (realtime-safe)"""
        with nogil:
            self._link.commitAudioSessionState(state._state)

    # ... more methods ...

cdef class SessionState:
    """Link session state snapshot"""
    cdef link_cpp.SessionState _state

    @staticmethod
    cdef SessionState _from_cpp(link_cpp.SessionState state):
        cdef SessionState result = SessionState.__new__(SessionState)
        result._state = state
        return result

    @property
    def tempo(self) -> float:
        """Current tempo in BPM"""
        return self._state.tempo()

    def beat_at_time(self, int64_t time_micros, double quantum) -> float:
        """Get beat value at given time for quantum"""
        cdef microseconds time = microseconds(time_micros)
        return self._state.beatAtTime(time, quantum)

    def phase_at_time(self, int64_t time_micros, double quantum) -> float:
        """Get phase at given time for quantum (in range [0, quantum))"""
        cdef microseconds time = microseconds(time_micros)
        return self._state.phaseAtTime(time, quantum)

    # ... more methods ...
```

### Phase 2: AudioEngine Integration

Modify the existing `AudioPlayer` class to optionally use Link:

**File: `src/coremusic/capi.pyx` (modifications)**

```cython
# Add Link-aware audio player state
cdef struct AudioPlayerStateWithLink:
    AudioPlayerState base_state
    link_cpp.Link* link_session
    cbool use_link
    double quantum
    uint64_t output_latency_micros

# Link-aware render callback
cdef OSStatus audio_player_render_callback_with_link(
    void* inRefCon,
    AudioUnitRenderActionFlags* ioActionFlags,
    const AudioTimeStamp* inTimeStamp,
    UInt32 inBusNumber,
    UInt32 inNumberFrames,
    AudioBufferList* ioData
) nogil:
    cdef AudioPlayerStateWithLink* state = <AudioPlayerStateWithLink*>inRefCon

    if not state.use_link:
        # Fall back to non-Link behavior
        return audio_player_render_callback(&state.base_state, ioActionFlags,
                                          inTimeStamp, inBusNumber,
                                          inNumberFrames, ioData)

    # Convert CoreAudio time to Link time with latency compensation
    cdef uint64_t host_time_ticks = inTimeStamp.mHostTime
    cdef link_cpp.Clock link_clock = state.link_session.clock()
    cdef microseconds output_time_micros = link_clock.ticksToMicros(host_time_ticks)
    cdef uint64_t output_time = (output_time_micros.count() +
                                  state.output_latency_micros)

    # Capture Link session state (realtime-safe)
    cdef link_cpp.SessionState link_state = state.link_session.captureAudioSessionState()

    # Get beat/phase information
    cdef double beat = link_state.beatAtTime(microseconds(output_time), state.quantum)
    cdef double phase = link_state.phaseAtTime(microseconds(output_time), state.quantum)

    # Use beat/phase to generate audio
    # (Could trigger samples on beat boundaries, sync loops, etc.)

    # ... existing audio rendering logic ...

    return 0  # noErr

# Extended AudioPlayer class
cdef class AudioPlayer:
    cdef AudioPlayerStateWithLink* _state_with_link
    cdef object _link_session  # Python reference to keep alive

    def __init__(self, link_session=None, quantum=4.0):
        """Create audio player with optional Link integration

        Args:
            link_session: Optional LinkSession instance for tempo sync
            quantum: Beat quantum for Link synchronization (default 4.0)
        """
        # ... existing initialization ...

        if link_session is not None:
            self._link_session = link_session
            self._state_with_link.link_session = (<LinkSession>link_session)._link
            self._state_with_link.use_link = True
            self._state_with_link.quantum = quantum
        else:
            self._state_with_link.use_link = False
```

### Phase 3: Python High-Level API

**File: `src/coremusic/objects.py` (additions)**

```python
from typing import Optional, Callable
from . import capi

class Link:
    """High-level Ableton Link session wrapper

    Provides Pythonic interface to Ableton Link tempo synchronization.
    Supports context manager protocol for automatic cleanup.

    Example:
        >>> with Link(bpm=120.0) as link:
        ...     link.enabled = True
        ...     print(f"Connected peers: {link.num_peers}")
        ...
        ...     # Capture session state
        ...     state = link.capture_app_session_state()
        ...     print(f"Current tempo: {state.tempo} BPM")
    """

    def __init__(self, bpm: float = 120.0):
        """Create Link session with initial tempo

        Args:
            bpm: Initial tempo in beats per minute
        """
        self._session = capi.LinkSession(bpm)
        self._tempo_callback: Optional[Callable[[float], None]] = None
        self._peers_callback: Optional[Callable[[int], None]] = None
        self._start_stop_callback: Optional[Callable[[bool], None]] = None

    def __enter__(self) -> 'Link':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.enabled = False

    @property
    def enabled(self) -> bool:
        """Whether Link networking is enabled"""
        return self._session.enabled

    @enabled.setter
    def enabled(self, value: bool):
        """Enable or disable Link networking"""
        self._session.enabled = value

    @property
    def num_peers(self) -> int:
        """Number of currently connected Link peers"""
        return self._session.num_peers

    def capture_audio_session_state(self) -> 'SessionState':
        """Capture session state from audio thread (realtime-safe)"""
        return SessionState(self._session.capture_audio_session_state())

    def commit_audio_session_state(self, state: 'SessionState'):
        """Commit session state from audio thread (realtime-safe)"""
        self._session.commit_audio_session_state(state._state)

    def capture_app_session_state(self) -> 'SessionState':
        """Capture session state from application thread"""
        return SessionState(self._session.capture_app_session_state())

    def commit_app_session_state(self, state: 'SessionState'):
        """Commit session state from application thread"""
        self._session.commit_app_session_state(state._state)

    def set_tempo_callback(self, callback: Callable[[float], None]):
        """Register callback for tempo changes

        Args:
            callback: Function called with new tempo (BPM) as argument
        """
        self._tempo_callback = callback
        # TODO: Wire up C++ callback

    def set_num_peers_callback(self, callback: Callable[[int], None]):
        """Register callback for peer count changes

        Args:
            callback: Function called with new peer count as argument
        """
        self._peers_callback = callback
        # TODO: Wire up C++ callback

class SessionState:
    """Link session state snapshot

    Represents a snapshot of the Link timeline and transport state.
    All methods are non-blocking and realtime-safe.
    """

    def __init__(self, state):
        self._state = state

    @property
    def tempo(self) -> float:
        """Current tempo in beats per minute"""
        return self._state.tempo

    def beat_at_time(self, time_micros: int, quantum: float) -> float:
        """Get beat value at given time for quantum

        Args:
            time_micros: Time in microseconds
            quantum: Beat quantum (typically 4.0 for 4/4)

        Returns:
            Beat value at the given time
        """
        return self._state.beat_at_time(time_micros, quantum)

    def phase_at_time(self, time_micros: int, quantum: float) -> float:
        """Get phase at given time for quantum

        Args:
            time_micros: Time in microseconds
            quantum: Beat quantum (typically 4.0 for 4/4)

        Returns:
            Phase in range [0, quantum)
        """
        return self._state.phase_at_time(time_micros, quantum)

    def set_tempo(self, bpm: float, time_micros: int):
        """Set tempo at given time

        Args:
            bpm: New tempo in beats per minute
            time_micros: Time when tempo change takes effect
        """
        self._state.set_tempo(bpm, time_micros)

    @property
    def is_playing(self) -> bool:
        """Whether transport is playing"""
        return self._state.is_playing

    def set_is_playing(self, playing: bool, time_micros: int):
        """Set transport play state at given time

        Args:
            playing: Whether to start or stop transport
            time_micros: Time when transport change takes effect
        """
        self._state.set_is_playing(playing, time_micros)
```

### Phase 4: Build System Integration

**Modifications to `setup.py`:**

```python
import os
from setuptools import setup, Extension
from Cython.Build import cythonize

LIMITED_API = False
LIMITED_API_PYTHON_VERSION = 0x030A0000  # 3.10

# Link requires C++11 and specific include paths
LINK_INCLUDES = [
    "thirdparty/link/include",
    "thirdparty/link/modules/asio-standalone/asio/include",
]

os.environ['LDFLAGS'] = " ".join([
    "-framework CoreServices",
    "-framework CoreFoundation",
    "-framework AudioUnit",
    "-framework AudioToolbox",
    "-framework CoreAudio",
])

DEFINE_MACROS = [
    ("LINK_PLATFORM_MACOSX", "1"),
]

if LIMITED_API:
    DEFINE_MACROS.append(
        ("Py_LIMITED_API", LIMITED_API_PYTHON_VERSION),
    )

extensions = [
    Extension(
        "coremusic.capi",
        sources=["src/coremusic/capi.pyx"],
        include_dirs=LINK_INCLUDES,
        define_macros=DEFINE_MACROS,
        py_limited_api=LIMITED_API,
        language="c++",
        extra_compile_args=["-std=c++11"],
    ),
    Extension(
        "coremusic.link",
        sources=["src/coremusic/link.pyx"],
        include_dirs=LINK_INCLUDES,
        define_macros=DEFINE_MACROS,
        py_limited_api=LIMITED_API,
        language="c++",
        extra_compile_args=["-std=c++11"],
    ),
]

setup(
    name="coremusic",
    description="coreaudio/coremidi/ableton-link in cython",
    version="0.2.0",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'embedsignature': True,
        }
    ),
    package_dir={"": "src"},
)
```

**Update `.gitmodules`:**

```gitmodules
[submodule "thirdparty/link"]
    path = thirdparty/link
    url = https://github.com/Ableton/link.git
```

**Initialize submodules:**

```bash
git submodule update --init --recursive
```

## Key Integration Points

### 1. Timing Bridge (Critical)

The timing bridge connects CoreAudio's `mach_absolute_time()` with Link's microsecond-based timing:

```cython
# In link.pyx
cdef class LinkClock:
    """Bridge between CoreAudio timing and Link timing"""
    cdef link_cpp.Clock _clock

    def __init__(self):
        self._clock = link_cpp.Clock()

    def micros(self) -> int:
        """Get current time in microseconds (for Link)"""
        return self._clock.micros().count()

    def ticks_to_micros(self, uint64_t ticks) -> int:
        """Convert mach_absolute_time ticks to microseconds

        Args:
            ticks: Value from AudioTimeStamp.mHostTime

        Returns:
            Time in microseconds
        """
        return self._clock.ticksToMicros(ticks).count()

    def micros_to_ticks(self, int64_t micros) -> int:
        """Convert microseconds to mach_absolute_time ticks

        Args:
            micros: Time in microseconds

        Returns:
            Ticks for mach_absolute_time
        """
        cdef microseconds us = microseconds(micros)
        return self._clock.microsToTicks(us)
```

### 2. Render Callback Integration

Integrate Link into existing audio render callbacks:

```cython
cdef OSStatus audio_player_render_callback_with_link(
    void* inRefCon,
    AudioUnitRenderActionFlags* ioActionFlags,
    const AudioTimeStamp* inTimeStamp,
    UInt32 inBusNumber,
    UInt32 inNumberFrames,
    AudioBufferList* ioData
) nogil:
    cdef AudioPlayerStateWithLink* state = <AudioPlayerStateWithLink*>inRefCon

    # Step 1: Convert CoreAudio time to Link time
    cdef uint64_t host_time_ticks = inTimeStamp.mHostTime
    cdef link_cpp.Clock link_clock = state.link_session.clock()
    cdef microseconds host_time_micros = link_clock.ticksToMicros(host_time_ticks)

    # Step 2: Add output latency compensation
    cdef uint64_t output_time = (host_time_micros.count() +
                                  state.output_latency_micros)

    # Step 3: Capture Link session state (realtime-safe, no allocation)
    cdef link_cpp.SessionState link_state
    link_state = state.link_session.captureAudioSessionState()

    # Step 4: Get beat/phase information for this buffer
    cdef microseconds output_time_us = microseconds(output_time)
    cdef double beat = link_state.beatAtTime(output_time_us, state.quantum)
    cdef double phase = link_state.phaseAtTime(output_time_us, state.quantum)
    cdef double tempo = link_state.tempo()

    # Step 5: Use beat/phase to drive audio generation
    # Examples:
    # - Trigger metronome clicks on beat boundaries
    # - Restart loops at quantized beat positions
    # - Modulate parameters based on phase
    # - Synchronize LFOs to beat grid

    cdef UInt32 frame_idx
    cdef double frame_time_micros, frame_beat, frame_phase
    cdef cbool is_beat_boundary

    for frame_idx in range(inNumberFrames):
        # Calculate precise time for this sample
        frame_time_micros = output_time + (frame_idx * 1000000.0 / state.base_state.sample_rate)
        frame_beat = link_state.beatAtTime(microseconds(<int64_t>frame_time_micros), state.quantum)
        frame_phase = frame_beat % 1.0

        # Detect beat boundaries (phase wrapping)
        is_beat_boundary = (frame_phase < 0.01) if frame_idx > 0 else False

        # Generate audio sample based on Link timing
        # ... (existing audio rendering logic with Link-aware timing)

    return 0  # noErr
```

### 3. Latency Compensation

Query and apply output latency for accurate synchronization:

```python
def setup_link_with_audio_unit(audio_unit, link_session):
    """Configure Link with proper latency compensation

    Args:
        audio_unit: AudioUnit instance
        link_session: Link session instance
    """
    import coremusic as cm

    # Query sample rate
    sample_rate = cm.audio_unit_get_property(
        audio_unit.unit_id,
        cm.get_audio_unit_property_sample_rate(),
        cm.get_audio_unit_scope_output(),
        0
    )

    # Query output latency (in frames)
    device_latency_frames = cm.audio_unit_get_property(
        audio_unit.unit_id,
        cm.get_audio_device_property_latency(),
        cm.get_audio_unit_scope_output(),
        0
    )

    # Query buffer size
    buffer_size = cm.audio_unit_get_property(
        audio_unit.unit_id,
        cm.get_audio_device_property_buffer_frame_size(),
        cm.get_audio_unit_scope_global(),
        0
    )

    # Calculate total latency in microseconds
    # Note: buffer and stream latencies are included in mHostTime,
    # but device latency is not
    latency_seconds = device_latency_frames / sample_rate
    latency_micros = int(latency_seconds * 1_000_000)

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Device latency: {device_latency_frames} frames ({latency_seconds*1000:.2f} ms)")
    print(f"Buffer size: {buffer_size} frames")

    return latency_micros
```

## Example Use Cases

### Example 1: Simple Tempo Sync

```python
import coremusic as cm
import time

# Create Link session
link = cm.Link(bpm=120.0)
link.enabled = True

print(f"Link enabled, waiting for peers...")

# Monitor connection
while True:
    state = link.capture_app_session_state()
    print(f"Peers: {link.num_peers}, Tempo: {state.tempo:.1f} BPM", end='\r')
    time.sleep(0.1)
```

### Example 2: Beat-Synchronized Audio Playback

```python
import coremusic as cm

# Create Link session
link = cm.Link(bpm=120.0)
link.enable()

# Create audio player with Link integration
player = cm.AudioPlayer(link=link, quantum=4.0)
player.load_file("loop.wav")

# Capture current session state
state = link.capture_app_session_state()
current_time = link.clock().micros()

# Calculate next bar boundary (quantum = 4 beats)
current_beat = state.beat_at_time(current_time, 4.0)
next_bar = (int(current_beat / 4) + 1) * 4.0

# Request to start at next bar
state.request_beat_at_time(next_bar, current_time, 4.0)
state.set_is_playing(True, current_time)

# Commit the state
link.commit_app_session_state(state)

# Start playback - will quantize to next bar automatically
player.start()

# Wait
input("Press Enter to stop...")

# Stop playback
player.stop()
link.enabled = False
```

### Example 3: Link-Driven Metronome

```python
import coremusic as cm
import numpy as np

class LinkMetronome:
    """Beat-accurate metronome synchronized via Link"""

    def __init__(self, bpm=120.0, quantum=4.0):
        self.link = cm.Link(bpm=bpm)
        self.quantum = quantum
        self.link.enabled = True

        # Generate click sounds
        self.click_strong = self._generate_click(880, 0.05)  # A5 for downbeat
        self.click_weak = self._generate_click(440, 0.05)    # A4 for other beats

    def _generate_click(self, frequency, duration, sample_rate=44100):
        """Generate a sine wave click"""
        samples = int(duration * sample_rate)
        t = np.linspace(0, duration, samples)
        envelope = np.exp(-t * 20)  # Decay envelope
        return (np.sin(2 * np.pi * frequency * t) * envelope * 0.5).astype(np.float32)

    def audio_callback(self, host_time_micros, num_frames, sample_rate):
        """Realtime audio callback with Link synchronization"""
        # Capture Link state
        state = self.link.capture_audio_session_state()

        # Generate audio for this buffer
        output = np.zeros(num_frames, dtype=np.float32)

        for i in range(num_frames):
            # Calculate time for this sample
            sample_time = host_time_micros + (i * 1_000_000 / sample_rate)

            # Get beat and phase
            beat = state.beat_at_time(int(sample_time), self.quantum)
            phase = state.phase_at_time(int(sample_time), self.quantum)

            # Detect beat boundaries
            if i > 0:
                prev_phase = beat % 1.0
                curr_phase = (beat + 1.0/sample_rate) % 1.0

                if curr_phase < prev_phase:  # Phase wrapped - beat occurred
                    beat_num = int(beat % self.quantum)

                    # Use strong click for downbeat, weak for others
                    click = self.click_strong if beat_num == 0 else self.click_weak

                    # Mix click into output
                    for j in range(min(len(click), num_frames - i)):
                        output[i + j] += click[j]

        return output

    def run(self):
        """Run the metronome"""
        # Set up AudioUnit and start
        # ... (implementation using cm.AudioUnit)
        pass

# Usage
metro = LinkMetronome(bpm=120.0)
metro.run()
```

### Example 4: Multi-Device Jam Session

```python
import coremusic as cm
import time

class LinkJamSession:
    """Coordinate a multi-device music session with Link"""

    def __init__(self, device_name="Python Musician", initial_bpm=120.0):
        self.device_name = device_name
        self.link = cm.Link(bpm=initial_bpm)
        self.quantum = 4.0  # 4/4 time signature

        # Register callbacks
        self.link.set_num_peers_callback(self.on_peers_changed)
        self.link.set_tempo_callback(self.on_tempo_changed)
        self.link.set_start_stop_callback(self.on_transport_changed)

        # Enable start/stop sync
        self.link.enable_start_stop_sync(True)
        self.link.enabled = True

        print(f"{device_name} joined the Link session at {initial_bpm} BPM")

    def on_peers_changed(self, num_peers):
        """Called when peers join/leave"""
        if num_peers == 0:
            print("Playing solo - waiting for others to join...")
        else:
            print(f"Now jamming with {num_peers} other musician(s)!")

    def on_tempo_changed(self, new_tempo):
        """Called when anyone changes the tempo"""
        print(f"Tempo changed to {new_tempo:.1f} BPM")

    def on_transport_changed(self, is_playing):
        """Called when anyone starts/stops transport"""
        print("Transport: PLAYING" if is_playing else "Transport: STOPPED")

    def change_tempo(self, new_bpm):
        """Change the session tempo"""
        state = self.link.capture_app_session_state()
        current_time = self.link.clock().micros()
        state.set_tempo(new_bpm, current_time)
        self.link.commit_app_session_state(state)
        print(f"Set tempo to {new_bpm} BPM")

    def start_playing(self):
        """Start transport at next quantum boundary"""
        state = self.link.capture_app_session_state()
        current_time = self.link.clock().micros()

        # Quantize start to next bar
        current_beat = state.beat_at_time(current_time, self.quantum)
        next_bar = (int(current_beat / self.quantum) + 1) * self.quantum

        state.request_beat_at_start_playing_time(next_bar, self.quantum)
        state.set_is_playing(True, current_time)

        self.link.commit_app_session_state(state)
        print("Starting playback at next bar...")

    def stop_playing(self):
        """Stop transport immediately"""
        state = self.link.capture_app_session_state()
        current_time = self.link.clock().micros()
        state.set_is_playing(False, current_time)
        self.link.commit_app_session_state(state)
        print("Stopping playback")

    def monitor(self, duration=60):
        """Monitor the session"""
        print(f"Monitoring Link session for {duration} seconds...")
        start_time = time.time()

        while time.time() - start_time < duration:
            state = self.link.capture_app_session_state()
            current_time = self.link.clock().micros()
            beat = state.beat_at_time(current_time, self.quantum)

            status = "PLAYING" if state.is_playing else "STOPPED"
            print(f"[{status}] Beat: {beat:7.2f} | "
                  f"Tempo: {state.tempo:6.1f} BPM | "
                  f"Peers: {self.link.num_peers}", end='\r')

            time.sleep(0.1)

        print("\nSession ended")
        self.link.enabled = False

# Usage
session = LinkJamSession("Python Synth")
session.start_playing()
session.monitor(duration=30)
```

## Benefits of Integration

### Immediate Value

1. **Multi-device synchronization** - Sync multiple Python audio scripts
2. **DAW integration** - Work alongside Ableton Live, Bitwig, etc.
3. **Network music** - Collaborative performances over LAN
4. **Beat-accurate loops** - Quantized playback start/stop
5. **Professional workflows** - Industry-standard sync in Python

### Technical Advantages

1. **No additional frameworks** - Link is header-only C++
2. **Proven technology** - Used in 100+ professional applications
3. **Robust networking** - Auto-discovery, reconnection, NAT traversal
4. **Realtime-safe** - Designed for audio threads from the ground up
5. **Cross-platform** - Works on macOS, Linux, Windows
6. **Active development** - Maintained by Ableton

### CoreMusic Enhancement

1. **Complete audio framework** - CoreAudio + MIDI + synchronization
2. **Unique capability** - No other Python Link wrapper exists
3. **Low-level access** - Full control over both CoreAudio and Link
4. **Professional positioning** - Enables serious music production in Python
5. **Educational value** - Learn professional audio synchronization

## Challenges & Solutions

| Challenge | Impact | Solution |
|-----------|--------|----------|
| C++ template wrapping in Cython | Medium | Use explicit instantiation: `BasicLink<platform::Clock>` → `Link` |
| Callback marshaling (C++ → Python) | High | Use Cython cppclass with `except +` and trampoline functions |
| ASIO submodule dependency | Low | Document git submodule initialization in README |
| Thread safety (Link callbacks) | High | Use GIL-aware callback dispatch with proper locking |
| Build complexity (C++11) | Medium | Update setup.py with `language="c++"`, `extra_compile_args=["-std=c++11"]` |
| Memory management (C++ objects) | Medium | Use Cython `__dealloc__` for automatic cleanup |
| Microsecond timing precision | Low | Already available via `mach_absolute_time()` in CoreAudio |
| Testing without multiple devices | Medium | Use Link's example apps for interop testing |

## Testing Strategy

### Unit Tests

```python
# tests/test_link.py
import coremusic as cm
import time

def test_link_creation():
    """Test Link instance creation"""
    link = cm.Link(bpm=120.0)
    assert not link.enabled
    assert link.num_peers == 0

def test_link_enable_disable():
    """Test enabling and disabling Link"""
    link = cm.Link(bpm=120.0)
    link.enabled = True
    assert link.enabled
    link.enabled = False
    assert not link.enabled

def test_session_state_capture():
    """Test session state capture"""
    link = cm.Link(bpm=120.0)
    state = link.capture_app_session_state()
    assert state.tempo == 120.0

def test_tempo_change():
    """Test tempo modification"""
    link = cm.Link(bpm=120.0)
    state = link.capture_app_session_state()
    current_time = link.clock().micros()
    state.set_tempo(140.0, current_time)
    link.commit_app_session_state(state)

    state2 = link.capture_app_session_state()
    assert abs(state2.tempo - 140.0) < 0.1

def test_beat_calculation():
    """Test beat time calculations"""
    link = cm.Link(bpm=120.0)
    link.enabled = True

    state = link.capture_app_session_state()
    current_time = link.clock().micros()

    # At 120 BPM, 1 beat = 0.5 seconds = 500000 microseconds
    future_time = current_time + 500000

    beat_now = state.beat_at_time(current_time, 4.0)
    beat_future = state.beat_at_time(future_time, 4.0)

    # Should be approximately 1 beat difference
    assert abs((beat_future - beat_now) - 1.0) < 0.01

def test_phase_calculation():
    """Test phase calculations"""
    link = cm.Link(bpm=120.0)
    state = link.capture_app_session_state()
    current_time = link.clock().micros()

    phase = state.phase_at_time(current_time, 4.0)
    assert 0.0 <= phase < 4.0
```

### Integration Tests

```python
# tests/test_link_integration.py
import coremusic as cm
import subprocess
import time
import pytest

def test_two_instance_sync():
    """Test synchronization between two Link instances"""
    link1 = cm.Link(bpm=120.0)
    link2 = cm.Link(bpm=120.0)

    link1.enabled = True
    link2.enabled = True

    # Wait for discovery
    time.sleep(1.0)

    # Should see each other
    assert link1.num_peers >= 1
    assert link2.num_peers >= 1

    # Change tempo on link1
    state1 = link1.capture_app_session_state()
    current_time = link1.clock().micros()
    state1.set_tempo(140.0, current_time)
    link1.commit_app_session_state(state1)

    # Wait for sync
    time.sleep(0.5)

    # link2 should see the tempo change
    state2 = link2.capture_app_session_state()
    assert abs(state2.tempo - 140.0) < 1.0

    link1.enabled = False
    link2.enabled = False

@pytest.mark.skipif("not has_linkhut_example()")
def test_interop_with_linkhut():
    """Test interoperability with official Link example"""
    # Start LinkHut example application
    linkhut = subprocess.Popen(
        ["thirdparty/link/build/bin/LinkHut"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )

    time.sleep(1.0)

    # Create Python Link instance
    link = cm.Link(bpm=120.0)
    link.enabled = True

    # Should discover LinkHut
    time.sleep(1.0)
    assert link.num_peers >= 1

    # Cleanup
    link.enabled = False
    linkhut.terminate()
```

## Documentation Requirements

### README Updates

Add Link to feature list:

```markdown
## Features

- Complete CoreAudio framework wrapping
- CoreMIDI support
- **Ableton Link tempo synchronization**
- Real-time audio processing
- Professional audio I/O
```

### Installation Instructions

```markdown
## Installation

### Prerequisites

- macOS 10.11 or later
- Xcode Command Line Tools
- Python 3.8+

### Install from source

```bash
# Clone repository
git clone https://github.com/yourusername/coremusic.git
cd coremusic

# Initialize Link submodule
git submodule update --init --recursive

# Install
pip install -e .
```

### API Documentation

Create comprehensive API docs:

- `docs/api/link.md` - Link class documentation
- `docs/api/session_state.md` - SessionState documentation
- `docs/examples/link_metronome.md` - Metronome example
- `docs/examples/link_sync.md` - Multi-device sync example

## Performance Considerations

### Realtime Safety

Link is designed to be realtime-safe:

- `captureAudioSessionState()` - No allocation, lock-free
- `commitAudioSessionState()` - No allocation, lock-free
- `beatAtTime()` / `phaseAtTime()` - Pure computation, no I/O
- Clock operations - No system calls in critical path

### Memory Management

- Link uses lock-free data structures internally
- No heap allocation in audio thread methods
- Cython wrappers should avoid Python object creation in callbacks
- Use `nogil` blocks for realtime operations

### Network Performance

- Link uses UDP multicast for discovery
- Synchronization packets are < 100 bytes
- Typical bandwidth: < 10 KB/s
- Latency: typically < 1ms on LAN

## Maintenance Considerations

### Upstream Updates

Monitor Ableton Link repository:

- Repository: https://github.com/Ableton/link
- Stable releases: Use tagged versions
- API stability: Link API has been stable since 2016

### Testing Matrix

| Configuration | Status |
|--------------|--------|
| macOS 12+ (Intel) | Primary target |
| macOS 12+ (ARM64) | Primary target |
| Python 3.8-3.12 | All supported |
| Xcode 13+ | Required |

### Dependencies

- Ableton Link (thirdparty/link) - GPLv2+
- ASIO standalone (Link dependency) - Boost license
- CoreAudio frameworks (system) - macOS only

## Timeline Estimate

### Phase 1: Basic Wrapper (2-3 days)

- [ ] Create `link.pxd` with C++ declarations
- [ ] Create `link.pyx` with Cython wrappers
- [ ] Implement `LinkSession` class
- [ ] Implement `SessionState` class
- [ ] Add to build system
- [ ] Basic unit tests

### Phase 2: CoreAudio Integration (2-3 days)

- [ ] Extend `AudioPlayer` with Link support
- [ ] Implement Link-aware render callback
- [ ] Add latency compensation
- [ ] Clock integration
- [ ] Integration tests

### Phase 3: Python API (1-2 days)

- [ ] High-level `Link` class in `objects.py`
- [ ] Callback support (Python functions)
- [ ] Context manager protocol
- [ ] Type hints and docstrings

### Phase 4: Testing & Documentation (2 days)

- [ ] Comprehensive unit tests
- [ ] Integration tests with LinkHut
- [ ] Example scripts
- [ ] API documentation
- [ ] Tutorial documentation
- [ ] README updates

**Total Estimate: 7-10 days**

## Success Criteria

Integration is complete when:

- [ ] Link instance can be created and enabled from Python
- [ ] Multiple Python scripts can sync tempo over network
- [ ] Python app can sync with Ableton Live / LinkHut
- [ ] AudioPlayer can use Link for beat-accurate playback
- [ ] All unit tests pass (>95% coverage)
- [ ] Integration tests pass with official Link examples
- [ ] Documentation is complete and clear
- [ ] Example scripts demonstrate key use cases
- [ ] Build system works cleanly on macOS

## Recommendation

**Priority: HIGH**

This integration provides exceptional value:

| Criterion | Assessment |
|-----------|-----------|
| Technical Feasibility | [ ] All prerequisites exist |
| Architecture Fit | [ ] Clean integration with existing design |
| Value Proposition | [ ] Unique capability in Python |
| Risk Level | [ ] Stable, proven technology |
| Maintenance Burden | [ ] Header-only, minimal dependencies |
| Community Impact | [ ] Enables professional workflows |

The integration would position coremusic as one of the most complete audio frameworks for Python, enabling professional music production workflows that currently require C++, Swift, or commercial DAWs.

## References

- [Ableton Link Repository](https://github.com/Ableton/link)
- [Link Documentation](http://ableton.github.io/link)
- [Link Guidelines PDF](thirdparty/link/Ableton%20Link%20Guidelines.pdf)
- [Link Test Plan](thirdparty/link/TEST-PLAN.md)
- [CoreAudio Programming Guide](https://developer.apple.com/library/archive/documentation/MusicAudio/Conceptual/CoreAudioOverview/)
- [mach_absolute_time() Documentation](https://developer.apple.com/documentation/kernel/1462446-mach_absolute_time)
