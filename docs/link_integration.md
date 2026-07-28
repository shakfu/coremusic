# Ableton Link Integration

CoreMusic provides complete integration with Ableton Link, enabling tempo synchronization, beat grid alignment, and transport control across devices and applications.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Link Basics](#link-basics)
- [Link + CoreAudio](#link--coreaudio)
- [Link + CoreMIDI](#link--coremidi)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)

## Overview

Ableton Link is a technology that synchronizes musical beat, tempo, and phase across multiple applications running on one or more devices. CoreMusic integrates Link with both CoreAudio and CoreMIDI for comprehensive music synchronization.

### What Link Provides

- **Tempo Synchronization**: Share tempo (BPM) across applications
- **Beat Grid Alignment**: Align beats and bars across devices
- **Transport Control**: Synchronized start/stop across applications
- **Network Sync**: Works over local network (WiFi/Ethernet)
- **Low Latency**: Typically < 1ms on LAN

### CoreMusic Link Features

1. **Link Session Management** - Complete Python wrapper for Link
2. **Link + AudioPlayer** - Synchronized audio playback
3. **Link + MIDI Clock** - MIDI clock messages synced to Link
4. **Link + MIDI Sequencer** - Beat-accurate MIDI events

## Quick Start

### Basic Link Session

```python
--8<-- "examples/link_integration/snippet_01.py:example"
```

### Query Beat Position

```python
--8<-- "examples/link_integration/snippet_02.py:example"
```

### Change Tempo

```python
--8<-- "examples/link_integration/snippet_03.py:example"
```

## Link Basics

### LinkSession Class

The main interface to Ableton Link.

```python
--8<-- "examples/link_integration/snippet_04.py:example"
```

### SessionState Class

Snapshot of Link timeline and transport state.

```python
# Capture state (thread-safe)
state = session.capture_app_session_state()

# Read tempo
tempo = state.tempo

# Read transport state
is_playing = state.is_playing

# Get beat at specific time
beat = state.beat_at_time(time_micros, quantum=4.0)

# Get phase (position within bar)
phase = state.phase_at_time(time_micros, quantum=4.0)

# Modify and commit
state.set_tempo(140.0, time_micros)
state.set_is_playing(True, time_micros)
session.commit_app_session_state(state)
```

### Clock Class

Platform-specific timing for Link.

```python
clock = session.clock

# Get current time in microseconds
time_micros = clock.micros()

# Get system ticks (mach_absolute_time)
ticks = clock.ticks()

# Convert between formats
micros = clock.ticks_to_micros(ticks)
ticks = clock.micros_to_ticks(micros)
```

## Link + CoreAudio

Integrate Link with CoreAudio for synchronized audio playback.

### AudioPlayer with Link

```python
--8<-- "examples/link_integration/snippet_07.py:example"
```

### Real-Time Beat Monitoring

```python
--8<-- "examples/link_integration/snippet_08.py:example"
```

### Quantized Playback Start

```python
--8<-- "examples/link_integration/snippet_09.py:example"
```

### Multiple Players Synchronized

```python
--8<-- "examples/link_integration/snippet_10.py:example"
```

## Link + CoreMIDI

Integrate Link with CoreMIDI for synchronized MIDI.

### MIDI Clock Synchronization

Send MIDI Clock messages (0xF8) synchronized to Link tempo.

```python
--8<-- "examples/link_integration/snippet_11.py:example"
```

### Beat-Accurate MIDI Sequencing

Schedule MIDI events at specific Link beat positions.

```python
--8<-- "examples/link_integration/snippet_12.py:example"
```

### MIDI CC Automation Synchronized to Link

```python
--8<-- "examples/link_integration/snippet_13.py:example"
```

### Looping MIDI Patterns

```python
--8<-- "examples/link_integration/snippet_14.py:example"
```

### Combined Audio + MIDI Synchronized

```python
--8<-- "examples/link_integration/snippet_15.py:example"
```

## API Reference

### LinkSession

```python
class LinkSession:
    """Main Link session interface"""

    def __init__(self, bpm: float = 120.0):
        """Create Link session with initial tempo"""

    def __enter__(self) -> 'LinkSession':
        """Context manager: enables Link"""

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager: disables Link"""

    # Properties
    enabled: bool  # Enable/disable networking
    num_peers: int  # Number of connected peers
    start_stop_sync_enabled: bool  # Transport sync
    clock: Clock  # Platform clock

    # Methods
    def capture_app_session_state(self) -> SessionState:
        """Capture state from app thread"""

    def commit_app_session_state(self, state: SessionState):
        """Commit state from app thread"""

    def capture_audio_session_state(self) -> SessionState:
        """Capture state from audio thread (realtime-safe)"""

    def commit_audio_session_state(self, state: SessionState):
        """Commit state from audio thread (realtime-safe)"""
```

### SessionState

```python
class SessionState:
    """Link timeline and transport snapshot"""

    # Properties
    tempo: float  # Current tempo in BPM
    is_playing: bool  # Transport state

    # Beat/Phase Queries
    def beat_at_time(self, time_micros: int, quantum: float) -> float:
        """Get beat at time"""

    def phase_at_time(self, time_micros: int, quantum: float) -> float:
        """Get phase (0 to quantum)"""

    def time_at_beat(self, beat: float, quantum: float) -> int:
        """Get time for beat"""

    # State Modification
    def set_tempo(self, bpm: float, time_micros: int):
        """Set tempo at time"""

    def set_is_playing(self, playing: bool, time_micros: int):
        """Set transport state"""

    def request_beat_at_time(self, beat: float, time_micros: int, quantum: float):
        """Request beat mapping (quantized if peers present)"""
```

### LinkMIDIClock

```python
class LinkMIDIClock:
    """MIDI Clock synchronized to Link"""

    def __init__(
        self,
        session: LinkSession,
        midi_port: int,
        midi_destination: int,
        quantum: float = 4.0
    ):
        """Create MIDI clock"""

    def start(self):
        """Start sending clock (sends MIDI Start)"""

    def stop(self):
        """Stop sending clock (sends MIDI Stop)"""
```

### LinkMIDISequencer

```python
class LinkMIDISequencer:
    """Beat-accurate MIDI sequencer"""

    def __init__(
        self,
        session: LinkSession,
        midi_port: int,
        midi_destination: int,
        quantum: float = 4.0
    ):
        """Create sequencer"""

    def schedule_event(self, beat: float, message: bytes):
        """Schedule MIDI message"""

    def schedule_note(
        self,
        beat: float,
        channel: int,
        note: int,
        velocity: int,
        duration: float
    ):
        """Schedule note with automatic note-off"""

    def schedule_cc(self, beat: float, channel: int, controller: int, value: int):
        """Schedule CC message"""

    def clear_events(self):
        """Clear all scheduled events"""

    def start(self):
        """Start sequencer"""

    def stop(self):
        """Stop sequencer"""
```

## Best Practices

### Thread Safety

- Use `capture_app_session_state()` from non-audio threads
- Use `capture_audio_session_state()` from audio threads only
- Link operations are realtime-safe (`nogil`)

### Timing Accuracy

- Query Link state as close to use time as possible
- Apply output latency compensation for audio sync
- Use high-resolution timing for MIDI events

### Resource Management

- Use context managers (`with` statement) for automatic cleanup
- Disable Link when not in use to save network bandwidth
- Dispose MIDI clients/ports properly

### Performance

- Link state capture is lock-free (no blocking)
- Keep quantum consistent across queries
- Minimize work in audio/MIDI threads

### Networking

- Link uses UDP multicast for discovery
- Requires local network access
- Typical latency < 1ms on LAN

## Troubleshooting

### No Peers Found

- Check firewall settings (allow UDP multicast)
- Ensure devices on same network
- Try enabling/disabling Link

### Timing Drift

- Verify quantum is consistent
- Check for output latency compensation
- Ensure high-resolution timing

### MIDI Not Sending

- Check MIDI port/destination IDs
- Verify MIDI device is connected
- Test with simple MIDI message first

## Additional Resources

- [Ableton Link Official Site](https://www.ableton.com/en/link/)
- [Link GitHub Repository](https://github.com/Ableton/link)
- [CoreMusic demos](https://github.com/shakfu/coremusic/tree/main/demos) - including `link_sequencer.py`, a Link-synced step sequencer
- [CoreAudio Documentation](https://developer.apple.com/documentation/coreaudio)
- [CoreMIDI Documentation](https://developer.apple.com/documentation/coremidi)
