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
import coremusic as cm

# Create Link session with context manager
with cm.link.LinkSession(bpm=120.0) as session:
    print(f"Link enabled: {session.enabled}")
    print(f"Connected peers: {session.num_peers}")

    # Get current state
    state = session.capture_app_session_state()
    print(f"Tempo: {state.tempo:.1f} BPM")
    print(f"Playing: {state.is_playing}")
```

### Query Beat Position

```python
with cm.link.LinkSession(bpm=120.0) as session:
    clock = session.clock

    # Get current beat position
    state = session.capture_app_session_state()
    current_time = clock.micros()
    beat = state.beat_at_time(current_time, quantum=4.0)
    phase = state.phase_at_time(current_time, quantum=4.0)

    print(f"Beat: {beat:.2f}, Phase: {phase:.2f}/4")
```

### Change Tempo

```python
with cm.link.LinkSession(bpm=120.0) as session:
    # Capture state
    state = session.capture_app_session_state()
    current_time = session.clock.micros()

    # Set new tempo
    state.set_tempo(140.0, current_time)
    session.commit_app_session_state(state)

    print("Tempo changed to 140 BPM")
```

## Link Basics

### LinkSession Class

The main interface to Ableton Link.

```python
# Create session with initial tempo
session = cm.link.LinkSession(bpm=120.0)

# Enable networking (discovers peers)
session.enabled = True

# Enable transport sync
session.start_stop_sync_enabled = True

# Check connections
print(f"Connected to {session.num_peers} peers")

# Access the clock
clock = session.clock

# Cleanup
session.enabled = False
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
import coremusic as cm
import time

# Create Link session
with cm.link.LinkSession(bpm=120.0) as session:
    # Create AudioPlayer with Link integration
    player = cm.AudioPlayer(link_session=session)

    # Load and setup audio
    player.load_file("loop.wav")
    player.setup_output()

    # Query Link timing before playback
    timing = player.get_link_timing(quantum=4.0)
    print(f"Starting at beat {timing['beat']:.2f}")
    print(f"Tempo: {timing['tempo']:.1f} BPM")

    # Start playback
    player.play()

    # Monitor playback with Link timing
    for _ in range(20):
        timing = player.get_link_timing(quantum=4.0)
        progress = player.get_progress()

        print(f"Beat: {timing['beat']:7.2f} | "
              f"Phase: {timing['phase']:4.2f} | "
              f"Progress: {progress*100:5.1f}%", end='\r')

        time.sleep(0.5)

    # Stop playback
    player.stop()
```

### Real-Time Beat Monitoring

```python
import coremusic as cm
import time

with cm.link.LinkSession(bpm=120.0) as session:
    player = cm.AudioPlayer(link_session=session)
    player.load_file("audio.wav")
    player.setup_output()
    player.play()
    player.start()

    # Monitor beats during playback
    while player.is_playing():
        timing = player.get_link_timing(quantum=4.0)
        beat = timing['beat']

        # Visual beat indicator
        indicator = "●" if int(beat) % 4 == 0 else "○"

        print(f"{indicator} Beat: {beat:7.2f} | "
              f"Tempo: {timing['tempo']:6.1f} BPM", end='\r')

        time.sleep(0.1)

    player.stop()
```

### Quantized Playback Start

```python
import coremusic as cm

with cm.link.LinkSession(bpm=120.0) as session:
    player = cm.AudioPlayer(link_session=session)
    player.load_file("loop.wav")
    player.setup_output()

    # Get current Link state
    state = session.capture_app_session_state()
    current_time = session.clock.micros()

    # Calculate next bar boundary (4 beats)
    current_beat = state.beat_at_time(current_time, quantum=4.0)
    next_bar = (int(current_beat / 4) + 1) * 4.0

    print(f"Current beat: {current_beat:.2f}")
    print(f"Waiting for beat {next_bar:.0f}...")

    # Wait for next bar
    while True:
        state = session.capture_app_session_state()
        current_time = session.clock.micros()
        beat = state.beat_at_time(current_time, quantum=4.0)

        if beat >= next_bar:
            break

        time.sleep(0.001)

    # Start playback exactly on the bar
    player.play()
    player.start()
    print("Started!")
```

### Multiple Players Synchronized

```python
import coremusic as cm

# Share one Link session across multiple players
with cm.link.LinkSession(bpm=120.0) as session:
    # Create multiple players
    player1 = cm.AudioPlayer(link_session=session)
    player2 = cm.AudioPlayer(link_session=session)

    player1.load_file("drums.wav")
    player2.load_file("bass.wav")

    player1.setup_output()
    player2.setup_output()

    # Both players see same Link timing
    timing1 = player1.get_link_timing()
    timing2 = player2.get_link_timing()

    assert timing1['tempo'] == timing2['tempo']
    assert abs(timing1['beat'] - timing2['beat']) < 0.01

    # Start both (synchronized via Link)
    player1.play()
    player2.play()
    player1.start()
    player2.start()
```

## Link + CoreMIDI

Integrate Link with CoreMIDI for synchronized MIDI.

### MIDI Clock Synchronization

Send MIDI Clock messages (0xF8) synchronized to Link tempo.

```python
import coremusic as cm
from coremusic import link_midi
import time

# Setup MIDI
client = cm.capi.midi_client_create("MIDI Clock")
port = cm.capi.midi_output_port_create(client, "Clock Out")
destination = cm.capi.midi_get_destination(0)

# Create Link session and MIDI clock
with cm.link.LinkSession(bpm=120.0) as session:
    # Create MIDI clock synchronized to Link
    clock = link_midi.LinkMIDIClock(session, port, destination)

    # Start sending MIDI clock
    clock.start()
    print("Sending MIDI Clock at 120 BPM")
    print("(24 clock messages per quarter note)")

    # Run for 10 seconds
    for i in range(20):
        state = session.capture_app_session_state()
        print(f"Tempo: {state.tempo:6.1f} BPM | "
              f"Peers: {session.num_peers}", end='\r')
        time.sleep(0.5)

    # Stop clock
    clock.stop()
    print("\nMIDI Clock stopped")

# Cleanup
cm.capi.midi_port_dispose(port)
cm.capi.midi_client_dispose(client)
```

### Beat-Accurate MIDI Sequencing

Schedule MIDI events at specific Link beat positions.

```python
import coremusic as cm
from coremusic import link_midi
import time

# Setup MIDI
client = cm.capi.midi_client_create("Sequencer")
port = cm.capi.midi_output_port_create(client, "Seq Out")
destination = cm.capi.midi_get_destination(0)

with cm.link.LinkSession(bpm=120.0) as session:
    # Create sequencer
    seq = link_midi.LinkMIDISequencer(session, port, destination)

    # Schedule a C major arpeggio (one note per beat)
    seq.schedule_note(beat=0.0, channel=0, note=60, velocity=100, duration=0.9)  # C4
    seq.schedule_note(beat=1.0, channel=0, note=64, velocity=100, duration=0.9)  # E4
    seq.schedule_note(beat=2.0, channel=0, note=67, velocity=100, duration=0.9)  # G4
    seq.schedule_note(beat=3.0, channel=0, note=72, velocity=100, duration=0.9)  # C5

    print(f"Scheduled {len(seq.events)} MIDI events")

    # Start sequencer
    seq.start()
    print("Sequencer running...")

    # Monitor playback
    for i in range(20):
        state = session.capture_app_session_state()
        current_time = session.clock.micros()
        beat = state.beat_at_time(current_time, 4.0)

        # Show which beat we're on
        beat_num = int(beat) % 4
        indicators = ["●" if i == beat_num else "○" for i in range(4)]
        print(f"{' '.join(indicators)}  Beat: {beat:7.2f}", end='\r')

        time.sleep(0.5)

    # Stop sequencer
    seq.stop()

cm.capi.midi_port_dispose(port)
cm.capi.midi_client_dispose(client)
```

### MIDI CC Automation Synchronized to Link

```python
import coremusic as cm
from coremusic import link_midi
import time

# Setup MIDI
client = cm.capi.midi_client_create("CC Automation")
port = cm.capi.midi_output_port_create(client, "CC Out")
destination = cm.capi.midi_get_destination(0)

with cm.link.LinkSession(bpm=120.0) as session:
    seq = link_midi.LinkMIDISequencer(session, port, destination)

    # Schedule filter cutoff sweep over 4 beats
    # CC #74 (Filter Cutoff) from 0 to 127
    for beat in range(0, 4):
        for substep in range(8):
            position = beat + (substep / 8.0)
            value = int((position / 4.0) * 127)
            seq.schedule_cc(
                beat=position,
                channel=0,
                controller=74,  # Filter Cutoff
                value=value
            )

    print(f"Scheduled {len(seq.events)} CC events")

    seq.start()
    time.sleep(5)
    seq.stop()

cm.capi.midi_port_dispose(port)
cm.capi.midi_client_dispose(client)
```

### Looping MIDI Patterns

```python
import coremusic as cm
from coremusic import link_midi
import time

client = cm.capi.midi_client_create("Loop Sequencer")
port = cm.capi.midi_output_port_create(client, "Loop Out")
destination = cm.capi.midi_get_destination(0)

with cm.link.LinkSession(bpm=120.0) as session:
    seq = link_midi.LinkMIDISequencer(session, port, destination)

    # Create a 4-beat pattern
    pattern = [
        (0.0, 60, 100),   # Beat 0: C4
        (0.5, 62, 80),    # Beat 0.5: D4
        (1.0, 64, 100),   # Beat 1: E4
        (2.0, 67, 100),   # Beat 2: G4
        (3.0, 65, 100),   # Beat 3: F4
        (3.5, 64, 80),    # Beat 3.5: E4
    ]

    # Schedule pattern for multiple bars
    num_bars = 4
    for bar in range(num_bars):
        for beat, note, velocity in pattern:
            absolute_beat = (bar * 4.0) + beat
            seq.schedule_note(
                beat=absolute_beat,
                channel=0,
                note=note,
                velocity=velocity,
                duration=0.4
            )

    print(f"Scheduled {num_bars} bars of pattern")

    seq.start()
    time.sleep(num_bars * 2)  # 2 seconds per bar at 120 BPM
    seq.stop()

cm.capi.midi_port_dispose(port)
cm.capi.midi_client_dispose(client)
```

### Combined Audio + MIDI Synchronized

```python
import coremusic as cm
from coremusic import link_midi
import time

# Setup MIDI
client = cm.capi.midi_client_create("Audio+MIDI")
port = cm.capi.midi_output_port_create(client, "Out")
destination = cm.capi.midi_get_destination(0)

# Share one Link session for both audio and MIDI
with cm.link.LinkSession(bpm=120.0) as session:
    # Setup audio player
    player = cm.AudioPlayer(link_session=session)
    player.load_file("drums.wav")
    player.setup_output()

    # Setup MIDI sequencer
    seq = link_midi.LinkMIDISequencer(session, port, destination)

    # Schedule bass notes every beat
    for beat in range(16):
        note = 36 if beat % 4 == 0 else 38  # Kick and snare pattern
        seq.schedule_note(
            beat=float(beat),
            channel=9,  # MIDI drum channel
            note=note,
            velocity=100,
            duration=0.9
        )

    # Start both audio and MIDI
    print("Starting synchronized audio + MIDI playback...")

    player.play()
    player.start()
    seq.start()

    # Monitor both
    for i in range(40):
        timing = player.get_link_timing(quantum=4.0)
        progress = player.get_progress()

        print(f"Beat: {timing['beat']:7.2f} | "
              f"Audio: {progress*100:5.1f}% | "
              f"Tempo: {timing['tempo']:6.1f} BPM", end='\r')

        time.sleep(0.5)

    # Stop both
    player.stop()
    seq.stop()

cm.capi.midi_port_dispose(port)
cm.capi.midi_client_dispose(client)
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
- [CoreMusic Examples](../tests/demos/)
- [CoreAudio Documentation](https://developer.apple.com/documentation/coreaudio)
- [CoreMIDI Documentation](https://developer.apple.com/documentation/coremidi)
