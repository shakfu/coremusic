# MIDI Processing

Recipes for MIDI input/output and processing with CoreMIDI.

## MIDI Device Discovery

### List MIDI Devices

Discover available MIDI sources and destinations:

```python
import coremusic.capi as capi

# List MIDI sources (input devices)
num_sources = capi.midi_get_number_of_sources()
print(f"MIDI Sources: {num_sources}")

for i in range(num_sources):
    source = capi.midi_get_source(i)
    name = capi.midi_object_get_string_property(source, "name")
    print(f"  {i}: {name}")

# List MIDI destinations (output devices)
num_dests = capi.midi_get_number_of_destinations()
print(f"\nMIDI Destinations: {num_dests}")

for i in range(num_dests):
    dest = capi.midi_get_destination(i)
    name = capi.midi_object_get_string_property(dest, "name")
    print(f"  {i}: {name}")
```

### Find Device by Name

Locate a specific MIDI device:

```python
import coremusic.capi as capi

def find_midi_source(device_name):
    """Find MIDI source by name"""
    num_sources = capi.midi_get_number_of_sources()

    for i in range(num_sources):
        source = capi.midi_get_source(i)
        name = capi.midi_object_get_string_property(source, "name")
        if device_name.lower() in name.lower():
            return source, name

    return None, None

# Find a specific device
source, name = find_midi_source("Keyboard")
if source:
    print(f"Found: {name}")
else:
    print("Device not found")
```

## MIDI Input

### How Incoming Packets Arrive

CoreMIDI delivers incoming data as *packets* on its own high-priority receive
thread. A packet is not the same thing as a MIDI message:

- One packet may contain several complete messages that arrived together.
- A large system-exclusive dump is spread across consecutive packets.
- Real-time bytes such as clock may be interleaved inside another message.

`MIDIMessageSplitter` turns a stream of packet payloads back into individual
messages, holding the state needed to span packets. Timestamps are host times
(mach absolute ticks); convert them with `capi.midi_host_time_to_seconds`.

An input port buffers packets by default, so the owning thread controls when
Python code runs. Pass a callback instead only when you need the lowest
possible latency and can guarantee the callback never blocks.

### Receive MIDI Messages

Poll the port from your own loop:

```python
import coremusic.capi as capi
from coremusic.midi import MIDIMessageSplitter

client = capi.midi_client_create("MIDI Input")
input_port = capi.midi_input_port_create(client, "Input")

source = capi.midi_get_source(0)
capi.midi_port_connect_source(input_port, source)

splitter = MIDIMessageSplitter()

print("Listening for MIDI... (Press Ctrl+C to stop)")
try:
    while True:
        # Blocks until a packet arrives or the timeout expires.
        if not capi.midi_input_wait(input_port, 0.1):
            continue

        for host_time, payload in capi.midi_input_poll(input_port):
            seconds = capi.midi_host_time_to_seconds(host_time)

            for data in splitter.push(payload):
                status = data[0]
                message_type = status & 0xF0
                channel = status & 0x0F

                if message_type == 0x90 and data[2] > 0:  # Note On
                    print(f"Note On: ch={channel}, note={data[1]}, vel={data[2]}")
                elif message_type == 0x80 or message_type == 0x90:  # Note Off
                    print(f"Note Off: ch={channel}, note={data[1]}")
                elif message_type == 0xB0:  # Control Change
                    print(f"CC: ch={channel}, ctrl={data[1]}, val={data[2]}")
except KeyboardInterrupt:
    print("\nStopped")

# Cleanup
capi.midi_port_disconnect_source(input_port, source)
capi.midi_port_dispose(input_port)
capi.midi_client_dispose(client)
```

If the loop cannot keep up, the oldest packets are discarded once the buffer is
full. Check `capi.midi_input_dropped(input_port)` to detect this, and raise
`queue_size` on `midi_input_port_create` if it happens.

### Receive with a Callback

A callback runs on the CoreMIDI receive thread, so it must be short and must
not block, allocate heavily, or acquire locks held by slow code. Exceptions
raised inside it are swallowed rather than propagated into the framework.

```python
import coremusic.capi as capi
import queue

incoming: queue.SimpleQueue = queue.SimpleQueue()

def midi_callback(data: bytes, host_time: int) -> None:
    # Hand off immediately; do the real work on your own thread.
    incoming.put((host_time, data))

client = capi.midi_client_create("MIDI Input")
input_port = capi.midi_input_port_create(client, "Input", midi_callback)
capi.midi_port_connect_source(input_port, capi.midi_get_source(0))
```

### Filter MIDI Messages

Filter specific MIDI message types:

```python
import coremusic.capi as capi
from coremusic.midi import MIDIMessageSplitter

class MIDIFilter:
    def __init__(self, filter_notes=False, filter_cc=False):
        self.filter_notes = filter_notes
        self.filter_cc = filter_cc
        self.splitter = MIDIMessageSplitter()

    def process(self, payload: bytes) -> None:
        for data in self.splitter.push(payload):
            message_type = data[0] & 0xF0

            # Filter note messages
            if message_type in (0x80, 0x90) and self.filter_notes:
                continue

            # Filter CC messages
            if message_type == 0xB0 and self.filter_cc:
                continue

            print(f"MIDI: {[hex(b) for b in data]}")

# Create filter that blocks notes but allows CC
midi_filter = MIDIFilter(filter_notes=True, filter_cc=False)

client = capi.midi_client_create("Filtered Input")
input_port = capi.midi_input_port_create(client, "Input")

# Connect, then feed polled packets to midi_filter.process()...
```

## MIDI Output

### Send MIDI Messages

Send MIDI messages to an output device:

```python
import coremusic.capi as capi
import time

# Create MIDI client and output port
client = capi.midi_client_create("MIDI Output")
output_port = capi.midi_output_port_create(client, "Output")

# Get first MIDI destination
dest = capi.midi_get_destination(0)

# Send Note On
note_on = bytes([0x90, 60, 100])  # Channel 1, Middle C, Velocity 100
capi.midi_send_data(output_port, dest, note_on)
print("Sent Note On")

time.sleep(1.0)

# Send Note Off
note_off = bytes([0x80, 60, 0])  # Channel 1, Middle C
capi.midi_send_data(output_port, dest, note_off)
print("Sent Note Off")

# Cleanup
capi.midi_port_dispose(output_port)
capi.midi_client_dispose(client)
```

### Play MIDI Sequence

Send a sequence of MIDI notes:

```python
import coremusic.capi as capi
import time

def play_note(port, dest, channel, note, velocity, duration):
    """Play a single note"""
    # Note On
    note_on = bytes([0x90 | channel, note, velocity])
    capi.midi_send_data(port, dest, note_on)

    # Wait
    time.sleep(duration)

    # Note Off
    note_off = bytes([0x80 | channel, note, 0])
    capi.midi_send_data(port, dest, note_off)

# Setup
client = capi.midi_client_create("Sequencer")
output_port = capi.midi_output_port_create(client, "Output")
dest = capi.midi_get_destination(0)

# Play C major scale
scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C D E F G A B C

for note in scale:
    play_note(output_port, dest, channel=0, note=note, velocity=100, duration=0.5)
    time.sleep(0.1)  # Gap between notes

# Cleanup
capi.midi_port_dispose(output_port)
capi.midi_client_dispose(client)
```

### Send Control Changes

Send MIDI CC messages for automation:

```python
import coremusic.capi as capi
import time

client = capi.midi_client_create("CC Controller")
output_port = capi.midi_output_port_create(client, "Output")
dest = capi.midi_get_destination(0)

# Start a note
note_on = bytes([0x90, 60, 100])
capi.midi_send_data(output_port, dest, note_on)

# Fade volume (CC 7) from 127 to 0
for volume in range(127, -1, -5):
    cc = bytes([0xB0, 7, volume])  # Channel 1, CC 7 (Volume), value
    capi.midi_send_data(output_port, dest, cc)
    time.sleep(0.05)

# Stop note
note_off = bytes([0x80, 60, 0])
capi.midi_send_data(output_port, dest, note_off)

# Cleanup
capi.midi_port_dispose(output_port)
capi.midi_client_dispose(client)
```

## MIDI Routing

### MIDI Thru

Route MIDI input directly to output:

```python
import coremusic.capi as capi

# Create client with input and output ports
client = capi.midi_client_create("MIDI Thru")

# Output port
output_port = capi.midi_output_port_create(client, "Output")
dest = capi.midi_get_destination(0)

# Input port
input_port = capi.midi_input_port_create(client, "Input")
source = capi.midi_get_source(0)
capi.midi_port_connect_source(input_port, source)

print("MIDI thru active... (Press Ctrl+C to stop)")
try:
    while True:
        if not capi.midi_input_wait(input_port, 0.1):
            continue

        # Packets can be forwarded verbatim; no need to split them first.
        for _host_time, payload in capi.midi_input_poll(input_port):
            capi.midi_send_data(output_port, dest, payload)
except KeyboardInterrupt:
    print("\nStopped")

# Cleanup
capi.midi_port_disconnect_source(input_port, source)
capi.midi_port_dispose(input_port)
capi.midi_port_dispose(output_port)
capi.midi_client_dispose(client)
```

### Channel Routing

Route MIDI from one channel to another:

```python
import coremusic.capi as capi
from coremusic.midi import MIDIMessageSplitter

class ChannelRouter:
    def __init__(self, output_port, dest, input_channel, output_channel):
        self.output_port = output_port
        self.dest = dest
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.splitter = MIDIMessageSplitter()

    def process(self, payload: bytes) -> None:
        for message in self.splitter.push(payload):
            status = message[0]

            # Channel voice messages only; system messages have no channel.
            if not 0x80 <= status <= 0xEF:
                continue

            if (status & 0x0F) == self.input_channel:
                data = bytearray(message)
                data[0] = (status & 0xF0) | self.output_channel
                capi.midi_send_data(self.output_port, self.dest, bytes(data))

# Route channel 0 -> channel 1
client = capi.midi_client_create("Channel Router")
output_port = capi.midi_output_port_create(client, "Output")
dest = capi.midi_get_destination(0)

router = ChannelRouter(output_port, dest, input_channel=0, output_channel=1)

input_port = capi.midi_input_port_create(client, "Input")
source = capi.midi_get_source(0)
capi.midi_port_connect_source(input_port, source)

while True:
    if capi.midi_input_wait(input_port, 0.1):
        for _host_time, payload in capi.midi_input_poll(input_port):
            router.process(payload)
```

## MIDI Transformation

### Transpose Notes

Transpose all incoming notes:

```python
import coremusic.capi as capi
from coremusic.midi import MIDIMessageSplitter

class Transposer:
    def __init__(self, output_port, dest, semitones):
        self.output_port = output_port
        self.dest = dest
        self.semitones = semitones
        self.splitter = MIDIMessageSplitter()

    def process(self, payload: bytes) -> None:
        for message in self.splitter.push(payload):
            data = bytearray(message)
            message_type = data[0] & 0xF0

            # Transpose note on/off messages
            if message_type in (0x80, 0x90) and len(data) >= 3:
                original_note = data[1]
                transposed_note = max(0, min(127, original_note + self.semitones))
                data[1] = transposed_note

                print(f"Transposed: {original_note} -> {transposed_note}")

            # Forward the (possibly modified) message
            capi.midi_send_data(self.output_port, self.dest, bytes(data))

# Transpose up one octave
client = capi.midi_client_create("Transposer")
output_port = capi.midi_output_port_create(client, "Output")
dest = capi.midi_get_destination(0)

transposer = Transposer(output_port, dest, semitones=12)

input_port = capi.midi_input_port_create(client, "Input")
source = capi.midi_get_source(0)
capi.midi_port_connect_source(input_port, source)

while True:
    if capi.midi_input_wait(input_port, 0.1):
        for _host_time, payload in capi.midi_input_poll(input_port):
            transposer.process(payload)
```

### Velocity Scaling

Scale note velocities:

```python
import coremusic.capi as capi
from coremusic.midi import MIDIMessageSplitter

class VelocityScaler:
    def __init__(self, output_port, dest, scale_factor):
        self.output_port = output_port
        self.dest = dest
        self.scale_factor = scale_factor
        self.splitter = MIDIMessageSplitter()

    def process(self, payload: bytes) -> None:
        for message in self.splitter.push(payload):
            data = bytearray(message)

            # Note On with a non-zero velocity; a zero velocity is a Note Off
            # and must keep its value.
            if (data[0] & 0xF0) == 0x90 and len(data) >= 3 and data[2] > 0:
                original_vel = data[2]
                scaled_vel = int(original_vel * self.scale_factor)
                data[2] = max(1, min(127, scaled_vel))  # Clamp to 1-127

                print(f"Velocity: {original_vel} -> {data[2]}")

            capi.midi_send_data(self.output_port, self.dest, bytes(data))

# Scale velocities to 80% (softer)
scaler = VelocityScaler(output_port, dest, scale_factor=0.8)

# Setup and run as in the Transpose Notes example...
```

## MIDI Recording

### Record MIDI Messages

Record MIDI to a list with timestamps:

Message times come from the packet host timestamp rather than the wall clock,
so they stay accurate even if the polling loop is briefly delayed.

```python
import coremusic.capi as capi
import json
import time
from coremusic.midi import MIDIMessageSplitter

class MIDIRecorder:
    def __init__(self, port_id):
        self.port_id = port_id
        self.splitter = MIDIMessageSplitter()
        self.origin = None
        self.recorded_messages = []

    def record(self, duration):
        print("Recording started")
        deadline = time.monotonic() + duration

        while time.monotonic() < deadline:
            if not capi.midi_input_wait(self.port_id, 0.1):
                continue

            for host_time, payload in capi.midi_input_poll(self.port_id):
                # A zero timestamp means "as soon as possible".
                if host_time == 0:
                    host_time = capi.midi_current_host_time()
                seconds = capi.midi_host_time_to_seconds(host_time)
                if self.origin is None:
                    self.origin = seconds

                for data in self.splitter.push(payload):
                    self.recorded_messages.append({
                        'time': seconds - self.origin,
                        'data': data,
                    })

        dropped = capi.midi_input_dropped(self.port_id)
        if dropped:
            print(f"Warning: dropped {dropped} packets")
        print(f"Recording stopped: {len(self.recorded_messages)} messages")

    def save(self, filename):
        """Save recorded messages to file"""
        with open(filename, 'w') as f:
            messages = [
                {'time': msg['time'], 'data': list(msg['data'])}
                for msg in self.recorded_messages
            ]
            json.dump(messages, f, indent=2)

        print(f"Saved to {filename}")

# Setup recorder
client = capi.midi_client_create("Recorder")
input_port = capi.midi_input_port_create(client, "Input")
source = capi.midi_get_source(0)
capi.midi_port_connect_source(input_port, source)

recorder = MIDIRecorder(input_port)
recorder.record(duration=10)
recorder.save("recorded_midi.json")

# Cleanup
capi.midi_port_disconnect_source(input_port, source)
capi.midi_port_dispose(input_port)
capi.midi_client_dispose(client)
```

### Playback Recorded MIDI

Play back recorded MIDI messages:

```python
import coremusic.capi as capi
import json
import time

def playback_midi(filename, output_port, dest):
    """Play back recorded MIDI"""
    # Load recorded messages
    with open(filename, 'r') as f:
        messages = json.load(f)

    if not messages:
        print("No messages to play")
        return

    print(f"Playing back {len(messages)} messages...")
    start_time = time.time()

    for msg in messages:
        # Wait until scheduled time
        target_time = start_time + msg['time']
        wait_time = target_time - time.time()

        if wait_time > 0:
            time.sleep(wait_time)

        # Send message
        data = bytes(msg['data'])
        capi.midi_send_data(output_port, dest, data)

    print("Playback complete")

# Setup playback
client = capi.midi_client_create("Playback")
output_port = capi.midi_output_port_create(client, "Output")
dest = capi.midi_get_destination(0)

# Play recording
playback_midi("recorded_midi.json", output_port, dest)

# Cleanup
capi.midi_port_dispose(output_port)
capi.midi_client_dispose(client)
```

## Complete Example: MIDI Monitor

Full-featured MIDI monitor with message parsing:

```python
import coremusic.capi as capi
from coremusic.midi import MIDIMessageSplitter

class MIDIMonitor:
    def __init__(self):
        self.message_count = 0
        self.splitter = MIDIMessageSplitter()

    def parse_message(self, data):
        """Parse and format MIDI message"""
        if len(data) == 0:
            return "Empty message"

        status = data[0]
        message_type = status & 0xF0
        channel = (status & 0x0F) + 1

        if message_type == 0x80:  # Note Off
            return f"Note Off  | Ch {channel:2d} | Note {data[1]:3d} | Vel {data[2]:3d}"
        elif message_type == 0x90:  # Note On
            if data[2] == 0:  # Velocity 0 = Note Off
                return f"Note Off  | Ch {channel:2d} | Note {data[1]:3d} | Vel {data[2]:3d}"
            return f"Note On   | Ch {channel:2d} | Note {data[1]:3d} | Vel {data[2]:3d}"
        elif message_type == 0xA0:  # Poly Aftertouch
            return f"Poly AT   | Ch {channel:2d} | Note {data[1]:3d} | Pressure {data[2]:3d}"
        elif message_type == 0xB0:  # Control Change
            return f"CC        | Ch {channel:2d} | Ctrl {data[1]:3d} | Val {data[2]:3d}"
        elif message_type == 0xC0:  # Program Change
            return f"Program   | Ch {channel:2d} | Program {data[1]:3d}"
        elif message_type == 0xD0:  # Channel Aftertouch
            return f"Channel AT| Ch {channel:2d} | Pressure {data[1]:3d}"
        elif message_type == 0xE0:  # Pitch Bend
            value = data[1] + (data[2] << 7)
            return f"Pitch Bend| Ch {channel:2d} | Value {value:5d}"
        elif status == 0xF8:  # Clock
            return "MIDI Clock"
        elif status == 0xFA:  # Start
            return "MIDI Start"
        elif status == 0xFB:  # Continue
            return "MIDI Continue"
        elif status == 0xFC:  # Stop
            return "MIDI Stop"
        else:
            hex_data = ' '.join(f'{b:02X}' for b in data)
            return f"Unknown   | {hex_data}"

    def process(self, payload):
        for data in self.splitter.push(payload):
            self.message_count += 1
            message = self.parse_message(data)
            print(f"[{self.message_count:6d}] {message}")

# Setup monitor
monitor = MIDIMonitor()

client = capi.midi_client_create("MIDI Monitor")
input_port = capi.midi_input_port_create(client, "Monitor Input")

# Connect to all MIDI sources
num_sources = capi.midi_get_number_of_sources()
print(f"Monitoring {num_sources} MIDI source(s)\n")

for i in range(num_sources):
    source = capi.midi_get_source(i)
    capi.midi_port_connect_source(input_port, source)
    name = capi.midi_object_get_string_property(source, "name")
    print(f"Connected to: {name}")

print("\nMIDI Monitor - Press Ctrl+C to stop")
print("-" * 70)

try:
    while True:
        if not capi.midi_input_wait(input_port, 0.1):
            continue
        for _host_time, payload in capi.midi_input_poll(input_port):
            monitor.process(payload)
except KeyboardInterrupt:
    print(f"\n\nStopped - Received {monitor.message_count} messages")

# Cleanup
for i in range(num_sources):
    source = capi.midi_get_source(i)
    capi.midi_port_disconnect_source(input_port, source)

capi.midi_port_dispose(input_port)
capi.midi_client_dispose(client)
```

## Best Practices

### Resource Management

Always dispose of MIDI resources:

```python
# Create resources
client = capi.midi_client_create("App")
port = capi.midi_output_port_create(client, "Out")

try:
    # Use resources
    pass
finally:
    # Always cleanup
    capi.midi_port_dispose(port)
    capi.midi_client_dispose(client)
```

### Error Handling

Handle MIDI errors gracefully:

```python
try:
    dest = capi.midi_get_destination(0)
except IndexError:
    print("No MIDI destinations available")
    return

try:
    capi.midi_send_data(port, dest, data)
except Exception as e:
    print(f"Failed to send MIDI: {e}")
```

### Timing Precision

MIDI timestamps are host times: mach absolute ticks on the same monotonic clock
CoreAudio uses. Schedule ahead of the current time rather than sending at the
moment the note should sound, so the CoreMIDI server absorbs the jitter of your
own loop. A timestamp of 0 means "as soon as possible".

```python
# Schedule a note 50 ms from now
when = capi.midi_current_host_time() + capi.midi_seconds_to_host_time(0.05)
capi.midi_send_data(port, dest, note_on, when)

# Convert an incoming packet timestamp back to seconds
for host_time, payload in capi.midi_input_poll(input_port):
    seconds = capi.midi_host_time_to_seconds(host_time)
```

### Thread Safety

CoreMIDI receives on its own thread. Polling with `midi_input_poll` keeps every
Python object you touch on your own thread, which is the simpler and safer
default.

A callback passed to `midi_input_port_create` runs on the CoreMIDI receive
thread instead. Keep it to a hand-off, and do the real work elsewhere:

```python
import queue

class MIDIProcessor:
    def __init__(self):
        self.incoming: queue.SimpleQueue = queue.SimpleQueue()

    def callback(self, data: bytes, host_time: int) -> None:
        # Runs on the CoreMIDI thread: hand off and return immediately.
        self.incoming.put((host_time, data))

    def run(self) -> None:
        # Runs on your own thread.
        while True:
            host_time, data = self.incoming.get()
            ...
```

## See Also

- [API Reference](../api/index.md) - Complete API reference
- [AudioUnit Hosting](audiounit_hosting.md) - AudioUnit plugin hosting (instruments)
- [Link Integration](link_integration.md) - Ableton Link tempo sync
- CoreMIDI documentation: https://developer.apple.com/documentation/coremidi
