# MIDI Processing

Recipes for MIDI input/output and processing with CoreMIDI.

## MIDI Device Discovery

### List MIDI Devices

Discover available MIDI sources and destinations:

```python
--8<-- "examples/cookbook/midi_processing/list_devices.py:example"
```

### Find Device by Name

Locate a specific MIDI device:

```python
--8<-- "examples/cookbook/midi_processing/find_device.py:example"
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
--8<-- "examples/cookbook/midi_processing/receive_poll.py:example"
```

If the loop cannot keep up, the oldest packets are discarded once the buffer is
full. Check `capi.midi_input_dropped(input_port)` to detect this, and raise
`queue_size` on `midi_input_port_create` if it happens.

### Receive with a Callback

A callback runs on the CoreMIDI receive thread, so it must be short and must
not block, allocate heavily, or acquire locks held by slow code. Exceptions
raised inside it are swallowed rather than propagated into the framework.

```python
--8<-- "examples/cookbook/midi_processing/receive_callback.py:example"
```

### Filter MIDI Messages

Filter specific MIDI message types:

```python
--8<-- "examples/cookbook/midi_processing/filter_messages.py:example"
```

## MIDI Output

### Send MIDI Messages

Send MIDI messages to an output device:

```python
--8<-- "examples/cookbook/midi_processing/send_messages.py:example"
```

### Play MIDI Sequence

Send a sequence of MIDI notes:

```python
--8<-- "examples/cookbook/midi_processing/play_sequence.py:example"
```

### Send Control Changes

Send MIDI CC messages for automation:

```python
--8<-- "examples/cookbook/midi_processing/control_changes.py:example"
```

## MIDI Routing

### MIDI Thru

Route MIDI input directly to output:

```python
--8<-- "examples/cookbook/midi_processing/midi_thru.py:example"
```

### Channel Routing

Route MIDI from one channel to another:

```python
--8<-- "examples/cookbook/midi_processing/channel_routing.py:example"
```

## MIDI Transformation

### Transpose Notes

Transpose all incoming notes:

```python
--8<-- "examples/cookbook/midi_processing/transpose.py:example"
```

### Velocity Scaling

Scale note velocities:

```python
--8<-- "examples/cookbook/midi_processing/velocity_scaling.py:example"
```

## MIDI Recording

### Record MIDI Messages

Record MIDI to a list with timestamps:

Message times come from the packet host timestamp rather than the wall clock,
so they stay accurate even if the polling loop is briefly delayed.

```python
--8<-- "examples/cookbook/midi_processing/record.py:example"
```

### Playback Recorded MIDI

Play back recorded MIDI messages:

```python
--8<-- "examples/cookbook/midi_processing/playback.py:example"
```

## Complete Example: MIDI Monitor

Full-featured MIDI monitor with message parsing:

```python
--8<-- "examples/cookbook/midi_processing/monitor.py:example"
```

## Best Practices

### Resource Management

Always dispose of MIDI resources:

```python
--8<-- "examples/cookbook/midi_processing/resource_management.py:example"
```

### Error Handling

Handle MIDI errors gracefully:

```python
--8<-- "examples/cookbook/midi_processing/error_handling.py:example"
```

### Timing Precision

MIDI timestamps are host times: mach absolute ticks on the same monotonic clock
CoreAudio uses. Schedule ahead of the current time rather than sending at the
moment the note should sound, so the CoreMIDI server absorbs the jitter of your
own loop. A timestamp of 0 means "as soon as possible".

```python
--8<-- "examples/cookbook/midi_processing/timing.py:example"
```

### Thread Safety

CoreMIDI receives on its own thread. Polling with `midi_input_poll` keeps every
Python object you touch on your own thread, which is the simpler and safer
default.

A callback passed to `midi_input_port_create` runs on the CoreMIDI receive
thread instead. Keep it to a hand-off, and do the real work elsewhere:

```python
--8<-- "examples/cookbook/midi_processing/thread_safety.py:example"
```

## See Also

- [API Reference](../api/index.md) - Complete API reference
- [AudioUnit Hosting](audiounit_hosting.md) - AudioUnit plugin hosting (instruments)
- [Link Integration](link_integration.md) - Ableton Link tempo sync
- CoreMIDI documentation: https://developer.apple.com/documentation/coremidi
