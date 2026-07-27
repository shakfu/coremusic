# MIDI Basics

This tutorial covers MIDI fundamentals with coremusic, including sending, receiving, and processing MIDI messages.

Every example on this page is a runnable program under
[`examples/tutorials/midi_basics/`](https://github.com/shakfu/coremusic/tree/main/examples/tutorials/midi_basics).

## Prerequisites

- coremusic installed and built
- Basic Python knowledge
- Optional: A MIDI controller or virtual MIDI device

## Understanding MIDI

MIDI (Musical Instrument Digital Interface) is a protocol for communicating musical information:

- **Note On/Off**: When keys are pressed/released
- **Control Change (CC)**: Knobs, sliders, pedals
- **Program Change**: Patch/preset selection
- **Pitch Bend**: Pitch wheel position
- **Aftertouch**: Pressure after key press

## Endpoints, Ports, and Clients

Three object types cover everything in this tutorial:

- **`MIDIClient`** owns everything else. Create one per application.
- **`MIDIPort`** is this application's connection to the MIDI system. An output
  port sends, an input port receives.
- **`MIDIEndpoint`** is the other end of a connection: a **source** produces
  MIDI (a keyboard), a **destination** consumes it (a synth). Endpoints are
  published system-wide by whichever process owns them, so the endpoints you
  send to normally belong to some other application or device.

Sending means: output port, plus a destination endpoint to aim at.

## MIDI Devices

### Listing Devices

```python
--8<-- "examples/tutorials/midi_basics/list_devices.py:example"
```

If both lists are empty, no MIDI hardware or software is publishing endpoints.
You can still follow along by creating a virtual endpoint of your own - see
[Virtual Endpoints](#virtual-endpoints) below.

### Using the CLI

```bash
# List MIDI devices, inputs, and outputs
coremusic midi list
```

## Creating a MIDI Client

All MIDI operations require a client:

```python
--8<-- "examples/tutorials/midi_basics/create_client.py:explicit"
```

Or use the context manager:

```python
--8<-- "examples/tutorials/midi_basics/create_client.py:context-manager"
```

## Sending MIDI Messages

### Creating an Output Port

```python
--8<-- "examples/tutorials/midi_basics/output_port.py:port"
```

### Choosing a Destination

`send_data` needs a destination endpoint. Pick one by index or by name:

```python
--8<-- "examples/tutorials/midi_basics/output_port.py:destination"
```

### Sending Note Messages

```python
--8<-- "examples/tutorials/midi_basics/send_note.py:example"
```

### Sending Control Change

```python
--8<-- "examples/tutorials/midi_basics/send_cc.py:example"
```

### Playing a Melody

```python
--8<-- "examples/tutorials/midi_basics/play_melody.py:example"
```

## Receiving MIDI Messages

### Polling an Input Port

An input port created without a callback buffers incoming packets. Drain them
with `poll()`, optionally blocking on `wait()` first:

```python
--8<-- "examples/tutorials/midi_basics/receive_poll.py:example"
```

`poll()` returns `(host_time, data)` tuples. Convert `host_time` to seconds with
`capi.midi_host_time_to_seconds()`.

### Using a Callback

Pass a callback to receive packets as they arrive instead. The callback runs on
the CoreMIDI receive thread, so it must be short and must not block:

```python
--8<-- "examples/tutorials/midi_basics/receive_callback.py:example"
```

### Splitting Packets Into Messages

One packet may hold several MIDI messages, and a SysEx message may span
packets. `MIDIMessageSplitter` keeps the state needed to separate them:

```python
--8<-- "examples/tutorials/midi_basics/split_packets.py:example"
```

Use one splitter per source; it carries running-status and SysEx state across
packets.

### Simple MIDI Monitor

```python
--8<-- "examples/tutorials/midi_basics/midi_monitor.py:example"
```

### Using the CLI

```bash
# Monitor MIDI input
coremusic midi monitor

# Display incoming MIDI as raw events
coremusic midi receive
```

## Virtual Endpoints

A client can publish its own endpoints, which other applications then see in
their MIDI device lists. This is also the easiest way to test send and receive
code without any hardware.

A **virtual destination** receives what other applications send you:

```python
--8<-- "examples/tutorials/midi_basics/virtual_endpoints.py:destination"
```

Like an input port, a virtual destination accepts a callback instead:

```python
--8<-- "examples/tutorials/midi_basics/virtual_endpoints.py:destination-callback"
```

A **virtual source** produces MIDI that other applications can subscribe to:

```python
--8<-- "examples/tutorials/midi_basics/virtual_endpoints.py:source"
```

Both are disposed with the client. Endpoints returned by `get_sources()` and
`get_destinations()` belong to other processes, so disposing those wrappers
leaves the underlying endpoint alone.

### Loopback Test

Putting both halves together gives a self-contained round trip:

```python
--8<-- "examples/tutorials/midi_basics/loopback.py:example"
```

## MIDI Message Reference

### Note Messages

```python
# Note On: 0x90 + channel, note, velocity
note_on = bytes([0x90, 60, 100])   # Middle C, velocity 100

# Note Off: 0x80 + channel, note, velocity
note_off = bytes([0x80, 60, 0])    # Middle C off

# Note numbers: 0-127
# Middle C (C4) = 60
# A440 = 69
```

### Control Change

```python
# CC: 0xB0 + channel, controller, value
modulation = bytes([0xB0, 1, 64])   # Mod wheel to 50%
volume = bytes([0xB0, 7, 100])      # Volume to 100
pan = bytes([0xB0, 10, 64])         # Pan center
sustain_on = bytes([0xB0, 64, 127]) # Sustain on
sustain_off = bytes([0xB0, 64, 0])  # Sustain off
all_off = bytes([0xB0, 123, 0])     # All Notes Off
```

### Program Change

```python
# Program Change: 0xC0 + channel, program
piano = bytes([0xC0, 0])     # Program 0 (Piano)
strings = bytes([0xC0, 48])  # Program 48 (Strings)
```

### Pitch Bend

```python
# Pitch Bend: 0xE0 + channel, LSB, MSB
# Value range: 0-16383, center = 8192

center = 8192
bend_up = bytes([0xE0, center & 0x7F, (center >> 7) & 0x7F])

max_up = 16383
bend_max = bytes([0xE0, max_up & 0x7F, (max_up >> 7) & 0x7F])
```

## Complete Example: MIDI Keyboard

A simple MIDI keyboard using computer keys:

```python
--8<-- "examples/tutorials/midi_basics/midi_keyboard.py:example"
```

## Troubleshooting

### No MIDI Devices Found

1. Check Audio MIDI Setup.app for device visibility
2. Enable the IAC Driver in Audio MIDI Setup to get a software loopback bus
3. Ensure MIDI devices are connected and powered on
4. Try unplugging and reconnecting USB MIDI devices
5. Check for driver requirements

### Messages Not Received

1. Verify source is connected to input port
2. Check device is sending on expected channel
3. Check `input_port.dropped` - a non-zero value means the port is not being
   polled fast enough
4. Use MIDI Monitor to verify messages

### Messages Not Sending

1. Verify a destination exists - `get_destinations()` returning an empty list
   is the most common cause
2. Check receiving device/software is listening
3. Try sending to a different destination

## Next Steps

- [MIDI Transform](midi_transform.md) - Transform and process MIDI
- [MIDI Processing Cookbook](../cookbook/midi_processing.md) - MIDI processing recipes
- [Link Integration Cookbook](../cookbook/link_integration.md) - Sync MIDI with Ableton Link

## See Also

- [API Reference](../api/index.md) - Complete API reference
- [CLI Guide](../guides/cli.md) - CLI MIDI commands
