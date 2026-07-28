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

### How Long to Keep a Client

Keep the client for as long as your program may need MIDI, rather than
creating one per operation:

```python
--8<-- "examples/tutorials/midi_basics/client_lifetime.py:example"
```

This matters more than it looks. `MIDIServer` is an on-demand system daemon:
it exits a few seconds after its last client disconnects, and that invalidates
this process's connection to CoreMIDI. The framework does not re-establish it,
so once it happens, every later `MIDIClient(...)` in the same process fails
with `MIDIClientCreate failed: Unknown error code -2` no matter how long you
wait. Only restarting the process clears it.

A program that disposes its last client between pieces of work - a tool that
idles between MIDI sessions, say - can therefore work perfectly on the first
run through and fail on the second. Holding one client open avoids it
entirely; the client costs nothing while idle and publishes no endpoints of
its own.

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

Build messages with the functions in `coremusic.midi` rather than assembling
bytes by hand. Each returns the `bytes` that `send_data` takes, validates its
arguments, and gets the awkward parts right - the two-byte messages and the
pitch bend split.

### Note Messages

```python
--8<-- "examples/tutorials/midi_basics/message_reference.py:notes"
```

Channel is keyword-only. `capi.midi_note_on` takes `(channel, note, velocity)`
and returns a tuple, so allowing a positional channel here would make
`note_on(0, 60, 100)` build a valid but completely different message. It raises
`TypeError` instead.

Octave numbering is scientific pitch notation, matching `note_name_to_midi`:
middle C is `"C4"` is 60. Ableton Live and Logic display that note as C3 and
Cakewalk as C5, so prefer the MIDI number when matching a DAW display.

### Control Change

```python
--8<-- "examples/tutorials/midi_basics/message_reference.py:control-change"
```

### Program Change

```python
--8<-- "examples/tutorials/midi_basics/message_reference.py:program-change"
```

### Pitch Bend

```python
--8<-- "examples/tutorials/midi_basics/message_reference.py:pitch-bend"
```

### Aftertouch

```python
--8<-- "examples/tutorials/midi_basics/message_reference.py:aftertouch"
```

### Validation

```python
--8<-- "examples/tutorials/midi_basics/message_reference.py:validation"
```

Note that `MIDIEvent.to_bytes()` masks instead of raising, so a velocity of 200
silently becomes 72 there.

### Not Interchangeable With `capi.midi_*`

The `capi.midi_note_on` family looks similar but serves a different target: it
returns a fixed `(status, data1, data2)` triple for
`capi.music_device_midi_event()`, the AudioUnit MusicDevice call, whose `data2`
is "0 if not needed". Program Change and Channel Aftertouch are two bytes on
the wire, so `bytes(capi.midi_program_change(...))` appends a `0x00` that a
receiver reads as data for a running-status message. Use the builders above for
anything sent through CoreMIDI.

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

### `MIDIClientCreate failed: Unknown error code -2`

The process has lost its connection to `MIDIServer`, which exits a few seconds
after its last client disconnects. CoreMIDI does not reconnect, so every
subsequent client creation in that process fails the same way and no amount of
retrying helps.

Restart the process to recover, and to prevent it, keep one client open for as
long as MIDI might be needed - see [How Long to Keep a
Client](#how-long-to-keep-a-client). Status `-304` has the same cause.

## Next Steps

- [MIDI Transform](midi_transform.md) - Transform and process MIDI
- [MIDI Processing Cookbook](../cookbook/midi_processing.md) - MIDI processing recipes
- [Link Integration Cookbook](../cookbook/link_integration.md) - Sync MIDI with Ableton Link

## See Also

- [API Reference](../api/index.md) - Complete API reference
- [CLI Guide](../guides/cli.md) - CLI MIDI commands
