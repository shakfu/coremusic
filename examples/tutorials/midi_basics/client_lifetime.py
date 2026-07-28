#!/usr/bin/env python3
"""Keep a MIDI client for as long as the program may need MIDI."""

# --8<-- [start:example]
import time

from coremusic.midi import MIDIClient, note_off, note_on


class Synth:
    """Holds its MIDI client open for the lifetime of the object.

    MIDIServer exits a few seconds after its last client disconnects, and that
    invalidates this process's connection to it for good. Creating a client
    per note - or disposing the last one between pieces of work - risks every
    later `MIDIClient(...)` failing with "Unknown error code -2" until the
    program is restarted.
    """

    def __init__(self, name="My Synth"):
        self.client = MIDIClient(name)
        self.port = self.client.create_output_port("Output")
        self.destination = self.client.create_virtual_destination(f"{name} In")

    def note(self, note, velocity=100, duration=0.2):
        self.port.send_data(self.destination, note_on(note, velocity))
        time.sleep(duration)
        self.port.send_data(self.destination, note_off(note))

    def close(self):
        self.client.dispose()


synth = Synth()
try:
    for note in (60, 64, 67):
        synth.note(note)

    # Idle time is fine: the client is still open, so the server stays up
    time.sleep(0.5)

    synth.note(72)
finally:
    synth.close()
# --8<-- [end:example]
