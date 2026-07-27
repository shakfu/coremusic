#!/usr/bin/env python3
"""Create and dispose a MIDI client, explicitly and with a context manager."""

# --8<-- [start:explicit]
from coremusic.midi import MIDIClient

client = MIDIClient("My Application")

try:
    print(f"Created MIDI client: {client.name}")

finally:
    # Always dispose when done - this also disposes its ports and
    # virtual endpoints
    client.dispose()
# --8<-- [end:explicit]

# --8<-- [start:context-manager]
from coremusic.midi import MIDIClient

with MIDIClient("My Application") as client:
    print(f"MIDI client active: {client.name}")
    # Client is automatically disposed when exiting
# --8<-- [end:context-manager]
