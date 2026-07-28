#!/usr/bin/env python3
"""Velocity Scaling."""

from coremusic import capi

client = capi.midi_client_create("Velocity Scaler")
output_port = capi.midi_output_port_create(client, "Output")
dest = capi.midi_destination_create(client, "Velocity Scaler Out")


# --8<-- [start:example]
from coremusic import capi
from coremusic.constants import MIDIStatus
from coremusic.midi import MIDIMessageSplitter, note_on


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
            if (
                (data[0] & 0xF0) == MIDIStatus.NOTE_ON
                and len(data) >= 3
                and data[2] > 0
            ):
                original_vel = data[2]
                scaled_vel = int(original_vel * self.scale_factor)
                data[2] = max(1, min(127, scaled_vel))  # Clamp to 1-127

                print(f"Velocity: {original_vel} -> {data[2]}")

            capi.midi_send_data(self.output_port, self.dest, bytes(data))


# Scale velocities to 80% (softer)
scaler = VelocityScaler(output_port, dest, scale_factor=0.8)

# Feed it as in the Transpose Notes example; here is one message by hand
scaler.process(note_on("C4", 100))
# --8<-- [end:example]

capi.midi_client_dispose(client)
