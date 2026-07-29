#!/usr/bin/env python3
"""Record."""

# --8<-- [start:example]
import json
import time

from coremusic import capi
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
                    self.recorded_messages.append(
                        {
                            "time": seconds - self.origin,
                            "data": data,
                        }
                    )

        dropped = capi.midi_input_dropped(self.port_id)
        if dropped:
            print(f"Warning: dropped {dropped} packets")
        print(f"Recording stopped: {len(self.recorded_messages)} messages")

    def save(self, filename):
        """Save recorded messages to file"""
        with open(filename, "w") as f:
            messages = [
                {"time": msg["time"], "data": list(msg["data"])}
                for msg in self.recorded_messages
            ]
            json.dump(messages, f, indent=2)

        print(f"Saved to {filename}")


# Setup recorder
client = capi.midi_client_create("Recorder")
input_port = capi.midi_input_port_create(client, "Input")
# Connect to the first source, or to one we publish ourselves when nothing
# else is connected
if capi.midi_get_number_of_sources() > 0:
    source = capi.midi_get_source(0)
else:
    source = capi.midi_source_create(client, "Cookbook Input")
capi.midi_port_connect_source(input_port, source)

recorder = MIDIRecorder(input_port)
recorder.record(duration=10)
recorder.save("recorded_midi.json")

# Cleanup
capi.midi_port_disconnect_source(input_port, source)
capi.midi_port_dispose(input_port)
capi.midi_client_dispose(client)
# --8<-- [end:example]
