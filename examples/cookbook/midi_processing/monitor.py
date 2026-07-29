#!/usr/bin/env python3
"""Monitor."""

# --8<-- [start:example]
import time

from coremusic import capi
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
            hex_data = " ".join(f"{b:02X}" for b in data)
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

print("\nMIDI Monitor - running for 1s")
print("-" * 70)

deadline = time.monotonic() + 1.0
try:
    while time.monotonic() < deadline:
        if not capi.midi_input_wait(input_port, 0.1):
            continue
        for _host_time, payload in capi.midi_input_poll(input_port):
            monitor.process(payload)
except KeyboardInterrupt:
    pass

print(f"\n\nStopped - Received {monitor.message_count} messages")

# Cleanup
for i in range(num_sources):
    source = capi.midi_get_source(i)
    capi.midi_port_disconnect_source(input_port, source)

capi.midi_port_dispose(input_port)
capi.midi_client_dispose(client)
# --8<-- [end:example]
