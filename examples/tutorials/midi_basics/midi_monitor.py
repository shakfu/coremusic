#!/usr/bin/env python3
"""Display incoming MIDI messages in a human-readable form."""

# --8<-- [start:example]
import time

from coremusic.midi import MIDIClient, MIDIMessageSplitter, get_sources


class MIDIMonitor:
    """Monitor and display incoming MIDI messages."""

    def __init__(self):
        self.client = MIDIClient("MIDI Monitor")
        self.splitter = MIDIMessageSplitter()

    def parse_message(self, data):
        """Parse MIDI message bytes."""
        if len(data) == 0:
            return None

        status = data[0]
        channel = status & 0x0F
        msg_type = status & 0xF0

        if msg_type == 0x90 and len(data) >= 3:
            # Note On
            note, velocity = data[1], data[2]
            if velocity > 0:
                return f"Note On  ch={channel} note={note} vel={velocity}"
            else:
                return f"Note Off ch={channel} note={note}"

        elif msg_type == 0x80 and len(data) >= 3:
            # Note Off
            note = data[1]
            return f"Note Off ch={channel} note={note}"

        elif msg_type == 0xB0 and len(data) >= 3:
            # Control Change
            cc, value = data[1], data[2]
            return f"CC       ch={channel} cc={cc} val={value}"

        elif msg_type == 0xC0 and len(data) >= 2:
            # Program Change
            program = data[1]
            return f"Program  ch={channel} prog={program}"

        elif msg_type == 0xE0 and len(data) >= 3:
            # Pitch Bend
            lsb, msb = data[1], data[2]
            value = (msb << 7) | lsb
            return f"PitchBnd ch={channel} val={value}"

        else:
            return f"Unknown  {' '.join(f'{b:02X}' for b in data)}"

    def run(self, seconds):
        """Print every message that arrives within `seconds`."""
        input_port = self.client.create_input_port("Monitor Input")

        sources = get_sources()
        print(f"Monitoring {len(sources)} MIDI sources for {seconds}s...")

        for source in sources:
            input_port.connect_source(source)

        deadline = time.monotonic() + seconds
        try:
            while time.monotonic() < deadline:
                if not input_port.wait(0.1):
                    continue
                for _host_time, data in input_port.poll():
                    for message in self.splitter.push(data):
                        text = self.parse_message(message)
                        if text:
                            print(text)
        except KeyboardInterrupt:
            print("\nStopping...")

        self.client.dispose()


monitor = MIDIMonitor()
monitor.run(seconds=1.0)
# --8<-- [end:example]
