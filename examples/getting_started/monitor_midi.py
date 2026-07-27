#!/usr/bin/env python3
"""Print incoming MIDI from every source."""

# --8<-- [start:example]
import time

from coremusic.midi import MIDIClient, get_sources


def monitor_midi(seconds):
    """Monitor MIDI input from all sources."""
    client = MIDIClient("MIDI Monitor")

    try:
        input_port = client.create_input_port(
            "Monitor Input",
            callback=lambda data, host_time: print(f"Received {data.hex()}"),
        )

        sources = get_sources()
        for source in sources:
            input_port.connect_source(source)

        print(f"Monitoring {len(sources)} MIDI sources...")
        print("Press Ctrl+C to stop")

        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping monitor...")
    finally:
        client.dispose()


if __name__ == "__main__":
    monitor_midi(seconds=1.0)
# --8<-- [end:example]
