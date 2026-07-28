#!/usr/bin/env python3
"""Thread Safety."""

# --8<-- [start:example]
import queue
import time

from coremusic import capi


class MIDIProcessor:
    def __init__(self):
        self.incoming: queue.SimpleQueue = queue.SimpleQueue()

    def callback(self, data: bytes, host_time: int) -> None:
        # Runs on the CoreMIDI thread: hand off and return immediately.
        self.incoming.put((host_time, data))

    def run(self, deadline: float) -> None:
        # Runs on your own thread.
        while time.monotonic() < deadline:
            try:
                host_time, data = self.incoming.get(timeout=0.1)
            except queue.Empty:
                continue
            print(f"{host_time}: {data.hex()}")


processor = MIDIProcessor()

# Whatever the callback is attached to, the hand-off looks like this
processor.callback(b"\x90\x3c\x64", capi.midi_current_host_time())
processor.run(deadline=time.monotonic() + 0.2)
# --8<-- [end:example]
