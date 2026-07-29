#!/usr/bin/env python3
"""Meter the input signal in real time."""

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_input_devices():
    print("No audio input device available.")
    raise SystemExit(0)

# --8<-- [start:example]
import sys
import time

from coremusic.audio.streaming import AudioInputStream

levels = [0.0, 0.0]


def measure(audio_data, frame_count):
    """Called on the audio thread for every captured block.

    Keep it short: this runs in real time, so do the display elsewhere.
    """
    import numpy as np

    if frame_count == 0:
        return
    peak = np.max(np.abs(audio_data), axis=0)
    levels[0] = float(peak[0])
    levels[1] = float(peak[-1])


stream = AudioInputStream(channels=2, sample_rate=44100.0, buffer_size=512)
stream.add_callback(measure)

try:
    stream.start()
except RuntimeError as e:
    # macOS refuses the input stream until the app has microphone permission
    print(e)
    raise SystemExit(0) from None

print("Monitoring input levels")
print("=" * 50)

deadline = time.monotonic() + 1.0
while time.monotonic() < deadline:
    meter_width = 20
    left_bar = "|" * int(levels[0] * meter_width)
    right_bar = "|" * int(levels[1] * meter_width)

    sys.stdout.write(f"\rL:{left_bar:<20} R:{right_bar:<20}")
    sys.stdout.flush()
    time.sleep(0.05)

stream.stop()
print(f"\nOverruns: {stream.overruns}")
# --8<-- [end:example]
