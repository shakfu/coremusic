#!/usr/bin/env python3
"""Generate audio straight to the output device."""

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_output_devices():
    print("No audio output device available.")
    raise SystemExit(0)

# --8<-- [start:example]
import math
import struct
import time

from coremusic.audio.streaming import AudioOutputStream


def make_sine(freq, sample_rate, channels, gain=0.2):
    """Return generator(frame_count) -> interleaved float32 bytes."""
    phase = 0.0
    step = 2.0 * math.pi * freq / sample_rate

    def generate(frame_count):
        nonlocal phase
        samples = []
        for _ in range(frame_count):
            samples.extend([math.sin(phase) * gain] * channels)
            phase = (phase + step) % (2.0 * math.pi)
        return struct.pack(f"<{len(samples)}f", *samples)

    return generate


# A small buffer means low latency; too small and the generator cannot keep up
stream = AudioOutputStream(channels=2, sample_rate=44100.0, buffer_size=256)
stream.set_generator(make_sine(440.0, 44100.0, channels=2))
stream.start()

print(f"Latency: {stream.latency * 1000:.1f}ms")
time.sleep(0.5)

stream.stop()
# --8<-- [end:example]
