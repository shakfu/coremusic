#!/usr/bin/env python3
"""Input to output, with processing in between."""

from coremusic.audio import AudioDeviceManager

if not (AudioDeviceManager.get_input_devices()
        and AudioDeviceManager.get_output_devices()):
    print("Real-time processing needs both an input and an output device.")
    raise SystemExit(0)

# --8<-- [start:example]
import time

from coremusic.audio.streaming import AudioProcessor


def gain(audio_data):
    """Called for every block: return the processed block."""
    return audio_data * 0.5


processor = AudioProcessor(gain, channels=2, sample_rate=44100.0, buffer_size=512)

try:
    processor.start()
except RuntimeError as e:
    print(e)
    raise SystemExit(0) from None

print(f"Round-trip latency: {processor.latency * 1000:.1f}ms")
time.sleep(0.5)
processor.stop()

print(f"Overruns: {processor.overruns}, underruns: {processor.underruns}")
# --8<-- [end:example]
