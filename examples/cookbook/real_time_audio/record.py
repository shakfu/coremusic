#!/usr/bin/env python3
"""Capture live input."""

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_input_devices():
    print("No audio input device available.")
    raise SystemExit(0)

# --8<-- [start:example]
import time

from coremusic.audio.streaming import AudioInputStream

captured = []


def collect(audio_data, frame_count):
    """Runs on the audio thread for every captured block - keep it short."""
    captured.append(audio_data)


stream = AudioInputStream(channels=2, sample_rate=44100.0, buffer_size=512)
stream.add_callback(collect)

try:
    stream.start()
except RuntimeError as e:
    # macOS refuses input until the app has microphone permission
    print(e)
    raise SystemExit(0)

time.sleep(0.5)
stream.stop()

print(f"Captured {len(captured)} blocks, {stream.overruns} overruns")
# --8<-- [end:example]
