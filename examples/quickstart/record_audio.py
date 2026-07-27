#!/usr/bin/env python3
"""Record from the default input device to a WAV file."""

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_input_devices():
    print("No audio input device available.")
    raise SystemExit(0)

# --8<-- [start:example]
from coremusic.capi import AudioRecorder

recorder = AudioRecorder(sample_rate=44100.0, channels=1)
recorder.setup_input(duration=1.0)
recorder.start()

# Pump the CoreAudio run loop until the requested duration has been captured
while recorder.is_recording():
    recorder.run_loop(0.1)

recorder.stop()
recorder.save_to_file("recording.wav")
print(f"Recorded {recorder.get_recorded_duration():.2f}s")
# --8<-- [end:example]
