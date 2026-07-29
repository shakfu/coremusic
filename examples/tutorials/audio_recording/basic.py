#!/usr/bin/env python3
"""Record from the default input device to a WAV file."""

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_input_devices():
    print("No audio input device available.")
    raise SystemExit(0)

# --8<-- [start:example]
from coremusic.capi import AudioRecorder


def record_audio(output_path, duration_seconds):
    """Record audio to a WAV file."""
    # The recorder allocates its buffer up front, so the maximum duration is
    # fixed when the input is set up.
    recorder = AudioRecorder(sample_rate=44100.0, channels=2)
    recorder.setup_input(duration=duration_seconds)

    print(f"Recording for {duration_seconds} seconds...")
    print("Press Ctrl+C to stop early")

    recorder.start()

    try:
        # run_loop() pumps the CoreAudio run loop; without it no buffers
        # arrive, because capture is driven from this thread.
        while recorder.is_recording():
            recorder.run_loop(0.1)
            print(f"Recording: {recorder.get_recorded_duration():.1f}s", end="\r")
    except KeyboardInterrupt:
        print("\nStopped early")

    recorder.stop()
    recorder.save_to_file(output_path)
    print(f"\nSaved to: {output_path}")


record_audio("my_recording.wav", duration_seconds=1)
# --8<-- [end:example]
