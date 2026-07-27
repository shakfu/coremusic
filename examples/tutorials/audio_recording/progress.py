#!/usr/bin/env python3
"""Record with a progress bar."""

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_input_devices():
    print("No audio input device available.")
    raise SystemExit(0)

# --8<-- [start:example]
import sys

from coremusic.capi import AudioRecorder


def record_with_progress(output_path, duration):
    """Record with visual progress bar."""
    recorder = AudioRecorder(sample_rate=44100.0, channels=2)
    recorder.setup_input(duration=duration)

    print(f"Recording: {output_path}")
    print(f"Duration: {duration}s")
    print()

    recorder.start()

    try:
        while recorder.is_recording():
            recorder.run_loop(0.05)

            progress = recorder.get_progress()
            elapsed = recorder.get_recorded_duration()

            bar_width = 40
            filled = int(bar_width * progress)
            bar = '=' * filled + '-' * (bar_width - filled)

            sys.stdout.write(f'\r[{bar}] {elapsed:.1f}s')
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nStopped by user")

    recorder.stop()
    recorder.save_to_file(output_path)
    print(f"\nRecording saved to: {output_path}")


record_with_progress("recording.wav", duration=1)
# --8<-- [end:example]
