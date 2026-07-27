#!/usr/bin/env python3
"""Recording at other sample rates and channel counts."""

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_input_devices():
    print("No audio input device available.")
    raise SystemExit(0)


def capture(recorder, path):
    recorder.start()
    while recorder.is_recording():
        recorder.run_loop(0.1)
    recorder.stop()
    recorder.save_to_file(path)


# --8<-- [start:high-quality]
from coremusic.capi import AudioRecorder


def record_high_quality(output_path, duration):
    """Record at a professional sample rate.

    The recorder always captures 32-bit float, which is what the AudioQueue
    hands over; convert on the way out if you need another depth.
    """
    recorder = AudioRecorder(sample_rate=96000.0, channels=2)
    recorder.setup_input(duration=duration)

    print("Recording at 96kHz...")
    capture(recorder, output_path)
    print(f"High-quality recording saved to: {output_path}")


record_high_quality("hq_recording.wav", duration=0.5)
# --8<-- [end:high-quality]

# --8<-- [start:mono]
from coremusic.capi import AudioRecorder


def record_mono(output_path, duration):
    """Record single channel (mono) audio."""
    recorder = AudioRecorder(sample_rate=44100.0, channels=1)
    recorder.setup_input(duration=duration)

    print("Recording mono...")
    capture(recorder, output_path)
    print(f"Mono recording saved to: {output_path}")


record_mono("mono_recording.wav", duration=0.5)
# --8<-- [end:mono]
