#!/usr/bin/env python3
"""Get the recorded samples as a NumPy array."""

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_input_devices():
    print("No audio input device available.")
    raise SystemExit(0)

# --8<-- [start:example]
from coremusic.base import NUMPY_AVAILABLE

if NUMPY_AVAILABLE:
    import numpy as np

    from coremusic.capi import AudioRecorder

    def record_to_numpy(duration, sample_rate=44100.0, channels=2):
        """Record audio and return it as a NumPy array."""
        recorder = AudioRecorder(sample_rate=sample_rate, channels=channels)
        recorder.setup_input(duration=duration)

        recorder.start()
        while recorder.is_recording():
            recorder.run_loop(0.05)
        recorder.stop()

        # The recorder captures interleaved float32
        raw = recorder.get_audio_data()
        audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, channels)

        return audio, sample_rate

    # Record and process
    audio, sr = record_to_numpy(duration=0.5)
    print(f"Recorded {len(audio)} frames at {sr}Hz")
    print(f"Peak amplitude: {np.max(np.abs(audio)):.4f}")
# --8<-- [end:example]
