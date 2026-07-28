#!/usr/bin/env python3
"""Meter the input in real time."""

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_input_devices():
    print("No audio input device available.")
    raise SystemExit(0)

# --8<-- [start:example]
import time

import numpy as np

from coremusic.audio.streaming import AudioInputStream


class AudioLevelMonitor:
    """Real-time audio level monitoring"""

    def __init__(self, sample_rate=44100.0):
        self.stream = AudioInputStream(channels=2, sample_rate=sample_rate)
        self.peak_level = 0.0
        self.rms_level = 0.0
        self.stream.add_callback(self._measure)

    def _measure(self, audio_data, frame_count):
        if frame_count == 0:
            return
        samples = np.asarray(audio_data, dtype=np.float32)
        self.peak_level = float(np.max(np.abs(samples)))
        self.rms_level = float(np.sqrt(np.mean(samples ** 2)))

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()

    def get_levels(self):
        """Get current levels in dB"""
        peak_db = 20 * np.log10(self.peak_level) if self.peak_level > 0 else -100
        rms_db = 20 * np.log10(self.rms_level) if self.rms_level > 0 else -100
        return {"peak_db": peak_db, "rms_db": rms_db}


monitor = AudioLevelMonitor()
try:
    monitor.start()
except RuntimeError as e:
    print(e)
    raise SystemExit(0) from None

for _ in range(5):
    levels = monitor.get_levels()
    print(f"Peak: {levels['peak_db']:.1f} dB, RMS: {levels['rms_db']:.1f} dB")
    time.sleep(0.1)

monitor.stop()
# --8<-- [end:example]
