#!/usr/bin/env python3
"""Detect Silence."""

# --8<-- [start:example]
import numpy as np

from coremusic.audio import AudioFile


def detect_silence(filepath, threshold=0.01, min_duration=0.5):
    """Detect silent regions in audio file."""
    with AudioFile(filepath) as audio:
        fmt = audio.format
        sample_rate = fmt.sample_rate

        # Read audio
        data, count = audio.read_packets(0, audio.packet_count)

        # Convert to numpy
        dtype = np.int16 if fmt.bits_per_channel == 16 else np.int32
        samples = np.frombuffer(data, dtype=dtype)

        # Normalize to [-1, 1]
        max_value = 2 ** (fmt.bits_per_channel - 1)
        samples = samples.astype(np.float32) / max_value

        # Calculate RMS in windows
        window_size = int(0.1 * sample_rate)  # 100ms windows
        num_windows = len(samples) // window_size

        silent_regions = []
        in_silence = False
        silence_start = 0

        for i in range(num_windows):
            window = samples[i * window_size : (i + 1) * window_size]
            rms = np.sqrt(np.mean(window**2))

            if rms < threshold:
                if not in_silence:
                    silence_start = i * window_size / sample_rate
                    in_silence = True
            else:
                if in_silence:
                    silence_end = i * window_size / sample_rate
                    duration = silence_end - silence_start

                    if duration >= min_duration:
                        silent_regions.append((silence_start, silence_end, duration))

                    in_silence = False

        return silent_regions


# Usage
silent_regions = detect_silence("audio.wav")
print(f"Found {len(silent_regions)} silent regions:")
for start, end, duration in silent_regions:
    print(f"  {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")
# --8<-- [end:example]
