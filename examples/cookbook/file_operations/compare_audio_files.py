#!/usr/bin/env python3
"""Compare Audio Files."""

# --8<-- [start:example]
from coremusic.audio import AudioFile

def compare_audio_files(file1, file2):
    """Compare two audio files for format compatibility."""
    with AudioFile(file1) as audio1, AudioFile(file2) as audio2:
        fmt1 = audio1.format
        fmt2 = audio2.format

        comparison = {
            'same_sample_rate': fmt1.sample_rate == fmt2.sample_rate,
            'same_channels': fmt1.channels_per_frame == fmt2.channels_per_frame,
            'same_bit_depth': fmt1.bits_per_channel == fmt2.bits_per_channel,
            'same_format': fmt1.format_id == fmt2.format_id,
            'same_duration': abs(audio1.duration - audio2.duration) < 0.01,
        }

        comparison['compatible'] = all(comparison.values())

        return comparison

# Usage
result = compare_audio_files("audio.wav", "input.wav")
print(f"Files compatible: {result['compatible']}")
# --8<-- [end:example]
