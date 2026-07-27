#!/usr/bin/env python3
"""Convert PCM data between formats with AudioConverter."""

# --8<-- [start:example]
from coremusic.audio import AudioConverter, AudioFormat

# Define formats
src_fmt = AudioFormat.pcm(44100.0, channels=2, bits=16)
dst_fmt = AudioFormat.pcm(44100.0, channels=2, bits=32, is_float=True)

# One second of silence, as 16-bit stereo
input_data = bytes(44100 * 2 * 2)

# Convert. `convert()` handles depth and channel changes; a change of sample
# rate needs `convert_with_callback()`, which pulls input as the converter
# asks for it.
converter = AudioConverter(src_fmt, dst_fmt)
output = converter.convert(input_data)
print(f"{len(input_data)} bytes in, {len(output)} bytes out")
# --8<-- [end:example]

# --8<-- [start:resample]
resampler = AudioConverter(
    AudioFormat.pcm(44100.0, channels=2, bits=16),
    AudioFormat.pcm(48000.0, channels=2, bits=16),
)
frame_count = len(input_data) // 4
output = resampler.convert_with_callback(input_data, frame_count)
print(f"resampled to {len(output)} bytes")
# --8<-- [end:resample]
