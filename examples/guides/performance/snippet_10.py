#!/usr/bin/env python3
"""Configuring the output unit for low latency."""

# --8<-- [start:example]
from coremusic.audio import AudioFormat, AudioUnit

# Create low-latency audio unit
unit = AudioUnit.default_output()

# Configure the format you will feed it. The output scope belongs to the
# device, so the client format goes on the input scope.
audio_format = AudioFormat.pcm(
    sample_rate=44100.0, channels=2, bits=32, is_float=True
)
unit.set_stream_format(audio_format, scope="input")

# Smaller slices mean lower latency: 256 frames at 44.1kHz is about 5.8ms
unit.max_frames_per_slice = 256

unit.initialize()
unit.start()
print(f"Latency: {unit.latency * 1000:.2f}ms")
unit.stop()
unit.dispose()
# --8<-- [end:example]
