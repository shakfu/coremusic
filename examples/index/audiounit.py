#!/usr/bin/env python3
"""Configure and run the default output AudioUnit."""

# --8<-- [start:example]
from coremusic.audio import AudioFormat, AudioUnit

# Create and configure an AudioUnit
with AudioUnit.default_output() as unit:
    format = AudioFormat.pcm(sample_rate=44100.0, channels=2, bits=16)
    unit.set_stream_format(format, scope="input")
    unit.initialize()
    unit.start()
    # ... audio processing ...
    unit.stop()
# --8<-- [end:example]
