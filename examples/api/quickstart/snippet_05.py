#!/usr/bin/env python3
"""Running the default output unit."""

# --8<-- [start:example]
from coremusic.audio import AudioFormat, AudioUnit

with AudioUnit.default_output() as unit:
    # Set the format you will feed the unit. The output scope is the device's
    # own format, so the client format goes on the input scope.
    audio_format = AudioFormat.pcm(44100.0, channels=2, bits=16)
    unit.set_stream_format(audio_format, scope="input")

    unit.initialize()

    # Start audio processing
    unit.start()
    # ... audio flows ...
    unit.stop()
# --8<-- [end:example]
