#!/usr/bin/env python3
"""Managing resources with context managers."""

# --8<-- [start:example]
import coremusic.capi as capi
from coremusic.audio import AudioFile, AudioFormat, ExtendedAudioFile

# Good: automatic cleanup
with AudioFile("audio.wav") as audio:
    data, count = audio.read_packets(0, 1024)
# File automatically closed

# Good: nested context managers
out_format = AudioFormat.pcm(44100.0, channels=2, bits=16)
with AudioFile("input.wav") as input_file:
    with ExtendedAudioFile.create(
        "output.wav", capi.fourchar_to_int("WAVE"), out_format
    ) as output_file:
        data, count = input_file.read_packets(0, 1024)
        output_file.write(count, data)

# Avoid: manual management (error-prone)
audio = AudioFile("audio.wav")
audio.open()
# If an exception is raised here, the file never closes
audio.close()
# --8<-- [end:example]
