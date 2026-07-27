#!/usr/bin/env python3
"""The same file read twice: functional API, then object-oriented API."""

# --8<-- [start:functional]
import coremusic.capi as capi

# Open audio file (manual resource management)
audio_file = capi.audio_file_open_url("audio.wav")
try:
    format_data = capi.audio_file_get_property(
        audio_file,
        capi.get_audio_file_property_data_format()
    )
    data, count = capi.audio_file_read_packets(audio_file, 0, 1000)
finally:
    capi.audio_file_close(audio_file)
# --8<-- [end:functional]

# --8<-- [start:object-oriented]
from coremusic.audio import AudioFile

# Automatic resource management with context manager
with AudioFile("audio.wav") as audio_file:
    print(f"Duration: {audio_file.duration:.2f}s")
    print(f"Sample rate: {audio_file.format.sample_rate}Hz")
    data, count = audio_file.read_packets(0, 1000)
# --8<-- [end:object-oriented]
