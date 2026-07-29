#!/usr/bin/env python3
"""Constants in API Calls."""

# --8<-- [start:example]
from coremusic import capi
from coremusic.constants import AudioFileProperty

# Use constant enum in functional API
file_id = capi.audio_file_open_url("audio.wav")
format_data = capi.audio_file_get_property(
    file_id,
    int(AudioFileProperty.DATA_FORMAT),  # Convert to int
)
capi.audio_file_close(file_id)
# --8<-- [end:example]
