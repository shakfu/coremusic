#!/usr/bin/env python3
"""File Properties."""

# --8<-- [start:example]
from coremusic import capi

file_id = capi.audio_file_open_url("audio.wav")
try:
    # Get audio format
    format_data = capi.audio_file_get_property(
        file_id, capi.get_audio_file_property_data_format()
    )
    print(f"Format: {format_data}")
finally:
    capi.audio_file_close(file_id)
# --8<-- [end:example]
