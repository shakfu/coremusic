#!/usr/bin/env python3
"""Opening and Closing Files."""

# --8<-- [start:example]
import coremusic.capi as capi

# Open audio file
file_id = capi.audio_file_open_url("audio.wav")
try:
    # Use file...
    pass
finally:
    capi.audio_file_close(file_id)
# --8<-- [end:example]
