#!/usr/bin/env python3
"""Functional API."""

# --8<-- [start:example]
from coremusic import capi

# Low-level audio file operations
file_id = capi.audio_file_open_url("audio.wav")
# ... operations ...
capi.audio_file_close(file_id)

# Low-level clock operations
clock_id = capi.ca_clock_new()
capi.ca_clock_start(clock_id)
# ... operations ...
capi.ca_clock_dispose(clock_id)
# --8<-- [end:example]
