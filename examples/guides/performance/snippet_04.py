#!/usr/bin/env python3
"""Resource Lifecycle."""

# --8<-- [start:example]
import coremusic.capi as capi

# Must manually clean up
file_id = capi.audio_file_open_url("audio.wav")
try:
    data = capi.audio_file_read_packets(file_id, 0, 1024)
finally:
    capi.audio_file_close(file_id)  # Don't forget!
# --8<-- [end:example]
