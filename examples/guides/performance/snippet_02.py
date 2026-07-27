#!/usr/bin/env python3
"""Hybrid Approach."""

# --8<-- [start:example]
import coremusic.capi as capi
from coremusic.audio import AudioFile

# Use OO API for file management
with AudioFile("input.wav") as audio:
    format = audio.format  # OO API convenience

    # Switch to functional API for bulk processing
    file_id = audio.object_id
    for i in range(0, audio.packet_count, 4096):
        # Direct C calls - maximum performance
        data, count = capi.audio_file_read_packets(
            file_id, i, 4096
        )
        # Process data...
# --8<-- [end:example]
