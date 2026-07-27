#!/usr/bin/env python3
"""Reading Audio Data."""

# --8<-- [start:example]
import coremusic.capi as capi

file_id = capi.audio_file_open_url("audio.wav")
try:
    # Read 1000 packets starting from packet 0
    data, packets_read = capi.audio_file_read_packets(file_id, 0, 1000)
    print(f"Read {packets_read} packets, {len(data)} bytes")
finally:
    capi.audio_file_close(file_id)
# --8<-- [end:example]
