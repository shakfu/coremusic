#!/usr/bin/env python3
"""Object layer versus functional layer, timed."""

# --8<-- [start:example]
import time

from coremusic import capi
from coremusic.audio import AudioFile

test_file = "audio.wav"

# Object-Oriented API
start = time.time()
with AudioFile(test_file) as audio:
    data, count = audio.read_packets(0, 1024)
oo_time = time.time() - start

# Functional API
start = time.time()
file_id = capi.audio_file_open_url(test_file)
data, count = capi.audio_file_read_packets(file_id, 0, 1024)
capi.audio_file_close(file_id)
func_time = time.time() - start

print(f"OO API: {oo_time:.4f}s")
print(f"Functional API: {func_time:.4f}s")
print(f"Overhead: {((oo_time / func_time - 1) * 100):.1f}%")
# --8<-- [end:example]
