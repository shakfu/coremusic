#!/usr/bin/env python3
"""Creating an output queue."""

# --8<-- [start:example]
from coremusic.audio import AudioFormat, AudioQueue

audio_format = AudioFormat.pcm(44100.0, channels=2, bits=16)

queue = AudioQueue.new_output(audio_format)
try:
    # Allocate buffer
    buffer = queue.allocate_buffer(4096)

    # Start playback
    queue.start()

    # ... fill buffer and enqueue ...

    queue.stop()
finally:
    queue.dispose()
# --8<-- [end:example]
