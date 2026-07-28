#!/usr/bin/env python3
"""Retry Pattern."""

# --8<-- [start:example]
import time

from coremusic.audio import AudioFile
from coremusic.exceptions import CoreAudioError


def retry_operation(func, max_retries=3, delay=0.5):
    """Retry an operation with exponential backoff."""
    last_error = None

    for attempt in range(max_retries):
        try:
            return func()
        except CoreAudioError as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))

    raise last_error

# Usage
def read_file():
    with AudioFile("audio.wav") as audio:
        return audio.read_packets(0, 1024)

data, count = retry_operation(read_file)
# --8<-- [end:example]
