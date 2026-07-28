#!/usr/bin/env python3
"""File Hash Cache."""

# --8<-- [start:example]
import hashlib
from pathlib import Path

from coremusic.audio import AudioFile


class AudioCache:
    """Cache audio data by file hash."""

    def __init__(self):
        self._cache = {}

    def _get_file_hash(self, filepath):
        """Get MD5 hash of file."""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_data(self, filepath):
        """Get cached audio data or load from file."""
        file_hash = self._get_file_hash(filepath)

        if file_hash not in self._cache:
            with AudioFile(filepath) as audio:
                data, count = audio.read_packets(0, audio.packet_count)
                self._cache[file_hash] = data

        return self._cache[file_hash]

# Usage
cache = AudioCache()
data = cache.get_data("audio.wav")
# --8<-- [end:example]
