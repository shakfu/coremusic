#!/usr/bin/env python3
"""Writing and reading audio file tags."""

# CAF supports the full tag dictionary; make one to write into.
import coremusic.capi as capi
from coremusic.audio import AudioFormat, ExtendedAudioFile

with ExtendedAudioFile.create(
    "song.caf", capi.fourchar_to_int("caff"), AudioFormat.pcm(44100.0, channels=2)
) as f:
    f.write(44100, bytes(44100 * 4))

# --8<-- [start:example]
from coremusic.audio import AudioFile

# Write tags to a CAF file
with AudioFile("song.caf", writable=True) as af:
    af.set_metadata({
        "title": "My Song",
        "artist": "Artist Name",
        "album": "Album Name",
        "genre": "Electronic",
        "year": "2026",
        "track number": "3",
        "comments": "Mixed in mono",
    })

# Read them back
with AudioFile("song.caf") as af:
    print(af.metadata)
# --8<-- [end:example]
