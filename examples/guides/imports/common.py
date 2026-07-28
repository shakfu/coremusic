#!/usr/bin/env python3
"""The imports most programs start with."""

# --8<-- [start:example]
# Object-oriented API, one subpackage per domain
from coremusic.audio import AudioFile, AudioFormat
from coremusic.base import AudioPlayer
from coremusic.midi import MIDIClient, MusicSequence

player = AudioPlayer()
audio = AudioFile("audio.wav")
sequence = MusicSequence()

# Functional C API (for performance)
from coremusic import capi

file_id = capi.audio_file_open_url("audio.wav")
capi.audio_file_close(file_id)
# --8<-- [end:example]

audio.dispose()
sequence.dispose()
