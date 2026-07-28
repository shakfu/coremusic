#!/usr/bin/env python3
"""Functional API and import style."""

# --8<-- [start:functional]
from coremusic import capi

# Direct C function calls
file_id = capi.audio_file_open_url("audio.wav")
data, count = capi.audio_file_read_packets(file_id, 0, 1024)
capi.audio_file_close(file_id)

# Constants are exposed as get_* functions
property_id = capi.get_audio_file_property_data_format()
format_id = capi.fourchar_to_int('lpcm')
# --8<-- [end:functional]

# --8<-- [start:specific]
# --8<-- [end:specific]
# --8<-- [start:grouping]
import time
from pathlib import Path

import numpy as np

from coremusic import link

# --8<-- [end:grouping]
# --8<-- [start:optional]
from coremusic.audio import NUMPY_AVAILABLE, AudioFile
from coremusic.audio.analysis import AudioAnalyzer
from coremusic.midi import MIDIClient

if NUMPY_AVAILABLE:
    ...
# --8<-- [end:optional]
