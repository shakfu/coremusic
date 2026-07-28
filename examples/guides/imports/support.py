#!/usr/bin/env python3
"""Importing from the supporting subpackages."""

# --8<-- [start:music]
from coremusic.music.theory import Chord, ChordType, Note, Scale, ScaleType

c_major = Scale(Note("C", 4), ScaleType.MAJOR)
# --8<-- [end:music]

# --8<-- [start:scipy]
# Individual helpers, or the module itself - handy for the availability flag
import coremusic.utils.scipy as spu
from coremusic.utils.scipy import (
    SCIPY_AVAILABLE,
    apply_lowpass_filter,
    compute_spectrum,
    resample_audio,
)

print(f"SciPy available: {spu.SCIPY_AVAILABLE}")
# --8<-- [end:scipy]

# --8<-- [start:fourcc]
from coremusic.utils.fourcc import fourcc_to_int, fourcc_to_str

print(fourcc_to_int("lpcm"), fourcc_to_str(1819304813))

# The same conversion exists in the functional API
from coremusic import capi

print(capi.fourchar_to_int("lpcm"), capi.int_to_fourchar(1819304813))
# --8<-- [end:fourcc]

# --8<-- [start:constants]
from coremusic.constants import AudioFileProperty, AudioFormatID, MIDIStatus

print(AudioFormatID.LINEAR_PCM, MIDIStatus.NOTE_ON)
# --8<-- [end:constants]

# --8<-- [start:exceptions]
from coremusic.exceptions import AudioFileError, CoreAudioError, MIDIError

# Every coremusic exception derives from CoreAudioError
print(issubclass(AudioFileError, CoreAudioError))
# --8<-- [end:exceptions]

# --8<-- [start:link]
from coremusic import link

session = link.LinkSession(bpm=120.0)
session.enabled = True

state = session.capture_app_session_state()
print(f"Tempo: {state.tempo} BPM")

clock = link.Clock()
print(f"Clock: {clock.micros()} us")

session.enabled = False
# --8<-- [end:link]

# --8<-- [start:shortcuts]
from coremusic.shortcuts import convert, get_info, play

info = get_info("audio.wav")
print(f"{info['duration']:.2f}s")
# --8<-- [end:shortcuts]
