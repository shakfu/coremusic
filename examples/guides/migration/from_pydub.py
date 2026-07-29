#!/usr/bin/env python3
"""The pydub operations, done with coremusic."""

# --8<-- [start:load]
from coremusic.audio import AudioFile, ExtendedAudioFile

# Load audio (supports WAV, MP3, AAC, AIFF, etc.)
with AudioFile("audio.wav") as audio:
    # Get properties
    duration = audio.duration  # seconds
    sample_rate = audio.format.sample_rate
    channels = audio.format.channels_per_frame

# Or, to decode a compressed file to PCM as you read it
with ExtendedAudioFile("audio.wav") as audio:
    file_format = audio.file_format
    frame_count = audio.frame_count
# --8<-- [end:load]

# --8<-- [start:operations]
import numpy as np

from coremusic import capi
from coremusic.audio import AudioFile, ExtendedAudioFile
from coremusic.audio.slicing import AudioSlicer

# Load the whole file as samples
with AudioFile("input.wav") as audio:
    samples = audio.read_as_numpy().astype(np.float32) / 32768.0
    audio_format = audio.format

# Volume adjustment
samples *= 3.16  # +10 dB
samples *= 0.56  # -5 dB

# Slicing: AudioSlicer finds musical boundaries rather than cutting blindly
slicer = AudioSlicer("input.wav")
slices = slicer.detect_slices()

# Export
out_format = capi.fourchar_to_int("WAVE")
float_format = audio_format.pcm(
    audio_format.sample_rate,
    channels=audio_format.channels_per_frame,
    bits=32,
    is_float=True,
)
with ExtendedAudioFile.create("output.wav", out_format, float_format) as output:
    output.write(len(samples), samples.tobytes())
# --8<-- [end:operations]
