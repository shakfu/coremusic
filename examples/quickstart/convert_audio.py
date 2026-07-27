#!/usr/bin/env python3
"""Convert an audio file to another format."""

# --8<-- [start:example]
from coremusic.audio import AudioFormatPresets, convert_audio_file

# Convert stereo to mono
output_format = AudioFormatPresets.wav_44100_mono()
convert_audio_file("input.wav", "output.wav", output_format)
# --8<-- [end:example]

# --8<-- [start:shortcut]
from coremusic.shortcuts import convert

convert("input.wav", "output_mono.wav", channels=1)
# --8<-- [end:shortcut]
