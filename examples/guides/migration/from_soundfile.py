#!/usr/bin/env python3
"""The soundfile and wave operations, done with coremusic."""

# --8<-- [start:read]
import numpy as np

from coremusic.audio import AudioFile

# Read entire file
with AudioFile("audio.wav") as audio:
    samples = audio.read_as_numpy()
    sample_rate = audio.format.sample_rate

# Get info without reading the samples
with AudioFile("audio.wav") as audio:
    duration = audio.duration
    channels = audio.format.channels_per_frame
    sample_rate = audio.format.sample_rate
# --8<-- [end:read]

# --8<-- [start:write]
import numpy as np

from coremusic import capi
from coremusic.audio import AudioFormat, ExtendedAudioFile

# Generate audio
data = (np.random.randn(44100) * 0.1).astype(np.float32)

# Describe it
audio_format = AudioFormat.pcm(
    sample_rate=44100.0,
    channels=1,
    bits=32,
    is_float=True,
)

# Write
with ExtendedAudioFile.create(
    "output.wav",
    capi.fourchar_to_int('WAVE'),
    audio_format,
) as audio:
    audio.write(len(data), data.tobytes())
# --8<-- [end:write]

# --8<-- [start:stream]
from coremusic.audio import ExtendedAudioFile

# Read in blocks
with ExtendedAudioFile("audio.wav") as audio:
    while True:
        data, count = audio.read(1024)
        if count == 0:
            break
        # Process block
# --8<-- [end:stream]

# --8<-- [start:wave-read]
from coremusic.audio import ExtendedAudioFile

with ExtendedAudioFile("audio.wav") as audio:
    # Get parameters
    file_format = audio.file_format
    channels = file_format.channels_per_frame
    sample_rate = file_format.sample_rate
    bits = file_format.bits_per_channel
    n_frames = audio.frame_count

    # Read frames
    data, count = audio.read(n_frames)
# --8<-- [end:wave-read]

# --8<-- [start:wave-write]
import numpy as np

from coremusic import capi
from coremusic.audio import AudioFormat, ExtendedAudioFile

data = np.random.randint(-32768, 32767, 44100, dtype=np.int16)

audio_format = AudioFormat.pcm(sample_rate=44100.0, channels=1, bits=16)

with ExtendedAudioFile.create(
    "output16.wav",
    capi.fourchar_to_int('WAVE'),
    audio_format,
) as audio:
    audio.write(len(data), data.tobytes())
# --8<-- [end:wave-write]
