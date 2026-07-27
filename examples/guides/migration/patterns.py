#!/usr/bin/env python3
"""Two whole tasks, ported."""

# --8<-- [start:audio]
import numpy as np

import coremusic.capi as capi
from coremusic.audio import AudioFile, AudioFormat, ExtendedAudioFile

with AudioFile("input.wav") as audio:
    samples = audio.read_as_numpy().astype(np.float32) / 32768.0
    samples *= 2.0  # Increase volume (~6dB)

    out_format = AudioFormat.pcm(
        audio.format.sample_rate,
        channels=audio.format.channels_per_frame,
        bits=32,
        is_float=True,
    )

with ExtendedAudioFile.create(
    "output.wav",
    capi.fourchar_to_int('WAVE'),
    out_format,
) as output:
    output.write(len(samples), samples.tobytes())
# --8<-- [end:audio]

# --8<-- [start:midi]
from coremusic.midi import MIDIClient, get_destinations

client = MIDIClient("App")
port = client.create_output_port("Out")

destinations = get_destinations()
dest = destinations[0] if destinations else client.create_virtual_destination("Out")

for note in [60, 64, 67]:
    port.send_data(dest, bytes([0x90, note, 100]))  # Note on

client.dispose()
# --8<-- [end:midi]
