#!/usr/bin/env python3
"""Basic Note Control."""

# --8<-- [start:example]
from coremusic.audio.audiounit_host import AudioUnitPlugin

import time

# Load instrument plugin
with AudioUnitPlugin.from_name("DLSMusicDevice", component_type='aumu') as synth:
    # Play middle C
    synth.note_on(channel=0, note=60, velocity=100)
    time.sleep(1.0)
    synth.note_off(channel=0, note=60)

    # Play a chord (C major: C, E, G)
    notes = [60, 64, 67]
    for note in notes:
        synth.note_on(channel=0, note=note, velocity=90)

    time.sleep(1.5)

    # Stop all notes at once
    synth.all_notes_off(channel=0)
# --8<-- [end:example]
