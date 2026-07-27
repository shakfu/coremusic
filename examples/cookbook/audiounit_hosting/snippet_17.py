#!/usr/bin/env python3
"""Multi-Channel Performance."""

# --8<-- [start:example]
from coremusic.audio.audiounit_host import AudioUnitPlugin

import time

with AudioUnitPlugin.from_name("DLSMusicDevice", component_type='aumu') as synth:
    # Setup different instruments on different channels
    synth.program_change(channel=0, program=0)   # Piano
    synth.program_change(channel=1, program=48)  # Strings
    synth.program_change(channel=2, program=56)  # Trumpet
    synth.program_change(channel=9, program=0)   # Drums (always channel 9)

    # Play multi-channel arrangement
    synth.note_on(channel=0, note=60, velocity=90)  # Piano: C
    time.sleep(0.25)

    synth.note_on(channel=1, note=64, velocity=70)  # Strings: E
    time.sleep(0.25)

    synth.note_on(channel=2, note=72, velocity=80)  # Trumpet: C (octave up)
    time.sleep(0.25)

    synth.note_on(channel=9, note=36, velocity=100)  # Drums: Kick
    time.sleep(0.5)

    # Clean stop all channels
    for ch in range(10):
        synth.all_notes_off(channel=ch)
# --8<-- [end:example]
