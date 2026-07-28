#!/usr/bin/env python3
"""Program Changes."""

# --8<-- [start:example]
import time

from coremusic.audio.audiounit_host import AudioUnitPlugin

with AudioUnitPlugin.from_name("DLSMusicDevice", component_type='aumu') as synth:
    # Acoustic Grand Piano (GM program 0)
    synth.program_change(channel=0, program=0)
    synth.note_on(channel=0, note=60, velocity=100)
    time.sleep(0.5)
    synth.note_off(channel=0, note=60)

    time.sleep(0.2)

    # Violin (GM program 40)
    synth.program_change(channel=0, program=40)
    synth.note_on(channel=0, note=60, velocity=100)
    time.sleep(0.5)
    synth.note_off(channel=0, note=60)

    time.sleep(0.2)

    # Trumpet (GM program 56)
    synth.program_change(channel=0, program=56)
    synth.note_on(channel=0, note=60, velocity=100)
    time.sleep(0.5)
    synth.note_off(channel=0, note=60)
# --8<-- [end:example]
