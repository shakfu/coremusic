#!/usr/bin/env python3
"""MIDI Controllers."""

# --8<-- [start:example]
import time

from coremusic.audio.audiounit_host import AudioUnitPlugin

with AudioUnitPlugin.from_name("DLSMusicDevice", component_type='aumu') as synth:
    synth.note_on(channel=0, note=60, velocity=100)

    # Volume fade (CC 7)
    for volume in range(127, 0, -10):
        synth.control_change(channel=0, controller=7, value=volume)
        time.sleep(0.1)

    synth.note_off(channel=0, note=60)

    # Pan sweep (CC 10)
    synth.note_on(channel=0, note=60, velocity=100)
    for pan in range(0, 128, 5):
        synth.control_change(channel=0, controller=10, value=pan)
        time.sleep(0.05)

    synth.note_off(channel=0, note=60)
# --8<-- [end:example]
