#!/usr/bin/env python3
"""Pitch Bend."""

# --8<-- [start:example]
import time

from coremusic.audio.audiounit_host import AudioUnitPlugin

with AudioUnitPlugin.from_name("DLSMusicDevice", component_type="aumu") as synth:
    synth.note_on(channel=0, note=60, velocity=100)

    # Center (no bend)
    synth.pitch_bend(channel=0, value=8192)
    time.sleep(0.3)

    # Bend up (one semitone)
    synth.pitch_bend(channel=0, value=12288)
    time.sleep(0.3)

    # Back to center
    synth.pitch_bend(channel=0, value=8192)
    time.sleep(0.3)

    # Bend down (one semitone)
    synth.pitch_bend(channel=0, value=4096)
    time.sleep(0.3)

    # Back to center
    synth.pitch_bend(channel=0, value=8192)
    time.sleep(0.3)

    synth.note_off(channel=0, note=60)
# --8<-- [end:example]
