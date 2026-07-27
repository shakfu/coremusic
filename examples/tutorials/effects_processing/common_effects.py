#!/usr/bin/env python3
"""Configuring the effects you reach for most."""

# --8<-- [start:reverb]
from coremusic.audio.audiounit_host import AudioUnitPlugin

ROOMS = {
    "small": "Small Room",
    "medium": "Medium Room",
    "large": "Large Room",
    "hall": "Large Hall",
}


def make_reverb(room_size="medium"):
    """Return an initialized reverb set to a room preset."""
    plugin = AudioUnitPlugin.from_name("AUMatrixReverb", component_type="aufx")
    plugin.instantiate().initialize()

    wanted = ROOMS.get(room_size, "Medium Room")
    for preset in plugin.factory_presets:
        if preset.name == wanted:
            plugin.load_factory_preset(preset)
            print(f"Reverb configured: {preset.name}")
            break

    return plugin


reverb = make_reverb("large")
reverb.dispose()
# --8<-- [end:reverb]

# --8<-- [start:delay]
from coremusic.audio.audiounit_host import AudioUnitPlugin

NOTE_VALUES = {
    "1/1": 4.0,
    "1/2": 2.0,
    "1/4": 1.0,
    "1/8": 0.5,
    "1/16": 0.25,
    "1/8T": 1.0 / 3.0,  # Triplet
    "1/8D": 0.75,       # Dotted
}


def make_delay(tempo_bpm=120, note_value="1/4"):
    """Return a delay whose time matches a note value at a tempo."""
    beat_duration = 60.0 / tempo_bpm
    delay_time = beat_duration * NOTE_VALUES.get(note_value, 1.0)

    plugin = AudioUnitPlugin.from_name("AUDelay", component_type="aufx")
    plugin.instantiate().initialize()
    plugin.set_parameter("Delay Time", delay_time)

    print(f"Delay configured for {tempo_bpm} BPM:")
    print(f"  Note value: {note_value}")
    print(f"  Delay time: {delay_time:.3f}s")

    return plugin


delay = make_delay(tempo_bpm=120, note_value="1/8")
delay.dispose()
# --8<-- [end:delay]

# --8<-- [start:eq]
from coremusic.audio.audiounit_host import AudioUnitPlugin


def make_eq(gains_db):
    """Return an N-band EQ with the given band gains applied."""
    plugin = AudioUnitPlugin.from_name("AUNBandEQ", component_type="aufx")
    plugin.instantiate().initialize()

    # Band gains are the parameters whose names end in "gain"
    gain_params = [p for p in plugin.parameters if p.name.lower().endswith("gain")]
    for param, gain in zip(gain_params, gains_db):
        plugin.set_parameter(param.id, gain)

    print(f"EQ configured across {len(gain_params)} bands")
    return plugin


eq = make_eq([-2, 0, 3, 2, -1])
eq.dispose()
# --8<-- [end:eq]
