#!/usr/bin/env python3
"""Factory Presets."""

# --8<-- [start:example]
from coremusic.audio.audiounit_host import AudioUnitPlugin

with AudioUnitPlugin.from_name("AUMatrixReverb") as plugin:
    # List factory presets
    print(f"Factory Presets ({len(plugin.factory_presets)}):")
    for preset in plugin.factory_presets:
        print(f"  - {preset.name}")

    # Load first factory preset
    if plugin.factory_presets:
        plugin.load_factory_preset(plugin.factory_presets[0])
        print(f"Loaded preset: {plugin.factory_presets[0].name}")
# --8<-- [end:example]
