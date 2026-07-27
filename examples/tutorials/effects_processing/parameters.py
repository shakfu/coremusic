#!/usr/bin/env python3
"""Reading and setting plugin parameters and presets."""

# --8<-- [start:list]
from coremusic.audio.audiounit_host import AudioUnitPlugin


def list_effect_parameters(effect_name):
    """List all parameters of an effect."""
    with AudioUnitPlugin.from_name(effect_name, component_type="aufx") as plugin:
        print(f"Parameters for {effect_name}:")
        print("-" * 50)

        for param in plugin.parameters:
            print(f"  {param.name}")
            print(f"    ID: {param.id}")
            print(f"    Range: {param.min_value} - {param.max_value}")
            print(f"    Default: {param.default_value}")
            print(f"    Value: {param.value}")
            print()


list_effect_parameters("AUDelay")
# --8<-- [end:list]

# --8<-- [start:configure]
from coremusic.audio.audiounit_host import AudioUnitPlugin


def configure_delay_effect():
    """Configure delay effect parameters."""
    with AudioUnitPlugin.from_name("AUDelay", component_type="aufx") as plugin:
        # Parameters are addressed by name or by id
        plugin.set_parameter("Delay Time", 0.25)     # seconds
        plugin.set_parameter("Feedback", 50.0)       # percent
        plugin['Dry/Wet Mix'] = 30.0                 # percent

        print("Delay configured:")
        print(f"  Delay Time: {plugin.get_parameter('Delay Time').value}s")
        print(f"  Feedback: {plugin.get_parameter('Feedback').value}%")
        print(f"  Mix: {plugin.get_parameter('Dry/Wet Mix').value}%")


configure_delay_effect()
# --8<-- [end:configure]

# --8<-- [start:presets]
from coremusic.audio.audiounit_host import AudioUnitPlugin


def use_effect_preset(effect_name, preset_name):
    """Apply a factory preset to an effect."""
    with AudioUnitPlugin.from_name(effect_name, component_type="aufx") as plugin:
        presets = plugin.factory_presets

        print(f"Available presets for {effect_name}:")
        for i, preset in enumerate(presets):
            print(f"  [{i}] {preset.name}")

        for preset in presets:
            if preset_name.lower() in preset.name.lower():
                plugin.load_factory_preset(preset)
                print(f"\nApplied preset: {preset.name}")
                return

        print(f"\nPreset not found: {preset_name}")


use_effect_preset("AUMatrixReverb", "Large Hall")
# --8<-- [end:presets]
