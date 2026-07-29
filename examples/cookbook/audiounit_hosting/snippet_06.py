#!/usr/bin/env python3
"""User Presets."""

# --8<-- [start:example]
from coremusic.audio.audiounit_host import AudioUnitPlugin

with AudioUnitPlugin.from_name("AUDelay") as plugin:
    # Configure plugin
    plugin["Delay Time"] = 0.5
    plugin["Feedback"] = 30.0
    plugin["Dry/Wet Mix"] = 80.0

    # Save as user preset with description
    preset_path = plugin.save_preset(
        "My Delay Setting", "500ms delay with light feedback"
    )
    print(f"Saved to: {preset_path}")

    # List all user presets
    user_presets = plugin.list_user_presets()
    print(f"User presets: {user_presets}")

    # Load user preset
    plugin.load_preset("My Delay Setting")
# --8<-- [end:example]
