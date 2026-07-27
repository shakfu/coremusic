#!/usr/bin/env python3
"""List and Control Parameters."""

# --8<-- [start:example]
from coremusic.audio.audiounit_host import AudioUnitPlugin

with AudioUnitPlugin.from_name("AUDelay") as plugin:
    # List all parameters
    print(f"Parameters ({len(plugin.parameters)}):")
    for param in plugin.parameters:
        print(f"  - {param.name}: {param.value} {param.unit_name}")
        print(f"    Range: [{param.min_value}, {param.max_value}], Default: {param.default_value}")

    # Set parameter by name
    plugin.set_parameter("Delay Time", 0.5)
    plugin.set_parameter("Feedback", 30.0)
    plugin.set_parameter("Dry/Wet Mix", 100.0)

    # Or use dictionary-style access
    plugin['Delay Time'] = 0.25
    current_delay = plugin['Delay Time']
    print(f"Current delay: {current_delay}")
# --8<-- [end:example]
