#!/usr/bin/env python3
"""Load Plugin by Name."""

# --8<-- [start:example]
from coremusic.audio.audiounit_host import AudioUnitPlugin

# Load plugin using context manager (automatic cleanup)
with AudioUnitPlugin.from_name("AUDelay") as plugin:
    print(f"Loaded: {plugin.name}")
    print(f"Manufacturer: {plugin.manufacturer}")
    print(f"Version: {plugin.version}")

    # Plugin is automatically disposed when exiting context
# --8<-- [end:example]
