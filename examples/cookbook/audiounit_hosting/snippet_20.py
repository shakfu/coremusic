#!/usr/bin/env python3
"""Handling a plugin that is not installed."""

input_data = bytes(512 * 2 * 4)

# --8<-- [start:example]
from coremusic.audio.audiounit_host import AudioUnitHost, AudioUnitPlugin

# from_name raises ValueError when nothing matches
try:
    with AudioUnitPlugin.from_name("NonExistentPlugin") as plugin:
        pass
except ValueError as e:
    print(f"Plugin not found: {e}")

# Or check before loading
host = AudioUnitHost()
effects = host.discover_plugins(type='effect')
plugin_names = [p['name'] for p in effects]

if any("AUDelay" in name for name in plugin_names):
    with AudioUnitPlugin.from_name("AUDelay") as plugin:
        output = plugin.process(input_data)
# --8<-- [end:example]
