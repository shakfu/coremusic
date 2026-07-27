#!/usr/bin/env python3
"""Disposing plugins."""

input_data = bytes(512 * 2 * 4)

# --8<-- [start:example]
from coremusic.audio.audiounit_host import AudioUnitPlugin

# Good: automatic cleanup. The context manager instantiates and initializes
# the plugin on entry, and disposes it on exit.
with AudioUnitPlugin.from_name("AUDelay") as plugin:
    output = plugin.process(input_data)

# Also fine, but you own every step
plugin = AudioUnitPlugin.from_name("AUDelay")
try:
    plugin.instantiate().initialize()
    output = plugin.process(input_data)
finally:
    plugin.dispose()
# --8<-- [end:example]
