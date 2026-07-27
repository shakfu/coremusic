#!/usr/bin/env python3
"""Export and Import Presets."""

# The preset exported below has to exist first
from coremusic.audio.audiounit_host import AudioUnitPlugin as _Plugin

with _Plugin.from_name("AUDelay") as _p:
    _p.save_preset("My Delay Setting", "500ms delay with light feedback")

# --8<-- [start:example]
from coremusic.audio.audiounit_host import AudioUnitPlugin

from pathlib import Path

with AudioUnitPlugin.from_name("AUDelay") as plugin:
    # Export preset to custom location
    export_path = Path("~/Desktop/my_delay.json").expanduser()
    plugin.export_preset("My Delay Setting", export_path)
    print(f"Exported to: {export_path}")

# Import preset (can be on different machine)
with AudioUnitPlugin.from_name("AUDelay") as plugin:
    imported_name = plugin.import_preset(export_path)
    print(f"Imported as: {imported_name}")

    # Load the imported preset
    plugin.load_preset(imported_name)
# --8<-- [end:example]
