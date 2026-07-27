#!/usr/bin/env python3
"""List the AudioUnits installed on the system."""

# --8<-- [start:example]
from coremusic.audio import get_audiounit_names, list_available_audio_units

# List all AudioUnits
units = list_available_audio_units()
for unit in units[:10]:
    print(f"{unit['name']} ({unit['type']})")

# List effect names only
effects = get_audiounit_names(filter_type='aufx')
print(f"{len(effects)} effects installed")
# --8<-- [end:example]
