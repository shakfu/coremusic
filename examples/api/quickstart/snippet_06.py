#!/usr/bin/env python3
"""Finding a component and instantiating it."""

# --8<-- [start:example]
from coremusic.audio import AudioComponent, AudioComponentDescription

# Create component description
desc = AudioComponentDescription(
    type='aufx',          # Effect
    subtype='dely',       # Delay
    manufacturer='appl',
)

# Find component
component = AudioComponent.find_next(desc)
if component:
    unit = component.create_instance()
    unit.initialize()
    # ... use the unit ...
    unit.dispose()
# --8<-- [end:example]
