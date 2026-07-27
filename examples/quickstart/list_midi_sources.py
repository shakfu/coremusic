#!/usr/bin/env python3
"""List the MIDI sources visible to this process."""

# --8<-- [start:example]
from coremusic.midi import get_sources

for i, source in enumerate(get_sources()):
    print(f"Source {i}: {source.name}")
# --8<-- [end:example]
