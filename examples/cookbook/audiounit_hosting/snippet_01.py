#!/usr/bin/env python3
"""List Available Plugins."""

# --8<-- [start:example]
from coremusic.audio.audiounit_host import AudioUnitHost

# Create host
host = AudioUnitHost()

# Discover all effect plugins
effects = host.discover_plugins(type="effect")
print(f"Found {len(effects)} effect plugins")

for plugin_info in effects[:10]:
    print(f"  - {plugin_info['name']} ({plugin_info['manufacturer']})")

# Discover instrument plugins
instruments = host.discover_plugins(type="instrument")
print(f"\nFound {len(instruments)} instrument plugins")

# Discover by manufacturer
apple_plugins = host.discover_plugins(manufacturer="appl")
print(f"\nFound {len(apple_plugins)} Apple plugins")
# --8<-- [end:example]
