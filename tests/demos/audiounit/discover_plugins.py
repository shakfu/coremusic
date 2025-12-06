#!/usr/bin/env python3
"""Discover AudioUnit plugins using high-level API.

Usage:
    python discover_plugins.py [type]

Where type is: effect, instrument, generator, mixer, output
"""

import sys
import coremusic as cm


def main():
    plugin_type = sys.argv[1] if len(sys.argv) > 1 else "effect"

    host = cm.AudioUnitHost()
    counts = host.get_plugin_count()

    print(f"Plugin counts: {counts}")

    plugins = host.discover_plugins(type=plugin_type)
    print(f"\n{plugin_type.capitalize()} plugins ({len(plugins)}):")
    for plugin in plugins[:15]:
        print(f"  {plugin['name']} ({plugin['manufacturer']})")
    if len(plugins) > 15:
        print(f"  ... and {len(plugins) - 15} more")


if __name__ == "__main__":
    main()
