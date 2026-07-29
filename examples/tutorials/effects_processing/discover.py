#!/usr/bin/env python3
"""Find the AudioUnits installed on this machine."""

# --8<-- [start:list-all]
from coremusic.audio import list_available_audio_units


def list_all_audio_units():
    """List all available AudioUnits."""
    units = list_available_audio_units()

    print(f"Found {len(units)} AudioUnits:\n")

    # Group by type
    by_type = {}
    for unit in units:
        by_type.setdefault(unit["type"], []).append(unit)

    type_names = {
        "aufx": "Effects",
        "aumu": "Instruments",
        "augn": "Generators",
        "aumx": "Mixers",
        "aufc": "Format Converters",
        "auou": "Output Units",
    }

    for unit_type, units_list in sorted(by_type.items()):
        name = type_names.get(unit_type, unit_type)
        print(f"{name} ({unit_type}): {len(units_list)} plugins")


list_all_audio_units()
# --8<-- [end:list-all]

# --8<-- [start:list-effects]
from coremusic.audio import get_audiounit_names


def list_effects():
    """List only effect AudioUnits."""
    names = get_audiounit_names(filter_type="aufx")

    print("Available Effects:")
    for name in sorted(names)[:10]:
        print(f"  {name}")

    return names


effects = list_effects()
# --8<-- [end:list-effects]

# --8<-- [start:find]
from coremusic.audio.audiounit_host import AudioUnitHost


def find_effect(name):
    """Find an effect by name."""
    host = AudioUnitHost()
    matches = [
        plugin
        for plugin in host.discover_plugins(type="effect")
        if name.lower() in plugin["name"].lower()
    ]

    if not matches:
        print(f"Not found: {name}")
        return None

    plugin = matches[0]
    print(f"Found: {plugin['name']}")
    print(f"  Type: {plugin['type']}")
    print(f"  Subtype: {plugin['subtype']}")
    print(f"  Manufacturer: {plugin['manufacturer']}")
    return plugin


# Find AUDelay
delay = find_effect("AUDelay")

# Find by partial name
reverb = find_effect("Reverb")
# --8<-- [end:find]
