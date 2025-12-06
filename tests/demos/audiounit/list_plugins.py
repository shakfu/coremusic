#!/usr/bin/env python3
"""List all AudioUnit plugins by category.

Usage:
    python list_plugins.py
"""

import coremusic.capi as capi


def main():
    categories = {
        "Effects": "aufx",
        "Instruments": "aumu",
        "Generators": "augn",
        "Mixers": "aumx",
        "Output": "auou",
    }

    total = 0

    for name, type_code in categories.items():
        components = capi.audio_unit_find_all_components(component_type=type_code)
        count = len(components)
        total += count

        print(f"\n{name} ({count}):")
        for comp_id in components[:10]:
            try:
                info = capi.audio_unit_get_component_info(comp_id)
                print(f"  {info['name']} ({info['manufacturer']})")
            except Exception:
                pass

        if count > 10:
            print(f"  ... and {count - 10} more")

    print(f"\nTotal: {total} plugins")


if __name__ == "__main__":
    main()
