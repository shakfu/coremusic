#!/usr/bin/env python3
"""List factory presets for an AudioUnit.

Usage:
    python factory_presets.py [plugin_name]
"""

import sys
import coremusic.capi as capi


def main():
    plugin_name = sys.argv[1] if len(sys.argv) > 1 else "AUReverb"

    # Find plugin
    for type_code in ['aufx', 'aumu', 'augn']:
        components = capi.audio_unit_find_all_components(component_type=type_code)
        for comp_id in components:
            try:
                info = capi.audio_unit_get_component_info(comp_id)
                if plugin_name.lower() in info['name'].lower():
                    print(f"Plugin: {info['name']}")

                    unit_id = capi.audio_component_instance_new(comp_id)
                    capi.audio_unit_initialize(unit_id)

                    try:
                        presets = capi.audio_unit_get_factory_presets(unit_id)
                        print(f"Factory Presets: {len(presets)}")
                        for preset in presets[:10]:
                            print(f"  [{preset['number']}] {preset['name']}")
                        if len(presets) > 10:
                            print(f"  ... and {len(presets) - 10} more")
                    finally:
                        capi.audio_unit_uninitialize(unit_id)
                        capi.audio_component_instance_dispose(unit_id)
                    return
            except Exception:
                pass

    print(f"Plugin not found: {plugin_name}")


if __name__ == "__main__":
    main()
