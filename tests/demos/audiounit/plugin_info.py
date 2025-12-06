#!/usr/bin/env python3
"""Show detailed information about an AudioUnit plugin.

Usage:
    python plugin_info.py <plugin_name>

Example:
    python plugin_info.py AUDelay
"""

import sys
import coremusic.capi as capi


def find_plugin(name):
    """Find plugin by name (partial match)."""
    for type_code in ["aufx", "aumu", "augn", "aumx", "auou"]:
        components = capi.audio_unit_find_all_components(component_type=type_code)
        for comp_id in components:
            try:
                info = capi.audio_unit_get_component_info(comp_id)
                if name.lower() in info["name"].lower():
                    return comp_id, info
            except Exception:
                pass
    return None, None


def main():
    if len(sys.argv) < 2:
        print("Usage: python plugin_info.py <plugin_name>")
        sys.exit(1)

    name = sys.argv[1]
    comp_id, info = find_plugin(name)

    if not comp_id:
        print(f"Plugin not found: {name}")
        sys.exit(1)

    print(f"Plugin: {info['name']}")
    print(f"  Type: {info['type']}")
    print(f"  Subtype: {info['subtype']}")
    print(f"  Manufacturer: {info['manufacturer']}")
    print(f"  Version: {info['version']}")

    # Instantiate to get parameters
    unit_id = capi.audio_component_instance_new(comp_id)
    capi.audio_unit_initialize(unit_id)

    try:
        params = capi.audio_unit_get_parameter_list(unit_id)
        print(f"\nParameters ({len(params)}):")

        for param_id in params[:20]:
            try:
                param_info = capi.audio_unit_get_parameter_info(unit_id, param_id)
                value = capi.audio_unit_get_parameter(unit_id, param_id)
                print(f"  {param_info['name']}: {value:.3f}")
                print(f"    Range: {param_info['min_value']:.3f} - {param_info['max_value']:.3f}")
            except Exception:
                pass

        if len(params) > 20:
            print(f"  ... and {len(params) - 20} more")

        presets = capi.audio_unit_get_factory_presets(unit_id)
        if presets:
            print(f"\nFactory Presets ({len(presets)}):")
            for preset in presets[:10]:
                print(f"  [{preset['number']}] {preset['name']}")
            if len(presets) > 10:
                print(f"  ... and {len(presets) - 10} more")

    finally:
        capi.audio_unit_uninitialize(unit_id)
        capi.audio_component_instance_dispose(unit_id)


if __name__ == "__main__":
    main()
