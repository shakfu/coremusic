#!/usr/bin/env python3
"""Control AudioUnit parameters.

Usage:
    python parameter_control.py [plugin_name]
"""

import sys
import coremusic.capi as capi


def main():
    plugin_name = sys.argv[1] if len(sys.argv) > 1 else "AUDelay"

    # Find plugin
    components = capi.audio_unit_find_all_components(component_type='aufx')
    target = None

    for comp_id in components:
        try:
            info = capi.audio_unit_get_component_info(comp_id)
            if plugin_name.lower() in info['name'].lower():
                target = (comp_id, info)
                break
        except Exception:
            pass

    if not target:
        print(f"Plugin not found: {plugin_name}")
        return

    comp_id, info = target
    print(f"Plugin: {info['name']}")

    # Instantiate
    unit_id = capi.audio_component_instance_new(comp_id)
    capi.audio_unit_initialize(unit_id)

    try:
        params = capi.audio_unit_get_parameter_list(unit_id)
        print(f"Parameters: {len(params)}")

        for param_id in params[:5]:
            try:
                param_info = capi.audio_unit_get_parameter_info(unit_id, param_id)
                value = capi.audio_unit_get_parameter(unit_id, param_id)
                print(f"  {param_info['name']}: {value:.3f} [{param_info['min_value']:.1f}-{param_info['max_value']:.1f}]")
            except Exception:
                pass
    finally:
        capi.audio_unit_uninitialize(unit_id)
        capi.audio_component_instance_dispose(unit_id)


if __name__ == "__main__":
    main()
