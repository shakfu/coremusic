"""AudioUnit plugin commands."""

from __future__ import annotations

import argparse

from ._formatters import output_json, output_table
from ._mappings import PLUGIN_TYPES, get_plugin_type, get_plugin_type_display
from ._utils import EXIT_SUCCESS, CLIError

# Parameter unit display names (kAudioUnitParameterUnit_* values)
# From constants.py / audiotoolbox.pxd
PARAM_UNIT_NAMES = {
    0: "",  # kAudioUnitParameterUnit_Generic
    1: "indexed",  # kAudioUnitParameterUnit_Indexed
    2: "bool",  # kAudioUnitParameterUnit_Boolean
    3: "%",  # kAudioUnitParameterUnit_Percent
    4: "sec",  # kAudioUnitParameterUnit_Seconds
    5: "samples",  # kAudioUnitParameterUnit_SampleFrames
    6: "phase",  # kAudioUnitParameterUnit_Phase
    7: "rate",  # kAudioUnitParameterUnit_Rate
    8: "Hz",  # kAudioUnitParameterUnit_Hertz
    9: "cents",  # kAudioUnitParameterUnit_Cents
    10: "rel. semitones",  # kAudioUnitParameterUnit_RelativeSemiTones
    11: "MIDI note",  # kAudioUnitParameterUnit_MIDINoteNumber
    12: "MIDI controller",  # kAudioUnitParameterUnit_MIDIController
    13: "dB",  # kAudioUnitParameterUnit_Decibels
    14: "linear gain",  # kAudioUnitParameterUnit_LinearGain
    15: "degrees",  # kAudioUnitParameterUnit_Degrees
    16: "crossfade",  # kAudioUnitParameterUnit_EqualPowerCrossfade
    17: "fader",  # kAudioUnitParameterUnit_MixerFaderCurve1
    18: "pan",  # kAudioUnitParameterUnit_Pan
    19: "meters",  # kAudioUnitParameterUnit_Meters
    20: "abs. cents",  # kAudioUnitParameterUnit_AbsoluteCents
    21: "oct",  # kAudioUnitParameterUnit_Octaves
    22: "BPM",  # kAudioUnitParameterUnit_BPM
    23: "beats",  # kAudioUnitParameterUnit_Beats
    24: "ms",  # kAudioUnitParameterUnit_Milliseconds
    25: "ratio",  # kAudioUnitParameterUnit_Ratio
    26: "custom",  # kAudioUnitParameterUnit_CustomUnit
}


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register plugins commands."""
    parser = subparsers.add_parser("plugins", help="AudioUnit plugin discovery")
    plugins_sub = parser.add_subparsers(dest="plugins_command", metavar="<subcommand>")

    # plugins list
    list_parser = plugins_sub.add_parser("list", help="List available AudioUnit plugins")
    list_parser.add_argument("--type", dest="plugin_type", choices=list(PLUGIN_TYPES.keys()),
                             help="Filter by plugin type")
    list_parser.add_argument("--manufacturer", help="Filter by manufacturer")
    list_parser.set_defaults(func=cmd_list)

    # plugins find
    find_parser = plugins_sub.add_parser("find", help="Search for plugins by name")
    find_parser.add_argument("query", help="Search query")
    find_parser.add_argument("--type", dest="plugin_type", choices=list(PLUGIN_TYPES.keys()),
                             help="Filter by plugin type")
    find_parser.set_defaults(func=cmd_find)

    # plugins info
    info_parser = plugins_sub.add_parser("info", help="Show detailed plugin information")
    info_parser.add_argument("name", help="Plugin name (partial match)")
    info_parser.set_defaults(func=cmd_info)

    # plugins params
    params_parser = plugins_sub.add_parser("params", help="List plugin parameters")
    params_parser.add_argument("name", help="Plugin name (partial match)")
    params_parser.set_defaults(func=cmd_params)

    parser.set_defaults(func=lambda args: parser.print_help() or EXIT_SUCCESS)


def _get_all_plugins(type_code: str | None = None) -> list[dict]:
    """Get all plugins with their info."""
    import coremusic.capi as capi

    component_ids = capi.audio_unit_find_all_components(type_code)
    components = []
    for comp_id in component_ids:
        try:
            info = capi.audio_unit_get_component_info(comp_id)
            components.append(info)
        except RuntimeError:
            # Skip components that fail to get info
            pass
    return components


def cmd_list(args: argparse.Namespace) -> int:
    """List available AudioUnit plugins."""
    # Build component description for filtering
    type_code = get_plugin_type(args.plugin_type) if args.plugin_type else None

    # Get all components
    components = _get_all_plugins(type_code)

    # Filter by manufacturer if specified
    if args.manufacturer:
        manufacturer_lower = args.manufacturer.lower()
        components = [c for c in components if manufacturer_lower in c.get("manufacturer", "").lower()]

    if args.json:
        output_json(components)
    else:
        if not components:
            print("No plugins found.")
            return EXIT_SUCCESS

        headers = ["Name", "Type", "Manufacturer"]
        rows = []
        for c in components:
            rows.append([
                c.get("name", "Unknown"),
                get_plugin_type_display(c.get("type", "")),
                c.get("manufacturer", "Unknown"),
            ])

        print(f"Found {len(components)} plugins:\n")
        output_table(headers, rows)

    return EXIT_SUCCESS


def cmd_find(args: argparse.Namespace) -> int:
    """Search for plugins by name."""
    type_code = get_plugin_type(args.plugin_type) if args.plugin_type else None
    components = _get_all_plugins(type_code)

    # Filter by query
    query_lower = args.query.lower()
    matches = [c for c in components if query_lower in c.get("name", "").lower()]

    if args.json:
        output_json(matches)
    else:
        if not matches:
            print(f"No plugins matching '{args.query}' found.")
            return EXIT_SUCCESS

        print(f"Found {len(matches)} plugins matching '{args.query}':\n")
        for c in matches:
            plugin_type_display = get_plugin_type_display(c.get("type", ""))
            print(f"  {c.get('name')} ({plugin_type_display}) - {c.get('manufacturer')}")

    return EXIT_SUCCESS


def _find_plugin_by_name(name: str) -> dict:
    """Find a plugin by name (partial match, returns first match)."""
    components = _get_all_plugins()
    name_lower = name.lower()

    for c in components:
        if name_lower in c.get("name", "").lower():
            return c

    raise CLIError(f"Plugin not found: {name}")


def _get_component_id_for_plugin(plugin_info: dict) -> int:
    """Get the component ID for a plugin by searching for it."""
    import coremusic.capi as capi

    # Search all components to find matching one
    component_ids = capi.audio_unit_find_all_components(None)
    for comp_id in component_ids:
        try:
            info = capi.audio_unit_get_component_info(comp_id)
            if (info.get("name") == plugin_info.get("name") and
                info.get("manufacturer") == plugin_info.get("manufacturer")):
                return comp_id
        except RuntimeError:
            pass

    raise CLIError(f"Could not find component for plugin: {plugin_info.get('name')}")


def cmd_info(args: argparse.Namespace) -> int:
    """Show detailed plugin information."""
    import coremusic.capi as capi

    plugin = _find_plugin_by_name(args.name)

    if args.json:
        output_json(plugin)
    else:
        print(f"Name:         {plugin.get('name', 'Unknown')}")
        print(f"Type:         {get_plugin_type_display(plugin.get('type', ''))}")
        print(f"Subtype:      {plugin.get('subtype', 'Unknown')}")
        print(f"Manufacturer: {plugin.get('manufacturer', 'Unknown')}")

        # Try to get parameter count
        try:
            comp_id = _get_component_id_for_plugin(plugin)
            au_id = capi.audio_component_instance_new(comp_id)
            try:
                capi.audio_unit_initialize(au_id)
                param_ids = capi.audio_unit_get_parameter_list(au_id, scope=0)
                print(f"Parameters:   {len(param_ids)}")
            finally:
                capi.audio_unit_uninitialize(au_id)
                capi.audio_component_instance_dispose(au_id)
        except Exception:
            print("Parameters:   (unable to query)")

    return EXIT_SUCCESS


def cmd_params(args: argparse.Namespace) -> int:
    """List plugin parameters."""
    import coremusic.capi as capi

    plugin = _find_plugin_by_name(args.name)
    comp_id = _get_component_id_for_plugin(plugin)

    # Create AudioUnit instance to query parameters
    au_id = capi.audio_component_instance_new(comp_id)
    try:
        capi.audio_unit_initialize(au_id)

        # Get parameter list
        param_ids = capi.audio_unit_get_parameter_list(au_id, scope=0)

        params = []
        for param_id in param_ids:
            try:
                info = capi.audio_unit_get_parameter_info(au_id, param_id, scope=0)
                value = capi.audio_unit_get_parameter(au_id, param_id, scope=0)
                params.append({
                    "id": param_id,
                    "name": info.get("name", f"Param {param_id}"),
                    "value": value,
                    "min": info.get("min_value", 0),
                    "max": info.get("max_value", 1),
                    "default": info.get("default_value", 0),
                    "unit": info.get("unit", 0),
                    "unit_name": PARAM_UNIT_NAMES.get(info.get("unit", 0), ""),
                })
            except Exception:
                # Skip parameters that fail to query
                pass

        if args.json:
            output_json({
                "plugin": plugin.get("name"),
                "parameters": params,
            })
        else:
            if not params:
                print(f"No parameters found for '{plugin.get('name')}'")
                return EXIT_SUCCESS

            print(f"Parameters for '{plugin.get('name')}':\n")
            headers = ["ID", "Name", "Value", "Range", "Unit"]
            rows = []
            for p in params:
                unit_str = p["unit_name"] if p["unit_name"] else ""
                rows.append([
                    str(p["id"]),
                    p["name"][:30],  # Truncate long names
                    f"{p['value']:.2f}",
                    f"{p['min']:.1f} - {p['max']:.1f}",
                    unit_str,
                ])
            output_table(headers, rows)

    finally:
        try:
            capi.audio_unit_uninitialize(au_id)
        except Exception:
            pass
        try:
            capi.audio_component_instance_dispose(au_id)
        except Exception:
            pass

    return EXIT_SUCCESS
