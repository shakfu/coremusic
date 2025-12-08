"""Audio device commands."""

from __future__ import annotations

import argparse

from ._formatters import output_json, output_table
from ._utils import EXIT_SUCCESS, DeviceNotFoundError


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register devices commands."""
    parser = subparsers.add_parser("devices", help="Audio device management")
    devices_sub = parser.add_subparsers(dest="devices_command", metavar="<subcommand>")

    # devices list
    list_parser = devices_sub.add_parser("list", help="List available audio devices")
    list_parser.set_defaults(func=cmd_list)

    # devices default
    default_parser = devices_sub.add_parser("default", help="Show default audio devices")
    default_parser.add_argument("--input", dest="input_device", action="store_true",
                                help="Show default input device only")
    default_parser.add_argument("--output", dest="output_device", action="store_true",
                                help="Show default output device only")
    default_parser.set_defaults(func=cmd_default)

    # devices info
    info_parser = devices_sub.add_parser("info", help="Show detailed device information")
    info_parser.add_argument("device", help="Device name or UID")
    info_parser.set_defaults(func=cmd_info)

    parser.set_defaults(func=lambda args: parser.print_help() or EXIT_SUCCESS)


def cmd_list(args: argparse.Namespace) -> int:
    """List available audio devices."""
    import coremusic as cm

    devices = cm.AudioDeviceManager.get_devices()

    if args.json:
        data = [
            {
                "name": d.name,
                "uid": d.uid,
                "manufacturer": d.manufacturer,
                "sample_rate": d.sample_rate,
                "is_alive": d.is_alive,
            }
            for d in devices
        ]
        output_json(data)
    else:
        if not devices:
            print("No audio devices found.")
            return EXIT_SUCCESS

        headers = ["Name", "Manufacturer", "Sample Rate"]
        rows = []
        for d in devices:
            rows.append([
                d.name or "(unknown)",
                d.manufacturer or "(unknown)",
                f"{d.sample_rate:.0f} Hz" if d.sample_rate else "N/A",
            ])

        output_table(headers, rows)

    return EXIT_SUCCESS


def cmd_default(args: argparse.Namespace) -> int:
    """Show default audio devices."""
    from typing import Any, Dict, Optional

    import coremusic as cm

    result: Dict[str, Optional[Dict[str, Any]]] = {}
    show_input = args.input_device
    show_output = args.output_device

    # Show both if neither specified
    if not show_input and not show_output:
        show_input = show_output = True

    if show_output:
        try:
            output_dev = cm.AudioDeviceManager.get_default_output_device()
            if output_dev:
                result["output"] = {
                    "name": output_dev.name,
                    "uid": output_dev.uid,
                    "sample_rate": output_dev.sample_rate,
                }
            else:
                result["output"] = None
        except (cm.AudioDeviceError, Exception):
            result["output"] = None

    if show_input:
        try:
            input_dev = cm.AudioDeviceManager.get_default_input_device()
            if input_dev:
                result["input"] = {
                    "name": input_dev.name,
                    "uid": input_dev.uid,
                    "sample_rate": input_dev.sample_rate,
                }
            else:
                result["input"] = None
        except (cm.AudioDeviceError, Exception):
            result["input"] = None

    if args.json:
        output_json(result)
    else:
        if "output" in result:
            if result["output"]:
                print(f"Default Output: {result['output']['name']}")
            else:
                print("Default Output: (none)")
        if "input" in result:
            if result["input"]:
                print(f"Default Input:  {result['input']['name']}")
            else:
                print("Default Input:  (none)")

    return EXIT_SUCCESS


def cmd_info(args: argparse.Namespace) -> int:
    """Show detailed device information."""
    import coremusic as cm

    # Find device by name or UID
    all_devices = cm.AudioDeviceManager.get_devices()
    found = None

    for d in all_devices:
        if d.name == args.device or d.uid == args.device:
            found = d
            break

    if not found:
        raise DeviceNotFoundError(f"Device not found: {args.device}")

    if args.json:
        data = {
            "name": found.name,
            "uid": found.uid,
            "manufacturer": found.manufacturer,
            "sample_rate": found.sample_rate,
            "is_alive": found.is_alive,
            "is_hidden": found.is_hidden,
            "transport_type": found.transport_type,
        }
        output_json(data)
    else:
        print(f"Name:         {found.name}")
        print(f"UID:          {found.uid}")
        print(f"Manufacturer: {found.manufacturer}")
        print(f"Sample Rate:  {found.sample_rate:.0f} Hz")
        print(f"Is Alive:     {'Yes' if found.is_alive else 'No'}")
        print(f"Is Hidden:    {'Yes' if found.is_hidden else 'No'}")

    return EXIT_SUCCESS
