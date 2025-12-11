"""Audio device commands."""

from __future__ import annotations

import argparse

from ._formatters import output_json, output_table
from ._utils import EXIT_SUCCESS, DeviceNotFoundError


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register device commands."""
    parser = subparsers.add_parser("device", help="Audio device management")
    device_sub = parser.add_subparsers(dest="device_command", metavar="<subcommand>")

    # device list
    list_parser = device_sub.add_parser("list", help="List available audio devices")
    list_parser.set_defaults(func=cmd_list)

    # device default
    default_parser = device_sub.add_parser("default", help="Show default audio devices")
    default_parser.add_argument("--input", dest="input_device", action="store_true",
                                help="Show default input device only")
    default_parser.add_argument("--output", dest="output_device", action="store_true",
                                help="Show default output device only")
    default_parser.set_defaults(func=cmd_default)

    # device info
    info_parser = device_sub.add_parser("info", help="Show detailed device information")
    info_parser.add_argument("device", help="Device name or UID")
    info_parser.set_defaults(func=cmd_info)

    # device volume
    volume_parser = device_sub.add_parser("volume", help="Get or set device volume")
    volume_parser.add_argument("device", help="Device name or UID")
    volume_parser.add_argument("level", nargs="?", type=float, default=None,
                               help="Volume level (0.0-1.0) to set, or omit to get current")
    volume_parser.add_argument("--scope", "-s", choices=["input", "output"], default="output",
                               help="Scope: input or output (default: output)")
    volume_parser.add_argument("--channel", "-c", type=int, default=0,
                               help="Channel index (default: 0 for main)")
    volume_parser.set_defaults(func=cmd_volume)

    # device set-default
    setdefault_parser = device_sub.add_parser("set-default", help="Set default audio device")
    setdefault_parser.add_argument("device", help="Device name or UID")
    setdefault_parser.add_argument("--input", dest="input_device", action="store_true",
                                   help="Set as default input device")
    setdefault_parser.add_argument("--output", dest="output_device", action="store_true",
                                   help="Set as default output device")
    setdefault_parser.set_defaults(func=cmd_set_default)

    # device mute
    mute_parser = device_sub.add_parser("mute", help="Get or set device mute state")
    mute_parser.add_argument("device", help="Device name or UID")
    mute_parser.add_argument("state", nargs="?", choices=["on", "off"], default=None,
                             help="Mute state: on or off, or omit to get current")
    mute_parser.add_argument("--scope", "-s", choices=["input", "output"], default="output",
                             help="Scope: input or output (default: output)")
    mute_parser.add_argument("--channel", "-c", type=int, default=0,
                             help="Channel index (default: 0 for main)")
    mute_parser.set_defaults(func=cmd_mute)

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


def _find_device(device_name: str):
    """Find device by name or UID."""
    import coremusic as cm

    all_devices = cm.AudioDeviceManager.get_devices()
    for d in all_devices:
        if d.name == device_name or d.uid == device_name:
            return d

    # Try case-insensitive match
    device_lower = device_name.lower()
    for d in all_devices:
        if d.name and d.name.lower() == device_lower:
            return d

    raise DeviceNotFoundError(f"Device not found: {device_name}")


def cmd_volume(args: argparse.Namespace) -> int:
    """Get or set device volume."""
    import coremusic as cm

    device = _find_device(args.device)

    if args.level is not None:
        # Set volume
        if not 0.0 <= args.level <= 1.0:
            from ._utils import CLIError
            raise CLIError("Volume level must be between 0.0 and 1.0")

        try:
            device.set_volume(args.level, scope=args.scope, channel=args.channel)
        except cm.AudioDeviceError as e:
            from ._utils import CLIError
            raise CLIError(str(e))

        if args.json:
            output_json({
                "device": device.name,
                "volume": args.level,
                "scope": args.scope,
                "channel": args.channel,
                "action": "set",
            })
        else:
            print(f"Volume set to {args.level:.0%} on {device.name}")
    else:
        # Get volume
        volume = device.get_volume(scope=args.scope, channel=args.channel)

        if args.json:
            output_json({
                "device": device.name,
                "volume": volume,
                "scope": args.scope,
                "channel": args.channel,
            })
        else:
            if volume is not None:
                print(f"{device.name}: {volume:.0%}")
            else:
                print(f"{device.name}: Volume not available for this device/channel")

    return EXIT_SUCCESS


def cmd_set_default(args: argparse.Namespace) -> int:
    """Set default audio device."""
    import coremusic as cm

    device = _find_device(args.device)

    # If neither specified, default to output
    set_output = args.output_device
    set_input = args.input_device
    if not set_output and not set_input:
        set_output = True

    results = {}

    if set_output:
        try:
            cm.AudioDeviceManager.set_default_output_device(device)
            results["output"] = {"success": True, "device": device.name}
            if not args.json:
                print(f"Default output device set to: {device.name}")
        except cm.AudioDeviceError as e:
            results["output"] = {"success": False, "error": str(e)}
            if not args.json:
                print(f"Failed to set default output: {e}")

    if set_input:
        try:
            cm.AudioDeviceManager.set_default_input_device(device)
            results["input"] = {"success": True, "device": device.name}
            if not args.json:
                print(f"Default input device set to: {device.name}")
        except cm.AudioDeviceError as e:
            results["input"] = {"success": False, "error": str(e)}
            if not args.json:
                print(f"Failed to set default input: {e}")

    if args.json:
        output_json(results)

    return EXIT_SUCCESS


def cmd_mute(args: argparse.Namespace) -> int:
    """Get or set device mute state."""
    import coremusic as cm

    device = _find_device(args.device)

    if args.state is not None:
        # Set mute state
        muted = args.state == "on"
        try:
            device.set_mute(muted, scope=args.scope, channel=args.channel)
        except cm.AudioDeviceError as e:
            from ._utils import CLIError
            raise CLIError(str(e))

        if args.json:
            output_json({
                "device": device.name,
                "muted": muted,
                "scope": args.scope,
                "channel": args.channel,
                "action": "set",
            })
        else:
            state_str = "muted" if muted else "unmuted"
            print(f"{device.name}: {state_str}")
    else:
        # Get mute state
        muted = device.get_mute(scope=args.scope, channel=args.channel)

        if args.json:
            output_json({
                "device": device.name,
                "muted": muted,
                "scope": args.scope,
                "channel": args.channel,
            })
        else:
            if muted is not None:
                state_str = "muted" if muted else "not muted"
                print(f"{device.name}: {state_str}")
            else:
                print(f"{device.name}: Mute state not available for this device/channel")

    return EXIT_SUCCESS
