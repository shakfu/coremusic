"""Installation diagnostics for the coremusic CLI.

The ``doctor`` command reports the state of the environment coremusic depends
on: Python and OS, optional analysis/visualization packages, audio hardware
access, and CoreMIDI/AudioUnit availability. It is the first thing to run when
something does not work.
"""

from __future__ import annotations

import argparse
import platform
import sys

from ._formatters import output_json
from ._utils import EXIT_SUCCESS


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the doctor command."""
    parser = subparsers.add_parser(
        "doctor", help="Diagnose the coremusic installation and environment"
    )
    parser.set_defaults(func=cmd_doctor)


def _check_optional_dependencies() -> dict[str, bool]:
    """Report which optional dependencies are importable."""
    results: dict[str, bool] = {}
    for name in ("numpy", "scipy", "matplotlib"):
        try:
            __import__(name)
            results[name] = True
        except Exception:
            results[name] = False
    return results


def _check_audio() -> dict[str, object]:
    """Report audio hardware access via CoreAudio."""
    info: dict[str, object] = {}
    try:
        from coremusic.audio.devices import AudioDeviceManager

        devices = AudioDeviceManager.get_devices()
        info["accessible"] = True
        info["device_count"] = len(devices)
        info["input_devices"] = len(AudioDeviceManager.get_input_devices())
        info["output_devices"] = len(AudioDeviceManager.get_output_devices())
        default_out = AudioDeviceManager.get_default_output_device()
        default_in = AudioDeviceManager.get_default_input_device()
        info["default_output"] = default_out.name if default_out else None
        info["default_input"] = default_in.name if default_in else None
    except Exception as e:
        info["accessible"] = False
        info["error"] = str(e)
    return info


def _check_plugins() -> dict[str, object]:
    """Report AudioUnit plugin availability."""
    info: dict[str, object] = {}
    try:
        from coremusic.shortcuts import list_plugins

        info["accessible"] = True
        info["total"] = len(list_plugins())
        info["effects"] = len(list_plugins(type="effect"))
        info["instruments"] = len(list_plugins(type="instrument"))
    except Exception as e:
        info["accessible"] = False
        info["error"] = str(e)
    return info


def _check_midi() -> dict[str, object]:
    """Report CoreMIDI endpoint availability."""
    info: dict[str, object] = {}
    try:
        from coremusic import capi

        info["accessible"] = True
        info["sources"] = int(capi.midi_get_number_of_sources())
        info["destinations"] = int(capi.midi_get_number_of_destinations())
    except Exception as e:
        info["accessible"] = False
        info["error"] = str(e)
    return info


def _gather_report() -> dict[str, object]:
    from coremusic import __version__

    return {
        "coremusic_version": __version__,
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "macos_version": platform.mac_ver()[0] or None,
        "is_macos": sys.platform == "darwin",
        "optional_dependencies": _check_optional_dependencies(),
        "audio": _check_audio(),
        "plugins": _check_plugins(),
        "midi": _check_midi(),
    }


def _status(ok: bool) -> str:
    return "ok" if ok else "FAIL"


def cmd_doctor(args: argparse.Namespace) -> int:
    """Diagnose the coremusic installation."""
    report = _gather_report()

    if args.json:
        output_json(report)
        return EXIT_SUCCESS

    print("coremusic doctor")
    print("=" * 40)
    print(f"coremusic version: {report['coremusic_version']}")
    print(f"Python:            {report['python_version']}")
    macos = report["macos_version"] or "unknown"
    platform_ok = bool(report["is_macos"])
    print(f"Platform:          {report['platform']} (macOS {macos}) [{_status(platform_ok)}]")
    if not platform_ok:
        print("  coremusic requires macOS; CoreAudio/CoreMIDI are unavailable here.")

    print("\nOptional dependencies:")
    deps = report["optional_dependencies"]
    assert isinstance(deps, dict)
    for name, present in deps.items():
        hint = "" if present else "  (pip install coremusic[all])"
        print(f"  {name:12s} [{_status(present)}]{hint}")

    audio = report["audio"]
    assert isinstance(audio, dict)
    print(f"\nAudio hardware:      [{_status(bool(audio.get('accessible')))}]")
    if audio.get("accessible"):
        print(f"  Devices:           {audio.get('device_count')}")
        print(f"  Inputs / Outputs:  {audio.get('input_devices')} / {audio.get('output_devices')}")
        print(f"  Default output:    {audio.get('default_output')}")
        print(f"  Default input:     {audio.get('default_input')}")
    else:
        print(f"  {audio.get('error')}")

    plugins = report["plugins"]
    assert isinstance(plugins, dict)
    print(f"\nAudioUnit plugins:   [{_status(bool(plugins.get('accessible')))}]")
    if plugins.get("accessible"):
        print(f"  Total:             {plugins.get('total')}")
        print(f"  Effects / Instr.:  {plugins.get('effects')} / {plugins.get('instruments')}")
    else:
        print(f"  {plugins.get('error')}")

    midi = report["midi"]
    assert isinstance(midi, dict)
    print(f"\nCoreMIDI:            [{_status(bool(midi.get('accessible')))}]")
    if midi.get("accessible"):
        print(f"  Sources:           {midi.get('sources')}")
        print(f"  Destinations:      {midi.get('destinations')}")
    else:
        print(f"  {midi.get('error')}")

    return EXIT_SUCCESS
