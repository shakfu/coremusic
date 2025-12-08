"""MIDI device commands."""

from __future__ import annotations

import argparse
from typing import Any, Dict

import coremusic.capi as capi

from ._formatters import format_duration, output_json, output_table
from ._utils import EXIT_SUCCESS, CLIError, require_file


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register MIDI commands."""
    parser = subparsers.add_parser("midi", help="MIDI device discovery and information")
    midi_sub = parser.add_subparsers(dest="midi_command", metavar="<subcommand>")

    # midi devices
    devices_parser = midi_sub.add_parser("devices", help="List all MIDI devices")
    devices_parser.set_defaults(func=cmd_devices)

    # midi inputs (sources)
    inputs_parser = midi_sub.add_parser("inputs", help="List MIDI input sources")
    inputs_parser.set_defaults(func=cmd_inputs)

    # midi outputs (destinations)
    outputs_parser = midi_sub.add_parser("outputs", help="List MIDI output destinations")
    outputs_parser.set_defaults(func=cmd_outputs)

    # midi send
    send_parser = midi_sub.add_parser("send", help="Send MIDI message")
    send_parser.add_argument(
        "--device", "-d", type=int, default=0,
        help="Output device index (default: 0)"
    )
    send_parser.add_argument(
        "--note", "-n", type=int,
        help="Send note on/off (MIDI note number 0-127)"
    )
    send_parser.add_argument(
        "--velocity", "-v", type=int, default=100,
        help="Note velocity (default: 100)"
    )
    send_parser.add_argument(
        "--channel", "-c", type=int, default=0,
        help="MIDI channel 0-15 (default: 0)"
    )
    send_parser.add_argument(
        "--cc", type=int, nargs=2, metavar=("NUM", "VAL"),
        help="Send control change (controller number and value)"
    )
    send_parser.add_argument(
        "--program", "-p", type=int,
        help="Send program change (0-127)"
    )
    send_parser.add_argument(
        "--duration", type=float, default=0.5,
        help="Note duration in seconds (default: 0.5)"
    )
    send_parser.set_defaults(func=cmd_send)

    # midi file (info about MIDI file)
    file_parser = midi_sub.add_parser("file", help="Show MIDI file information")
    file_parser.add_argument("path", help="MIDI file path")
    file_parser.set_defaults(func=cmd_file)

    parser.set_defaults(func=lambda args: parser.print_help() or EXIT_SUCCESS)


def _get_endpoint_name(endpoint_id: int) -> str:
    """Get the display name of a MIDI endpoint."""

    try:
        # Try kMIDIPropertyDisplayName first
        name = capi.midi_object_get_string_property(endpoint_id, "displayName")
        if name:
            return name
    except Exception:
        pass

    try:
        # Fall back to kMIDIPropertyName
        name = capi.midi_object_get_string_property(endpoint_id, "name")
        if name:
            return name
    except Exception:
        pass

    return f"Unknown ({endpoint_id})"


def _get_device_info(device_id: int) -> dict:
    """Get detailed information about a MIDI device."""

    info: Dict[str, Any] = {"id": device_id}

    # Get device properties
    try:
        info["name"] = capi.midi_object_get_string_property(device_id, "name") or "Unknown"
    except Exception:
        info["name"] = "Unknown"

    try:
        info["manufacturer"] = capi.midi_object_get_string_property(device_id, "manufacturer") or ""
    except Exception:
        info["manufacturer"] = ""

    try:
        info["model"] = capi.midi_object_get_string_property(device_id, "model") or ""
    except Exception:
        info["model"] = ""

    # Count entities, sources, destinations
    try:
        num_entities = capi.midi_device_get_number_of_entities(device_id)
        info["entities"] = num_entities

        sources = 0
        destinations = 0
        for i in range(num_entities):
            entity = capi.midi_device_get_entity(device_id, i)
            sources += capi.midi_entity_get_number_of_sources(entity)
            destinations += capi.midi_entity_get_number_of_destinations(entity)

        info["sources"] = sources
        info["destinations"] = destinations
    except Exception:
        info["entities"] = 0
        info["sources"] = 0
        info["destinations"] = 0

    return info


def cmd_devices(args: argparse.Namespace) -> int:
    """List all MIDI devices."""

    num_devices = capi.midi_get_number_of_devices()
    devices = []

    for i in range(num_devices):
        device_id = capi.midi_get_device(i)
        info = _get_device_info(device_id)
        devices.append(info)

    if args.json:
        output_json(devices)
    else:
        if not devices:
            print("No MIDI devices found.")
            return EXIT_SUCCESS

        headers = ["Name", "Manufacturer", "Inputs", "Outputs"]
        rows = []
        for d in devices:
            rows.append([
                d["name"],
                d["manufacturer"] or "(unknown)",
                str(d["sources"]),
                str(d["destinations"]),
            ])

        print(f"Found {len(devices)} MIDI devices:\n")
        output_table(headers, rows)

    return EXIT_SUCCESS


def cmd_inputs(args: argparse.Namespace) -> int:
    """List MIDI input sources."""

    num_sources = capi.midi_get_number_of_sources()
    sources = []

    for i in range(num_sources):
        source_id = capi.midi_get_source(i)
        name = _get_endpoint_name(source_id)
        sources.append({
            "index": i,
            "id": source_id,
            "name": name,
        })

    if args.json:
        output_json(sources)
    else:
        if not sources:
            print("No MIDI input sources found.")
            return EXIT_SUCCESS

        print(f"Found {len(sources)} MIDI inputs:\n")
        for s in sources:
            print(f"  [{s['index']}] {s['name']}")

    return EXIT_SUCCESS


def cmd_outputs(args: argparse.Namespace) -> int:
    """List MIDI output destinations."""

    num_dests = capi.midi_get_number_of_destinations()
    destinations = []

    for i in range(num_dests):
        dest_id = capi.midi_get_destination(i)
        name = _get_endpoint_name(dest_id)
        destinations.append({
            "index": i,
            "id": dest_id,
            "name": name,
        })

    if args.json:
        output_json(destinations)
    else:
        if not destinations:
            print("No MIDI output destinations found.")
            return EXIT_SUCCESS

        print(f"Found {len(destinations)} MIDI outputs:\n")
        for d in destinations:
            print(f"  [{d['index']}] {d['name']}")

    return EXIT_SUCCESS


def cmd_send(args: argparse.Namespace) -> int:
    """Send MIDI message."""
    import time

    # Validate inputs
    if args.note is None and args.cc is None and args.program is None:
        raise CLIError("Must specify --note, --cc, or --program")

    if not 0 <= args.channel <= 15:
        raise CLIError(f"Channel must be 0-15, got {args.channel}")

    # Check for MIDI outputs
    num_dests = capi.midi_get_number_of_destinations()
    if num_dests == 0:
        raise CLIError("No MIDI output destinations available")

    if args.device >= num_dests:
        raise CLIError(f"Device index {args.device} out of range (0-{num_dests-1})")

    dest_id = capi.midi_get_destination(args.device)
    dest_name = _get_endpoint_name(dest_id)

    # Create MIDI client and output port
    client_id = capi.midi_client_create("coremusic-send")
    try:
        port_id = capi.midi_output_port_create(client_id, "output")

        messages_sent = []

        # Send note
        if args.note is not None:
            if not 0 <= args.note <= 127:
                raise CLIError(f"Note must be 0-127, got {args.note}")
            if not 0 <= args.velocity <= 127:
                raise CLIError(f"Velocity must be 0-127, got {args.velocity}")

            # Note on
            note_on_data = bytes([0x90 | args.channel, args.note, args.velocity])
            capi.midi_send(port_id, dest_id, note_on_data, 0)
            messages_sent.append(f"Note On: {args.note} vel={args.velocity}")

            # Wait for duration
            time.sleep(args.duration)

            # Note off
            note_off_data = bytes([0x80 | args.channel, args.note, 0])
            capi.midi_send(port_id, dest_id, note_off_data, 0)
            messages_sent.append(f"Note Off: {args.note}")

        # Send CC
        if args.cc is not None:
            cc_num, cc_val = args.cc
            if not 0 <= cc_num <= 127:
                raise CLIError(f"CC number must be 0-127, got {cc_num}")
            if not 0 <= cc_val <= 127:
                raise CLIError(f"CC value must be 0-127, got {cc_val}")

            cc_data = bytes([0xB0 | args.channel, cc_num, cc_val])
            capi.midi_send(port_id, dest_id, cc_data, 0)
            messages_sent.append(f"CC: {cc_num}={cc_val}")

        # Send program change
        if args.program is not None:
            if not 0 <= args.program <= 127:
                raise CLIError(f"Program must be 0-127, got {args.program}")

            program_data = bytes([0xC0 | args.channel, args.program])
            capi.midi_send(port_id, dest_id, program_data, 0)
            messages_sent.append(f"Program: {args.program}")

        if args.json:
            output_json({
                "device": dest_name,
                "channel": args.channel,
                "messages": messages_sent,
            })
        else:
            print(f"Sent to {dest_name} (ch {args.channel}):")
            for msg in messages_sent:
                print(f"  {msg}")

    finally:
        try:
            capi.midi_client_dispose(client_id)
        except Exception:
            pass

    return EXIT_SUCCESS


def cmd_file(args: argparse.Namespace) -> int:
    """Show MIDI file information."""
    from coremusic.midi.utilities import MIDISequence

    path = require_file(args.path)

    try:
        seq = MIDISequence.load(str(path))
    except Exception as e:
        raise CLIError(f"Failed to load MIDI file: {e}")

    # Gather statistics
    total_events = sum(len(t.events) for t in seq.tracks)
    total_notes = sum(
        len([e for e in t.events if e.status == 0x90])
        for t in seq.tracks
    )
    duration = seq.duration

    if args.json:
        tracks_info = []
        for i, track in enumerate(seq.tracks):
            note_events = [e for e in track.events if e.status == 0x90]
            tracks_info.append({
                "index": i,
                "name": track.name,
                "events": len(track.events),
                "notes": len(note_events),
                "channel": track.channel,
            })

        output_json({
            "file": str(path.absolute()),
            "tempo": seq.tempo,
            "ppq": seq.ppq,
            "duration_seconds": duration,
            "track_count": len(seq.tracks),
            "total_events": total_events,
            "total_notes": total_notes,
            "tracks": tracks_info,
        })
    else:
        print(f"File:     {path.name}")
        print(f"Path:     {path.absolute()}")
        print()
        print(f"Tempo:    {seq.tempo:.1f} BPM")
        print(f"PPQ:      {seq.ppq}")
        print(f"Duration: {format_duration(duration)}")
        print()
        print(f"Tracks:   {len(seq.tracks)}")
        print(f"Events:   {total_events:,}")
        print(f"Notes:    {total_notes:,}")

        if seq.tracks:
            print()
            print("Tracks:")
            for i, track in enumerate(seq.tracks):
                note_count = len([e for e in track.events if e.status == 0x90])
                name = track.name or "(unnamed)"
                print(f"  {i}. {name} - {note_count} notes")

    return EXIT_SUCCESS
