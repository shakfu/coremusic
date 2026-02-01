"""MIDI device commands."""

from __future__ import annotations

import argparse
from types import FrameType
from typing import Any, Dict, Tuple

import coremusic.capi as capi

from ._formatters import format_duration, output_json, output_table
from ._utils import EXIT_SUCCESS, CLIError, print_help_default, require_file


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register MIDI commands."""
    parser = subparsers.add_parser("midi", help="MIDI operations")
    midi_sub = parser.add_subparsers(dest="midi_command", metavar="<subcommand>")

    # midi list [--verbose]
    list_parser = midi_sub.add_parser("list", help="List MIDI devices, inputs, and outputs")
    list_parser.add_argument("--verbose", "-v", action="store_true",
                             help="Show detailed device information")
    list_parser.set_defaults(func=cmd_list)

    # midi info <file> [--events]
    info_parser = midi_sub.add_parser("info", help="Show MIDI file information")
    info_parser.add_argument("path", help="MIDI file path")
    info_parser.add_argument("--events", "-e", action="store_true",
                             help="Show event table instead of summary")
    info_parser.add_argument("--track", "-t", type=int, default=None,
                             help="Track index to show (default: all tracks)")
    info_parser.add_argument("--limit", "-n", type=int, default=100,
                             help="Maximum events to display (default: 100)")
    info_parser.set_defaults(func=cmd_info)

    # midi play <file>
    play_parser = midi_sub.add_parser("play", help="Play MIDI file to output device")
    play_parser.add_argument("path", help="MIDI file path")
    play_parser.add_argument("--device", "-d", type=int, default=0,
                             help="Output device index (default: 0)")
    play_parser.set_defaults(func=cmd_play)

    # midi quantize <file> -o <output>
    quantize_parser = midi_sub.add_parser("quantize", help="Quantize MIDI note timing to grid")
    quantize_parser.add_argument("path", help="Input MIDI file path")
    quantize_parser.add_argument("-o", "--output", required=True,
                                 help="Output MIDI file path")
    quantize_parser.add_argument("--grid", "-g", type=str, default="1/16",
                                 help="Quantize grid (e.g., 1/4, 1/8, 1/16, 1/32) (default: 1/16)")
    quantize_parser.add_argument("--strength", "-s", type=float, default=1.0,
                                 help="Quantize strength 0.0-1.0 (default: 1.0 = full)")
    quantize_parser.set_defaults(func=cmd_quantize)

    # midi receive [--device N] [-o file] [--plugin name] [--quiet]
    receive_parser = midi_sub.add_parser("receive", help="Receive MIDI input (display, record, or route to plugin)")
    receive_parser.add_argument("--device", "-d", type=int, default=0,
                                help="Input device index (default: 0)")
    receive_parser.add_argument("-o", "--output", default=None,
                                help="Output file (.mid for MIDI, .wav/.aif for audio with --plugin)")
    receive_parser.add_argument("--plugin", "-p", default=None,
                                help="Route MIDI to AudioUnit plugin (plays audio)")
    receive_parser.add_argument("--quiet", "-q", action="store_true",
                                help="Suppress audio output (only save to file)")
    receive_parser.add_argument("--duration", type=float, default=None,
                                help="Recording duration in seconds (default: until Ctrl+C)")
    receive_parser.add_argument("--tempo", "-t", type=float, default=120.0,
                                help="Tempo in BPM for MIDI file output (default: 120)")
    receive_parser.set_defaults(func=cmd_receive)

    # midi send <dest> [--note N] [--cc N N] [--program N] [--test]
    send_parser = midi_sub.add_parser("send", help="Send MIDI message to output")
    send_parser.add_argument("dest", type=int, nargs="?", default=0,
                             help="Output destination index (default: 0)")
    send_parser.add_argument("--test", action="store_true",
                             help="Send test note (middle C) to verify connectivity")
    send_parser.add_argument("--note", "-n", type=int,
                             help="Send note on/off (MIDI note number 0-127)")
    send_parser.add_argument("--velocity", "-v", type=int, default=100,
                             help="Note velocity (default: 100)")
    send_parser.add_argument("--channel", "-c", type=int, default=0,
                             help="MIDI channel 0-15 (default: 0)")
    send_parser.add_argument("--cc", type=int, nargs=2, metavar=("NUM", "VAL"),
                             help="Send control change (controller number and value)")
    send_parser.add_argument("--program", "-p", type=int,
                             help="Send program change (0-127)")
    send_parser.add_argument("--duration", type=float, default=0.5,
                             help="Note duration in seconds (default: 0.5)")
    send_parser.set_defaults(func=cmd_send)

    # midi panic [<dest>]
    panic_parser = midi_sub.add_parser("panic", help="Send all-notes-off on all channels")
    panic_parser.add_argument("dest", type=int, nargs="?", default=None,
                              help="Output destination index (default: all outputs)")
    panic_parser.set_defaults(func=cmd_panic)

    parser.set_defaults(func=lambda args: print_help_default(parser))


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


def _get_device_info(device_id: int) -> dict[str, Any]:
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


def cmd_list(args: argparse.Namespace) -> int:
    """List all MIDI devices, inputs, and outputs."""
    # Gather devices
    num_devices = capi.midi_get_number_of_devices()
    devices = []
    for i in range(num_devices):
        device_id = capi.midi_get_device(i)
        info = _get_device_info(device_id)
        info["index"] = i
        devices.append(info)

    # Gather inputs
    num_sources = capi.midi_get_number_of_sources()
    inputs = []
    for i in range(num_sources):
        source_id = capi.midi_get_source(i)
        name = _get_endpoint_name(source_id)
        inputs.append({"index": i, "id": source_id, "name": name})

    # Gather outputs
    num_dests = capi.midi_get_number_of_destinations()
    outputs = []
    for i in range(num_dests):
        dest_id = capi.midi_get_destination(i)
        name = _get_endpoint_name(dest_id)
        outputs.append({"index": i, "id": dest_id, "name": name})

    if args.json:
        output_json({
            "devices": devices,
            "inputs": inputs,
            "outputs": outputs,
        })
        return EXIT_SUCCESS

    # Text output
    if not devices and not inputs and not outputs:
        print("No MIDI devices found.")
        return EXIT_SUCCESS

    # Devices section
    if devices:
        print(f"Devices ({len(devices)}):")
        if args.verbose:
            for d in devices:
                print(f"\n  [{d['index']}] {d['name']}")
                if d["manufacturer"]:
                    print(f"      Manufacturer: {d['manufacturer']}")
                if d["model"]:
                    print(f"      Model:        {d['model']}")
                print(f"      Inputs:       {d['sources']}")
                print(f"      Outputs:      {d['destinations']}")
        else:
            for d in devices:
                mfr = f" ({d['manufacturer']})" if d["manufacturer"] else ""
                print(f"  [{d['index']}] {d['name']}{mfr}")

    # Inputs section
    if inputs:
        print(f"\nInputs ({len(inputs)}):")
        for inp in inputs:
            print(f"  [{inp['index']}] {inp['name']}")

    # Outputs section
    if outputs:
        print(f"\nOutputs ({len(outputs)}):")
        for out in outputs:
            print(f"  [{out['index']}] {out['name']}")

    return EXIT_SUCCESS


def cmd_send(args: argparse.Namespace) -> int:
    """Send MIDI message."""
    import time

    # Handle --test flag (send middle C)
    if args.test:
        args.note = 60
        args.velocity = 80
        args.duration = 0.3

    # Validate inputs
    if args.note is None and args.cc is None and args.program is None:
        raise CLIError("Must specify --note, --cc, --program, or --test")

    if not 0 <= args.channel <= 15:
        raise CLIError(f"Channel must be 0-15, got {args.channel}")

    # Check for MIDI outputs
    num_dests = capi.midi_get_number_of_destinations()
    if num_dests == 0:
        raise CLIError("No MIDI output destinations available")

    if args.dest >= num_dests:
        raise CLIError(f"Device index {args.dest} out of range (0-{num_dests-1})")

    dest_id = capi.midi_get_destination(args.dest)
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


def cmd_info(args: argparse.Namespace) -> int:
    """Show MIDI file information or event table."""
    from coremusic.midi.utilities import MIDISequence

    path = require_file(args.path)

    try:
        seq = MIDISequence.load(str(path))
    except Exception as e:
        raise CLIError(f"Failed to load MIDI file: {e}")

    # If --events flag, show event table (formerly cmd_file_dump)
    if args.events:
        return _show_events(args, path, seq)

    # Otherwise show summary (formerly cmd_file)
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


def _show_events(args: argparse.Namespace, path: Any, seq: Any) -> int:
    """Show MIDI event table (helper for cmd_info --events)."""
    # Collect events to display
    events_to_show = []

    if args.track is not None:
        if args.track >= len(seq.tracks):
            raise CLIError(f"Track {args.track} out of range (0-{len(seq.tracks)-1})")
        tracks_to_dump = [(args.track, seq.tracks[args.track])]
    else:
        tracks_to_dump = list(enumerate(seq.tracks))

    for track_idx, track in tracks_to_dump:
        for event in track.events:
            events_to_show.append({
                "track": track_idx,
                "time": event.time,
                "status": event.status,
                "channel": event.channel,
                "data1": event.data1,
                "data2": event.data2,
                "bytes": event.to_bytes().hex(),
            })

    # Sort by time
    events_to_show.sort(key=lambda e: (e["time"], e["track"]))

    # Limit output
    total_events = len(events_to_show)
    events_to_show = events_to_show[:args.limit]

    if args.json:
        output_json({
            "file": str(path.absolute()),
            "total_events": total_events,
            "displayed": len(events_to_show),
            "events": events_to_show,
        })
    else:
        print(f"File: {path.name}")
        print(f"Events: {total_events} total, showing {len(events_to_show)}\n")

        headers = ["Time", "Track", "Ch", "Type", "Data", "Hex"]
        rows = []
        for e in events_to_show:
            status_name = _get_status_name(e["status"])
            if e["status"] in (0xC0, 0xD0):
                data_str = f"{e['data1']:3d}"
            else:
                data_str = f"{e['data1']:3d} {e['data2']:3d}"

            rows.append([
                f"{e['time']:.3f}",
                str(e["track"]),
                str(e["channel"]),
                status_name,
                data_str,
                e["bytes"],
            ])

        output_table(headers, rows)

        if total_events > args.limit:
            print(f"\n... and {total_events - args.limit} more events (use --limit to show more)")

    return EXIT_SUCCESS


def cmd_panic(args: argparse.Namespace) -> int:
    """Send all-notes-off on all channels to stop stuck notes."""
    num_dests = capi.midi_get_number_of_destinations()
    if num_dests == 0:
        raise CLIError("No MIDI output destinations available")

    # Determine which outputs to send panic to
    if args.dest is not None:
        if args.dest >= num_dests:
            raise CLIError(f"Output index {args.dest} out of range (0-{num_dests-1})")
        dest_indices = [args.dest]
    else:
        dest_indices = list(range(num_dests))

    client_id = capi.midi_client_create("coremusic-panic")
    try:
        port_id = capi.midi_output_port_create(client_id, "output")

        results = []
        for idx in dest_indices:
            dest_id = capi.midi_get_destination(idx)
            dest_name = _get_endpoint_name(dest_id)

            # Send CC 123 (All Notes Off) on all 16 channels
            for channel in range(16):
                cc_data = bytes([0xB0 | channel, 123, 0])
                capi.midi_send(port_id, dest_id, cc_data, 0)

            # Also send CC 120 (All Sound Off) for immediate silence
            for channel in range(16):
                cc_data = bytes([0xB0 | channel, 120, 0])
                capi.midi_send(port_id, dest_id, cc_data, 0)

            results.append({"index": idx, "name": dest_name})

            if not args.json:
                print(f"Panic sent to: {dest_name}")

        if args.json:
            output_json({
                "destinations": results,
                "channels": 16,
                "messages": ["All Notes Off (CC 123)", "All Sound Off (CC 120)"],
            })

    finally:
        try:
            capi.midi_client_dispose(client_id)
        except Exception:
            pass

    return EXIT_SUCCESS


def cmd_receive(args: argparse.Namespace) -> int:
    """Receive MIDI input - display, record to file, or route to plugin.

    Modes:
    - No args: Display incoming MIDI messages
    - -o file.mid: Save to MIDI file
    - --plugin Name: Route to AudioUnit plugin, play audio
    - --plugin Name -o file.wav: Route to plugin, play audio AND save to file
    - --plugin Name -o file.wav --quiet: Route to plugin, only save to file
    """
    import signal
    import threading
    import time
    from pathlib import Path

    # Validate arguments
    if args.quiet and not args.output:
        raise CLIError("--quiet requires -o/--output")

    if args.quiet and not args.plugin:
        raise CLIError("--quiet only makes sense with --plugin (no audio to suppress otherwise)")

    num_sources = capi.midi_get_number_of_sources()
    if num_sources == 0:
        raise CLIError("No MIDI input sources available")

    if args.device >= num_sources:
        raise CLIError(f"Input index {args.device} out of range (0-{num_sources-1})")

    source_id = capi.midi_get_source(args.device)
    source_name = _get_endpoint_name(source_id)

    # Determine output path and type
    output_path = Path(args.output) if args.output else None
    is_midi_output = output_path and output_path.suffix.lower() in (".mid", ".midi")
    is_audio_output = output_path and output_path.suffix.lower() in (".wav", ".aif", ".aiff", ".m4a", ".caf")

    # Validate output type
    if output_path and args.plugin and is_midi_output:
        raise CLIError("Cannot save MIDI file when using --plugin (use .wav/.aif for audio)")
    if output_path and not args.plugin and is_audio_output:
        raise CLIError("Audio output requires --plugin")

    # Add .mid extension if saving MIDI without extension
    if output_path and not args.plugin and not is_midi_output and not is_audio_output:
        output_path = output_path.with_suffix(".mid")
        is_midi_output = True

    # Route to appropriate handler
    if args.plugin:
        return _receive_with_plugin(args, source_id, source_name, output_path, is_audio_output)
    elif output_path:
        return _receive_to_midi_file(args, source_id, source_name, output_path)
    else:
        return _receive_display(args, source_id, source_name)


def _receive_display(args: argparse.Namespace, source_id: int, source_name: str) -> int:
    """Display incoming MIDI messages (formerly cmd_monitor)."""
    import signal
    import threading

    event_count = 0
    events_list: list[Dict[str, Any]] = []
    stop_event = threading.Event()

    def format_midi_message(data: bytes, timestamp: float) -> Dict[str, Any]:
        """Format MIDI message for display."""
        if len(data) < 1:
            return {"type": "unknown", "data": data.hex()}

        status = data[0] & 0xF0
        channel = data[0] & 0x0F
        d1 = data[1] if len(data) > 1 else 0
        d2 = data[2] if len(data) > 2 else 0

        msg: Dict[str, Any] = {
            "channel": channel,
            "timestamp": timestamp,
            "raw": data.hex(),
        }

        if status == 0x90 and d2 > 0:
            msg["type"] = "note_on"
            msg["note"] = d1
            msg["velocity"] = d2
        elif status == 0x80 or (status == 0x90 and d2 == 0):
            msg["type"] = "note_off"
            msg["note"] = d1
        elif status == 0xB0:
            msg["type"] = "cc"
            msg["controller"] = d1
            msg["value"] = d2
        elif status == 0xC0:
            msg["type"] = "program"
            msg["program"] = d1
        elif status == 0xE0:
            msg["type"] = "pitch_bend"
            msg["value"] = d1 | (d2 << 7)
        elif status == 0xA0:
            msg["type"] = "poly_aftertouch"
            msg["note"] = d1
            msg["pressure"] = d2
        elif status == 0xD0:
            msg["type"] = "channel_aftertouch"
            msg["pressure"] = d1
        else:
            msg["type"] = "other"
            msg["status"] = hex(status)

        return msg

    def print_midi_message(msg: Dict[str, Any]) -> None:
        """Print formatted MIDI message."""
        ch = msg["channel"]
        ts = msg.get("timestamp", 0)

        if msg["type"] == "note_on":
            print(f"[{ts:10.3f}] ch={ch:2d}  Note On   note={msg['note']:3d}  vel={msg['velocity']:3d}")
        elif msg["type"] == "note_off":
            print(f"[{ts:10.3f}] ch={ch:2d}  Note Off  note={msg['note']:3d}")
        elif msg["type"] == "cc":
            print(f"[{ts:10.3f}] ch={ch:2d}  CC        cc={msg['controller']:3d}  val={msg['value']:3d}")
        elif msg["type"] == "program":
            print(f"[{ts:10.3f}] ch={ch:2d}  Program   prog={msg['program']:3d}")
        elif msg["type"] == "pitch_bend":
            print(f"[{ts:10.3f}] ch={ch:2d}  Pitch     val={msg['value']:5d}")
        elif msg["type"] == "poly_aftertouch":
            print(f"[{ts:10.3f}] ch={ch:2d}  PolyAT    note={msg['note']:3d}  pres={msg['pressure']:3d}")
        elif msg["type"] == "channel_aftertouch":
            print(f"[{ts:10.3f}] ch={ch:2d}  ChanAT    pres={msg['pressure']:3d}")
        else:
            print(f"[{ts:10.3f}] ch={ch:2d}  Other     {msg['raw']}")

    def midi_callback(data: bytes, timestamp: float) -> None:
        nonlocal event_count
        event_count += 1
        msg = format_midi_message(data, timestamp)
        if args.json:
            events_list.append(msg)
        else:
            print_midi_message(msg)

    def signal_handler(sig: int, frame: FrameType | None) -> None:
        stop_event.set()

    original_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        client_id = capi.midi_client_create("coremusic-receive")
        try:
            port_id = capi.midi_input_port_create(client_id, "input")
            capi.midi_port_connect_source(port_id, source_id)

            if not args.json:
                print(f"Receiving from: {source_name}")
                print("Press Ctrl+C to stop...\n")

            while not stop_event.is_set():
                stop_event.wait(timeout=0.1)

        finally:
            try:
                capi.midi_client_dispose(client_id)
            except Exception:
                pass

        if args.json:
            output_json({
                "source": source_name,
                "index": args.device,
                "event_count": event_count,
                "events": events_list,
            })
        else:
            print(f"\nStopped. Received {event_count} events.")

    finally:
        signal.signal(signal.SIGINT, original_handler)

    return EXIT_SUCCESS


def _receive_to_midi_file(args: argparse.Namespace, source_id: int, source_name: str, output_path: Any) -> int:
    """Save incoming MIDI to file (formerly cmd_record)."""
    import signal
    import threading
    import time

    from coremusic.midi.utilities import MIDIEvent, MIDISequence

    recorded_events: list[Tuple[float, bytes]] = []
    start_time: float = 0.0
    stop_event = threading.Event()
    event_lock = threading.Lock()

    def midi_callback(data: bytes, timestamp: float) -> None:
        nonlocal start_time
        if start_time == 0.0:
            start_time = timestamp
        relative_time = timestamp - start_time
        with event_lock:
            recorded_events.append((relative_time, data))

    def signal_handler(sig: int, frame: FrameType | None) -> None:
        stop_event.set()

    original_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        client_id = capi.midi_client_create("coremusic-receive")
        try:
            port_id = capi.midi_input_port_create(client_id, "input")
            capi.midi_port_connect_source(port_id, source_id)

            if not args.json:
                print(f"Recording from: {source_name}")
                print(f"Output: {output_path}")
                if args.duration:
                    print(f"Duration: {args.duration:.1f} seconds")
                else:
                    print("Press Ctrl+C to stop...")
                print()

            recording_start = time.time()
            while not stop_event.is_set():
                if args.duration and (time.time() - recording_start) >= args.duration:
                    break
                stop_event.wait(timeout=0.1)

        finally:
            try:
                capi.midi_client_dispose(client_id)
            except Exception:
                pass

        with event_lock:
            events_copy = list(recorded_events)

        if not events_copy:
            if not args.json:
                print("No MIDI events recorded.")
            return EXIT_SUCCESS

        seq = MIDISequence(tempo=args.tempo)
        track = seq.add_track("Recorded")

        for rel_time, data in events_copy:
            if len(data) >= 2:
                status = data[0] & 0xF0
                channel = data[0] & 0x0F
                d1 = data[1]
                d2 = data[2] if len(data) > 2 else 0
                event = MIDIEvent(
                    time=rel_time,
                    status=status,
                    channel=channel,
                    data1=d1,
                    data2=d2,
                )
                track.events.append(event)

        try:
            seq.save(str(output_path))
        except Exception as e:
            raise CLIError(f"Failed to save MIDI file: {e}")

        recording_duration = events_copy[-1][0] if events_copy else 0.0
        note_on_count = sum(1 for _, d in events_copy if len(d) >= 1 and (d[0] & 0xF0) == 0x90)

        if args.json:
            output_json({
                "source": source_name,
                "index": args.device,
                "output": str(output_path.absolute()),
                "event_count": len(events_copy),
                "note_count": note_on_count,
                "duration_seconds": recording_duration,
                "tempo": args.tempo,
            })
        else:
            print(f"\nRecording saved: {output_path}")
            print(f"  Events:   {len(events_copy)}")
            print(f"  Notes:    {note_on_count}")
            print(f"  Duration: {format_duration(recording_duration)}")

    finally:
        signal.signal(signal.SIGINT, original_handler)

    return EXIT_SUCCESS


def _receive_with_plugin(args: argparse.Namespace, source_id: int, source_name: str,
                         output_path: Any, is_audio_output: bool) -> int:
    """Route incoming MIDI to AudioUnit plugin with optional audio capture."""
    import signal
    import threading
    import time

    from coremusic.audio.audiounit_host import AudioUnitHost

    stop_event = threading.Event()
    event_count = 0
    note_count = 0

    def signal_handler(sig: int, frame: FrameType | None) -> None:
        stop_event.set()

    original_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        # Load the AudioUnit plugin
        host = AudioUnitHost()
        try:
            plugin = host.load_plugin(args.plugin, type='instrument')
        except Exception as e:
            raise CLIError(f"Failed to load plugin '{args.plugin}': {e}")

        if not args.json:
            print(f"Receiving from: {source_name}")
            print(f"Plugin: {args.plugin}")
            if output_path:
                print(f"Output: {output_path}")
            if args.quiet:
                print("Audio output: disabled (quiet mode)")
            if args.duration:
                print(f"Duration: {args.duration:.1f} seconds")
            else:
                print("Press Ctrl+C to stop...")
            print()

        # Set up audio output and optional file capture
        audio_chunks: list[bytes] = []
        sample_rate = 44100
        channels = 2

        # Initialize plugin
        try:
            plugin.instantiate()
            plugin.initialize()
        except Exception as e:
            raise CLIError(f"Failed to initialize plugin: {e}")

        # Set up audio output queue if not quiet
        output_queue = None
        if not args.quiet:
            import coremusic as cm
            audio_format = cm.AudioFormat(
                sample_rate=float(sample_rate),
                format_id="lpcm",
                channels_per_frame=channels,
                bits_per_channel=32,
            )
            output_queue = cm.AudioQueue.new_output(audio_format)

        # Create MIDI client
        client_id = capi.midi_client_create("coremusic-receive")
        try:
            port_id = capi.midi_input_port_create(client_id, "input")
            capi.midi_port_connect_source(port_id, source_id)

            start_time = time.time()
            last_render_time = start_time

            while not stop_event.is_set():
                if args.duration and (time.time() - start_time) >= args.duration:
                    break

                # Render audio from plugin periodically
                current_time = time.time()
                if current_time - last_render_time >= 0.01:  # ~100Hz render rate
                    try:
                        # Render a small buffer
                        frames = 512
                        rendered = plugin.render(frames)
                        if rendered and len(rendered) > 0:
                            if output_path:
                                audio_chunks.append(rendered)
                            if output_queue:
                                output_queue.enqueue_buffer(rendered)
                    except Exception:
                        pass  # Ignore render errors
                    last_render_time = current_time

                stop_event.wait(timeout=0.01)

        finally:
            try:
                capi.midi_client_dispose(client_id)
            except Exception:
                pass

        # Clean up plugin
        try:
            plugin.dispose()
        except Exception:
            pass

        # Save audio file if requested
        if output_path and audio_chunks:
            try:
                import coremusic as cm
                from coremusic.constants import AudioFileType

                audio_data = b''.join(audio_chunks)
                audio_format = cm.AudioFormat(
                    sample_rate=float(sample_rate),
                    format_id="lpcm",
                    channels_per_frame=channels,
                    bits_per_channel=32,
                )

                ext = output_path.suffix.lower()
                file_type = {
                    '.wav': AudioFileType.WAVE,
                    '.aif': AudioFileType.AIFF,
                    '.aiff': AudioFileType.AIFF,
                    '.m4a': AudioFileType.M4A,
                    '.caf': AudioFileType.CAF,
                }.get(ext, AudioFileType.WAVE)

                with cm.ExtendedAudioFile.create(str(output_path), file_type, audio_format) as out_file:
                    num_frames = len(audio_data) // (channels * 4)  # 4 bytes per float32
                    out_file.write(num_frames, audio_data)

                if not args.json:
                    duration = num_frames / sample_rate
                    print(f"\nAudio saved: {output_path}")
                    print(f"  Duration: {format_duration(duration)}")

            except Exception as e:
                raise CLIError(f"Failed to save audio file: {e}")

        if args.json:
            output_json({
                "source": source_name,
                "index": args.device,
                "plugin": args.plugin,
                "output": str(output_path) if output_path else None,
                "event_count": event_count,
                "note_count": note_count,
            })
        elif not output_path:
            print(f"\nStopped.")

    finally:
        signal.signal(signal.SIGINT, original_handler)

    return EXIT_SUCCESS


def _get_status_name(status: int) -> str:
    """Get human-readable name for MIDI status byte."""
    names = {
        0x80: "Note Off",
        0x90: "Note On",
        0xA0: "Poly AT",
        0xB0: "CC",
        0xC0: "Program",
        0xD0: "Chan AT",
        0xE0: "Pitch Bend",
    }
    return names.get(status, f"0x{status:02X}")


def cmd_play(args: argparse.Namespace) -> int:
    """Play MIDI file to output device."""
    import signal
    import time

    from coremusic.midi.utilities import MIDISequence

    path = require_file(args.path)

    try:
        seq = MIDISequence.load(str(path))
    except Exception as e:
        raise CLIError(f"Failed to load MIDI file: {e}")

    num_dests = capi.midi_get_number_of_destinations()
    if num_dests == 0:
        raise CLIError("No MIDI output destinations available")

    if args.device >= num_dests:
        raise CLIError(f"Output index {args.device} out of range (0-{num_dests-1})")

    dest_id = capi.midi_get_destination(args.device)
    dest_name = _get_endpoint_name(dest_id)

    # Collect all events with absolute time
    all_events = []
    for track in seq.tracks:
        for event in track.events:
            all_events.append(event)

    # Sort by time
    all_events.sort(key=lambda e: e.time)

    if not all_events:
        if not args.json:
            print("No events to play.")
        return EXIT_SUCCESS

    # Stop flag for Ctrl+C
    stop_requested = False

    def signal_handler(sig: int, frame: FrameType | None) -> None:
        nonlocal stop_requested
        stop_requested = True

    original_handler = signal.signal(signal.SIGINT, signal_handler)

    client_id = capi.midi_client_create("coremusic-player")
    try:
        port_id = capi.midi_output_port_create(client_id, "output")

        duration = seq.duration
        if not args.json:
            print(f"Playing: {path.name}")
            print(f"Output:  {dest_name}")
            print(f"Tempo:   {seq.tempo:.1f} BPM")
            print(f"Duration:{format_duration(duration)}")
            print("Press Ctrl+C to stop...\n")

        start_time = time.time()
        events_played = 0

        for event in all_events:
            if stop_requested:
                break

            # Wait until event time
            target_time = start_time + event.time
            now = time.time()
            wait_time = target_time - now
            if wait_time > 0:
                time.sleep(wait_time)

            if stop_requested:
                break

            # Send MIDI event
            midi_data = event.to_bytes()
            capi.midi_send(port_id, dest_id, midi_data, 0)
            events_played += 1

        # Send all-notes-off to clean up
        if stop_requested:
            for channel in range(16):
                cc_data = bytes([0xB0 | channel, 123, 0])
                capi.midi_send(port_id, dest_id, cc_data, 0)

        if args.json:
            output_json({
                "file": str(path.absolute()),
                "device": dest_name,
                "events_played": events_played,
                "total_events": len(all_events),
                "stopped": stop_requested,
            })
        else:
            if stop_requested:
                print(f"\nStopped. Played {events_played}/{len(all_events)} events.")
            else:
                print(f"Finished. Played {events_played} events.")

    finally:
        try:
            capi.midi_client_dispose(client_id)
        except Exception:
            pass
        signal.signal(signal.SIGINT, original_handler)

    return EXIT_SUCCESS


def cmd_quantize(args: argparse.Namespace) -> int:
    """Quantize MIDI note timing to grid."""
    from pathlib import Path

    from coremusic.midi.utilities import MIDISequence, MIDIStatus

    input_path = require_file(args.path)
    output_path = Path(args.output)

    # Parse grid value (e.g., "1/16" -> 0.25 beats)
    try:
        if "/" in args.grid:
            num, denom = args.grid.split("/")
            grid_beats = float(num) / float(denom) * 4  # Convert to beats (1/4 = 1 beat)
        else:
            grid_beats = float(args.grid)
    except ValueError:
        raise CLIError(f"Invalid grid value: {args.grid}. Use format like 1/4, 1/8, 1/16, or decimal.")

    if grid_beats <= 0:
        raise CLIError("Grid value must be positive")

    strength = max(0.0, min(1.0, args.strength))

    # Load MIDI file
    try:
        seq = MIDISequence.load(str(input_path))
    except Exception as e:
        raise CLIError(f"Failed to load MIDI file: {e}")

    # Calculate grid in seconds based on tempo
    tempo = seq.tempo if seq.tempo else 120.0
    seconds_per_beat = 60.0 / tempo
    grid_seconds = grid_beats * seconds_per_beat

    if not args.json:
        print(f"Quantizing: {input_path.name}")
        print(f"Grid:       {args.grid} ({grid_beats:.4f} beats, {grid_seconds:.4f}s)")
        print(f"Strength:   {strength:.0%}")
        print(f"Tempo:      {tempo:.1f} BPM")

    # Quantize note events in all tracks
    total_events = 0
    quantized_events = 0

    for track in seq.tracks:
        for event in track.events:
            # Only quantize note events
            if event.status in (MIDIStatus.NOTE_ON, MIDIStatus.NOTE_OFF):
                total_events += 1

                # Find nearest grid point
                nearest_grid = round(event.time / grid_seconds) * grid_seconds

                # Apply strength (interpolate between original and quantized)
                new_time = event.time + (nearest_grid - event.time) * strength

                # Only count as quantized if time actually changed
                if abs(new_time - event.time) > 0.0001:
                    quantized_events += 1
                    event.time = max(0.0, new_time)

    # Save quantized file
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        seq.save(str(output_path))
    except Exception as e:
        raise CLIError(f"Failed to save MIDI file: {e}")

    if args.json:
        output_json({
            "input": str(input_path.absolute()),
            "output": str(output_path.absolute()),
            "grid": args.grid,
            "grid_beats": grid_beats,
            "grid_seconds": grid_seconds,
            "strength": strength,
            "tempo": tempo,
            "total_note_events": total_events,
            "quantized_events": quantized_events,
        })
    else:
        print(f"\nOutput:     {output_path}")
        print(f"Events:     {quantized_events}/{total_events} notes adjusted")

    return EXIT_SUCCESS
