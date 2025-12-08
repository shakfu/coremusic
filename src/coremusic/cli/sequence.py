"""MIDI sequence commands."""

from __future__ import annotations

import argparse

from ._formatters import format_duration, output_json, output_table
from ._utils import EXIT_SUCCESS, CLIError, require_file


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register sequence commands."""
    parser = subparsers.add_parser("sequence", help="MIDI sequence operations")
    seq_sub = parser.add_subparsers(dest="sequence_command", metavar="<subcommand>")

    # sequence info
    info_parser = seq_sub.add_parser("info", help="Display MIDI file information")
    info_parser.add_argument("file", help="MIDI file path")
    info_parser.set_defaults(func=cmd_info)

    # sequence play
    play_parser = seq_sub.add_parser("play", help="Play MIDI file (requires output device)")
    play_parser.add_argument("file", help="MIDI file path")
    play_parser.add_argument(
        "--device", "-d", type=int, default=0,
        help="MIDI output device index (default: 0)"
    )
    play_parser.add_argument(
        "--tempo", "-t", type=float, default=None,
        help="Override tempo (BPM)"
    )
    play_parser.set_defaults(func=cmd_play)

    # sequence list
    list_parser = seq_sub.add_parser("tracks", help="List tracks in MIDI file")
    list_parser.add_argument("file", help="MIDI file path")
    list_parser.set_defaults(func=cmd_tracks)

    parser.set_defaults(func=lambda args: parser.print_help() or EXIT_SUCCESS)


def cmd_info(args: argparse.Namespace) -> int:
    """Display MIDI file information."""
    from coremusic.midi.utilities import MIDISequence

    path = require_file(args.file)

    try:
        seq = MIDISequence.load(str(path))
    except Exception as e:
        raise CLIError(f"Failed to load MIDI file: {e}")

    # Gather info
    total_events = sum(len(t.events) for t in seq.tracks)
    duration = seq.duration

    if args.json:
        tracks_info = []
        for i, track in enumerate(seq.tracks):
            tracks_info.append({
                "index": i,
                "name": track.name,
                "events": len(track.events),
                "notes": len([e for e in track.events if e.status == 0x90]),
            })

        output_json({
            "file": str(path.absolute()),
            "tempo": seq.tempo,
            "ppq": seq.ppq,
            "duration_seconds": duration,
            "track_count": len(seq.tracks),
            "total_events": total_events,
            "tracks": tracks_info,
        })
    else:
        print(f"File: {path.name}")
        print(f"Path: {path.absolute()}")
        print()
        print(f"Tempo:    {seq.tempo:.1f} BPM")
        print(f"PPQ:      {seq.ppq}")
        print(f"Duration: {format_duration(duration)}")
        print(f"Tracks:   {len(seq.tracks)}")
        print(f"Events:   {total_events:,}")

    return EXIT_SUCCESS


def cmd_tracks(args: argparse.Namespace) -> int:
    """List tracks in MIDI file."""
    from coremusic.midi.utilities import MIDISequence

    path = require_file(args.file)

    try:
        seq = MIDISequence.load(str(path))
    except Exception as e:
        raise CLIError(f"Failed to load MIDI file: {e}")

    if args.json:
        tracks_info = []
        for i, track in enumerate(seq.tracks):
            note_on_events = [e for e in track.events if e.status == 0x90]
            tracks_info.append({
                "index": i,
                "name": track.name,
                "events": len(track.events),
                "notes": len(note_on_events),
                "channel": track.channel,
            })
        output_json(tracks_info)
    else:
        if not seq.tracks:
            print("No tracks found.")
            return EXIT_SUCCESS

        print(f"Tracks in {path.name}:\n")
        headers = ["#", "Name", "Events", "Notes", "Channel"]
        rows = []
        for i, track in enumerate(seq.tracks):
            note_on_events = [e for e in track.events if e.status == 0x90]
            rows.append([
                str(i),
                track.name[:30] if track.name else "(unnamed)",
                str(len(track.events)),
                str(len(note_on_events)),
                str(track.channel),
            ])
        output_table(headers, rows)

    return EXIT_SUCCESS


def cmd_play(args: argparse.Namespace) -> int:
    """Play MIDI file."""
    import time

    import coremusic.capi as capi
    from coremusic.midi.utilities import MIDISequence

    path = require_file(args.file)

    try:
        seq = MIDISequence.load(str(path))
    except Exception as e:
        raise CLIError(f"Failed to load MIDI file: {e}")

    # Override tempo if specified
    if args.tempo:
        seq.tempo = args.tempo

    # Check for MIDI outputs
    num_dests = capi.midi_get_number_of_destinations()
    if num_dests == 0:
        raise CLIError("No MIDI output destinations available")

    if args.device >= num_dests:
        raise CLIError(f"Device index {args.device} out of range (0-{num_dests-1})")

    dest_id = capi.midi_get_destination(args.device)

    # Get device name
    try:
        dest_name = capi.midi_object_get_string_property(dest_id, "name")
    except Exception:
        dest_name = f"Device {args.device}"

    duration = seq.duration

    if not args.json:
        print(f"Playing: {path.name}")
        print(f"Device:  {dest_name}")
        print(f"Tempo:   {seq.tempo:.1f} BPM")
        print(f"Duration: {format_duration(duration)}")
        print()
        print("Press Ctrl+C to stop...")

    # Create MIDI client and output port
    try:
        client_id = capi.midi_client_create("coremusic-player")
        port_id = capi.midi_output_port_create(client_id, "output")

        # Collect and sort all events by time
        all_events = []
        for track in seq.tracks:
            for event in track.events:
                all_events.append(event)

        all_events.sort(key=lambda e: e.time)

        # Play events
        start_time = time.time()
        event_index = 0

        try:
            while event_index < len(all_events):
                current_time = time.time() - start_time
                event = all_events[event_index]

                if event.time <= current_time:
                    # Send MIDI message
                    if event.status in (0x80, 0x90):  # Note on/off
                        try:
                            midi_data = bytes([event.status | event.channel, event.data1, event.data2])
                            capi.midi_send(port_id, dest_id, midi_data, 0)
                        except Exception:
                            pass  # Continue on send errors
                    event_index += 1
                else:
                    # Wait a bit
                    time.sleep(0.001)

        except KeyboardInterrupt:
            if not args.json:
                print("\nStopped.")

        # Send all notes off
        for channel in range(16):
            try:
                # All notes off (CC 123)
                all_notes_off = bytes([0xB0 | channel, 123, 0])
                capi.midi_send(port_id, dest_id, all_notes_off, 0)
            except Exception:
                pass

    finally:
        try:
            capi.midi_client_dispose(client_id)
        except Exception:
            pass

    if args.json:
        output_json({
            "file": str(path.absolute()),
            "device": dest_name,
            "tempo": seq.tempo,
            "events_played": event_index,
            "total_events": len(all_events),
        })
    else:
        print(f"\nFinished ({event_index}/{len(all_events)} events)")

    return EXIT_SUCCESS
