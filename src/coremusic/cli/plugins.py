"""AudioUnit plugin commands."""

from __future__ import annotations

import argparse

import coremusic.capi as capi
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
    """Register plugin commands."""
    parser = subparsers.add_parser("plugin", help="AudioUnit plugin discovery")
    plugin_sub = parser.add_subparsers(dest="plugin_command", metavar="<subcommand>")

    # plugin list
    list_parser = plugin_sub.add_parser("list", help="List available AudioUnit plugins")
    list_parser.add_argument("--type", dest="plugin_type", choices=list(PLUGIN_TYPES.keys()),
                             help="Filter by plugin type")
    list_parser.add_argument("--manufacturer", help="Filter by manufacturer")
    list_parser.add_argument("--name-only", action="store_true",
                             help="Print only unique plugin names")
    list_parser.set_defaults(func=cmd_list)

    # plugin find
    find_parser = plugin_sub.add_parser("find", help="Search for plugins by name")
    find_parser.add_argument("query", help="Search query")
    find_parser.add_argument("--type", dest="plugin_type", choices=list(PLUGIN_TYPES.keys()),
                             help="Filter by plugin type")
    find_parser.set_defaults(func=cmd_find)

    # plugin info
    info_parser = plugin_sub.add_parser("info", help="Show detailed plugin information")
    info_parser.add_argument("name", help="Plugin name (partial match)")
    info_parser.set_defaults(func=cmd_info)

    # plugin params
    params_parser = plugin_sub.add_parser("params", help="List plugin parameters")
    params_parser.add_argument("name", help="Plugin name (partial match)")
    params_parser.set_defaults(func=cmd_params)

    # plugin process
    process_parser = plugin_sub.add_parser("process", help="Apply effect plugin to audio file")
    process_parser.add_argument("name", help="Plugin name (partial match)")
    process_parser.add_argument("input", help="Input audio file path")
    process_parser.add_argument("-o", "--output", required=True, help="Output audio file path")
    process_parser.add_argument("--preset", type=str, default=None,
                                help="Factory preset name or number to load")
    process_parser.set_defaults(func=cmd_process)

    # plugin render
    render_parser = plugin_sub.add_parser("render", help="Render MIDI through instrument plugin")
    render_parser.add_argument("name", help="Plugin name (partial match)")
    render_parser.add_argument("midi", help="Input MIDI file path")
    render_parser.add_argument("-o", "--output", required=True, help="Output audio file path")
    render_parser.add_argument("--sample-rate", type=int, default=44100,
                               help="Output sample rate (default: 44100)")
    render_parser.add_argument("--duration", type=float, default=None,
                               help="Extra duration in seconds after MIDI ends (default: 1.0)")
    render_parser.add_argument("--preset", type=str, default=None,
                               help="Factory preset name or number to load")
    render_parser.set_defaults(func=cmd_render)

    # plugin preset
    preset_parser = plugin_sub.add_parser("preset", help="Plugin preset management")
    preset_sub = preset_parser.add_subparsers(dest="preset_command", metavar="<subcommand>")

    # plugin preset list
    preset_list_parser = preset_sub.add_parser("list", help="List factory presets for a plugin")
    preset_list_parser.add_argument("name", help="Plugin name (partial match)")
    preset_list_parser.set_defaults(func=cmd_preset_list)

    preset_parser.set_defaults(func=lambda args: preset_parser.print_help() or EXIT_SUCCESS)

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
    elif args.name_only:
        # Print unique names only
        names = sorted(set(c.get("name", "Unknown") for c in components))
        for name in names:
            print(name)
    else:
        if not components:
            print("No plugins found.")
            return EXIT_SUCCESS

        headers = ["Type", "Mfr", "Name"]
        rows = []
        for c in components:
            rows.append([
                get_plugin_type_display(c.get("type", "")),
                c.get("manufacturer", "Unknown"),
                c.get("name", "Unknown"),
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


def cmd_preset_list(args: argparse.Namespace) -> int:
    """List factory presets for a plugin."""
    import coremusic.capi as capi

    plugin = _find_plugin_by_name(args.name)
    comp_id = _get_component_id_for_plugin(plugin)

    # Create AudioUnit instance to query presets
    au_id = capi.audio_component_instance_new(comp_id)
    try:
        capi.audio_unit_initialize(au_id)

        # Get factory presets
        presets = capi.audio_unit_get_factory_presets(au_id)

        if args.json:
            output_json({
                "plugin": plugin.get("name"),
                "presets": presets,
            })
        else:
            if not presets:
                print(f"No factory presets found for '{plugin.get('name')}'")
                return EXIT_SUCCESS

            print(f"Factory presets for '{plugin.get('name')}':\n")
            headers = ["Number", "Name"]
            rows = [[str(p["number"]), p["name"]] for p in presets]
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


def _load_preset_by_name_or_number(au_id: int, preset_arg: str) -> str | None:
    """Load a factory preset by name or number. Returns preset name if found."""
    import coremusic.capi as capi

    presets = capi.audio_unit_get_factory_presets(au_id)
    if not presets:
        return None

    # Try to parse as number first
    try:
        preset_num = int(preset_arg)
        for p in presets:
            if p["number"] == preset_num:
                capi.audio_unit_set_current_preset(au_id, preset_num)
                return str(p["name"])
        # Number not found in presets
        return None
    except ValueError:
        pass

    # Try to match by name (case-insensitive partial match)
    preset_lower = preset_arg.lower()
    for p in presets:
        if preset_lower in str(p["name"]).lower():
            capi.audio_unit_set_current_preset(au_id, int(p["number"]))
            return str(p["name"])

    return None


def cmd_process(args: argparse.Namespace) -> int:
    """Apply effect plugin to audio file."""
    import struct
    import wave
    from pathlib import Path

    import coremusic as cm

    from ._utils import require_file

    input_path = require_file(args.input)
    output_path = Path(args.output)

    # Find the plugin
    plugin = _find_plugin_by_name(args.name)
    plugin_type = plugin.get("type", "")

    # Check if it's an effect plugin
    if plugin_type != "aufx":
        raise CLIError(f"Plugin '{plugin.get('name')}' is not an effect plugin (type: {plugin_type})")

    # Load input audio file
    try:
        with cm.AudioFile(str(input_path)) as audio_file:
            fmt = audio_file.format
            duration = audio_file.duration
            total_frames = int(duration * fmt.sample_rate)

            # Read all audio data
            audio_data, num_packets = capi.audio_file_read_packets(
                audio_file.object_id, 0, total_frames
            )
    except Exception as e:
        raise CLIError(f"Failed to read audio file: {e}")

    if not args.json:
        print(f"Processing: {input_path.name}")
        print(f"Plugin:     {plugin.get('name')}")
        print(f"Duration:   {duration:.2f}s")
        print(f"Format:     {fmt.sample_rate:.0f}Hz, {fmt.channels_per_frame}ch")

    # Create and initialize the AudioUnit
    comp_id = _get_component_id_for_plugin(plugin)
    au_id = capi.audio_component_instance_new(comp_id)

    try:
        capi.audio_unit_initialize(au_id)

        # Load preset if specified
        preset_name = None
        if args.preset:
            preset_name = _load_preset_by_name_or_number(au_id, args.preset)
            if preset_name is None:
                raise CLIError(f"Preset not found: {args.preset}")
            if not args.json:
                print(f"Preset:     {preset_name}")

        # Convert input to float32 if needed
        if fmt.bits_per_channel == 16:
            # Convert int16 to float32
            num_samples = len(audio_data) // 2
            samples = struct.unpack(f'<{num_samples}h', audio_data)
            float_samples = [s / 32768.0 for s in samples]
            audio_data = struct.pack(f'{len(float_samples)}f', *float_samples)
        elif fmt.bits_per_channel == 32 and fmt.format_flags & 0x1 == 0:
            # Already int32, convert to float32
            num_samples = len(audio_data) // 4
            samples = struct.unpack(f'<{num_samples}i', audio_data)
            float_samples = [s / 2147483648.0 for s in samples]
            audio_data = struct.pack(f'{len(float_samples)}f', *float_samples)

        # Process audio through plugin in chunks
        chunk_size = 4096
        processed_chunks = []
        num_frames = len(audio_data) // (4 * fmt.channels_per_frame)  # float32

        for offset in range(0, num_frames, chunk_size):
            frames_to_process = min(chunk_size, num_frames - offset)
            start_byte = offset * 4 * fmt.channels_per_frame
            end_byte = (offset + frames_to_process) * 4 * fmt.channels_per_frame
            chunk = audio_data[start_byte:end_byte]

            processed = capi.audio_unit_render(
                au_id, chunk, frames_to_process,
                fmt.sample_rate, fmt.channels_per_frame
            )
            processed_chunks.append(processed)

            if not args.json:
                progress = min(100, int((offset + frames_to_process) / num_frames * 100))
                print(f"\rProcessing: {progress}%", end="", flush=True)

        processed_data = b''.join(processed_chunks)

        if not args.json:
            print("\rProcessing: 100%")

        # Convert back to int16 for WAV output
        num_samples = len(processed_data) // 4
        float_samples = list(struct.unpack(f'{num_samples}f', processed_data))
        int_samples = [int(max(-32768, min(32767, s * 32768))) for s in float_samples]
        output_data = struct.pack(f'<{len(int_samples)}h', *int_samples)

        # Write output WAV file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), 'wb') as wav_out:
            wav_out.setnchannels(fmt.channels_per_frame)
            wav_out.setsampwidth(2)  # 16-bit
            wav_out.setframerate(int(fmt.sample_rate))
            wav_out.writeframes(output_data)

        if args.json:
            output_json({
                "input": str(input_path.absolute()),
                "output": str(output_path.absolute()),
                "plugin": plugin.get("name"),
                "duration": duration,
                "sample_rate": fmt.sample_rate,
                "channels": fmt.channels_per_frame,
            })
        else:
            print(f"Output:     {output_path}")

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


def cmd_render(args: argparse.Namespace) -> int:
    """Render MIDI through instrument plugin to audio file."""
    import struct
    import wave
    from pathlib import Path

    from coremusic.midi.utilities import MIDISequence

    from ._utils import require_file

    midi_path = require_file(args.midi)
    output_path = Path(args.output)

    # Load MIDI file
    try:
        seq = MIDISequence.load(str(midi_path))
    except Exception as e:
        raise CLIError(f"Failed to load MIDI file: {e}")

    # Find the plugin
    plugin = _find_plugin_by_name(args.name)
    plugin_type = plugin.get("type", "")

    # Check if it's an instrument plugin
    if plugin_type != "aumu":
        raise CLIError(f"Plugin '{plugin.get('name')}' is not an instrument plugin (type: {plugin_type})")

    sample_rate = args.sample_rate
    num_channels = 2
    extra_duration = args.duration if args.duration is not None else 1.0
    total_duration = seq.duration + extra_duration
    total_frames = int(total_duration * sample_rate)

    if not args.json:
        print(f"Rendering:  {midi_path.name}")
        print(f"Plugin:     {plugin.get('name')}")
        print(f"MIDI dur:   {seq.duration:.2f}s")
        print(f"Total dur:  {total_duration:.2f}s")
        print(f"Format:     {sample_rate}Hz, {num_channels}ch")

    # Create and initialize the AudioUnit
    comp_id = _get_component_id_for_plugin(plugin)
    au_id = capi.audio_component_instance_new(comp_id)

    try:
        capi.audio_unit_initialize(au_id)

        # Load preset if specified
        preset_name = None
        if args.preset:
            preset_name = _load_preset_by_name_or_number(au_id, args.preset)
            if preset_name is None:
                raise CLIError(f"Preset not found: {args.preset}")
            if not args.json:
                print(f"Preset:     {preset_name}")

        # Collect all MIDI events sorted by time
        all_events = []
        for track in seq.tracks:
            for event in track.events:
                all_events.append(event)
        all_events.sort(key=lambda e: e.time)

        # Render audio in chunks, scheduling MIDI events
        chunk_size = 512
        rendered_chunks = []
        event_index = 0

        for frame_offset in range(0, total_frames, chunk_size):
            frames_to_render = min(chunk_size, total_frames - frame_offset)
            chunk_start_time = frame_offset / sample_rate
            chunk_end_time = (frame_offset + frames_to_render) / sample_rate

            # Send MIDI events that fall in this chunk
            while event_index < len(all_events):
                event = all_events[event_index]
                if event.time >= chunk_end_time:
                    break

                # Calculate offset within this chunk
                offset_samples = int((event.time - chunk_start_time) * sample_rate)
                offset_samples = max(0, min(offset_samples, frames_to_render - 1))

                # Send MIDI event to instrument
                status_byte = (event.status & 0xF0) | (event.channel & 0x0F)
                try:
                    capi.music_device_midi_event(
                        au_id, status_byte, event.data1, event.data2, offset_samples
                    )
                except Exception:
                    pass  # Skip events that fail

                event_index += 1

            # Render this chunk (instrument generates audio from MIDI)
            # We pass silence as input since it's a generator
            silence = bytes(frames_to_render * num_channels * 4)  # float32 silence
            try:
                rendered = capi.audio_unit_render(
                    au_id, silence, frames_to_render, sample_rate, num_channels
                )
                rendered_chunks.append(rendered)
            except Exception:
                # If render fails, add silence
                rendered_chunks.append(silence)

            if not args.json:
                progress = min(100, int((frame_offset + frames_to_render) / total_frames * 100))
                print(f"\rRendering: {progress}%", end="", flush=True)

        rendered_data = b''.join(rendered_chunks)

        if not args.json:
            print("\rRendering: 100%")

        # Convert float32 to int16 for WAV output
        num_samples = len(rendered_data) // 4
        float_samples = struct.unpack(f'{num_samples}f', rendered_data)
        int_samples = [int(max(-32768, min(32767, s * 32768))) for s in float_samples]
        output_data = struct.pack(f'<{len(int_samples)}h', *int_samples)

        # Write output WAV file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), 'wb') as wav_out:
            wav_out.setnchannels(num_channels)
            wav_out.setsampwidth(2)  # 16-bit
            wav_out.setframerate(sample_rate)
            wav_out.writeframes(output_data)

        if args.json:
            output_json({
                "input": str(midi_path.absolute()),
                "output": str(output_path.absolute()),
                "plugin": plugin.get("name"),
                "midi_duration": seq.duration,
                "total_duration": total_duration,
                "sample_rate": sample_rate,
                "channels": num_channels,
                "events": len(all_events),
            })
        else:
            print(f"Output:     {output_path}")
            print(f"Events:     {len(all_events)}")

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
