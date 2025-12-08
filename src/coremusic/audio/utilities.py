#!/usr/bin/env python3
"""High-level audio processing utilities for CoreMusic.

This module provides convenient, high-level utilities for common audio
processing tasks, built on top of the CoreAudio APIs.

Features:
- Audio analysis (silence detection, peak detection, RMS calculation)
- Batch file processing and conversion
- Format conversion helpers
- File metadata extraction
- AudioStreamBasicDescription parsing
"""

import glob
import logging
import struct
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..objects import (NUMPY_AVAILABLE, AudioConverter, AudioFile, AudioFormat,
                       ExtendedAudioFile)
from ..utils.batch import batch_process_files, batch_process_parallel

logger = logging.getLogger(__name__)

if NUMPY_AVAILABLE:
    pass

# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # NOTE: AudioAnalyzer is intentionally not exported to avoid conflict with
    # coremusic.audio.analysis.AudioAnalyzer. Access via coremusic.audio.utilities.AudioAnalyzer
    "AudioEffectsChain",
    "AudioFormatPresets",
    "batch_convert",
    "convert_audio_file",
    "create_simple_effect_chain",
    "find_audio_unit_by_name",
    "get_audiounit_names",
    "list_available_audio_units",
    "parse_audio_stream_basic_description",
    "trim_audio",
    # Re-export batch processing utilities for convenience
    "batch_process_parallel",
    "batch_process_files",
]

# ============================================================================
# AudioStreamBasicDescription Parsing
# ============================================================================


def parse_audio_stream_basic_description(format_data: bytes) -> Dict[str, Any]:
    """Parse AudioStreamBasicDescription from raw bytes.

    AudioStreamBasicDescription (ASBD) is a CoreAudio structure that describes
    audio data format. This function parses the raw 40-byte structure returned
    by CoreAudio APIs into a Python dictionary.

    Args:
        format_data: 40 bytes of ASBD data from CoreAudio APIs
            (e.g., from audio_file_get_property with kAudioFilePropertyDataFormat)

    Returns:
        Dictionary containing parsed ASBD fields:
            - sample_rate (float): Sample rate in Hz
            - format_id (str): Format identifier FourCC (e.g., 'lpcm', 'aac ')
            - format_flags (int): Format-specific flags
            - bytes_per_packet (int): Bytes per packet
            - frames_per_packet (int): Frames per packet
            - bytes_per_frame (int): Bytes per frame
            - channels_per_frame (int): Number of channels
            - bits_per_channel (int): Bits per channel
            - reserved (int): Reserved field (usually 0)

    Raises:
        ValueError: If format_data is not exactly 40 bytes

    Example::

        import coremusic as cm
        import coremusic.capi as capi

        # Using functional API
        file_id = capi.audio_file_open_url("audio.wav")
        format_data = capi.audio_file_get_property(
            file_id,
            capi.get_audio_file_property_data_format()
        )
        asbd = cm.parse_audio_stream_basic_description(format_data)
        print(f"Sample rate: {asbd['sample_rate']} Hz")
        print(f"Channels: {asbd['channels_per_frame']}")
        capi.audio_file_close(file_id)

    Note:
        If using the object-oriented API, prefer the AudioFile.format property
        which automatically parses the format data::

            with cm.AudioFile("audio.wav") as audio:
                fmt = audio.format
                print(fmt.sample_rate, fmt.channels_per_frame)

    Structure Layout:
        The AudioStreamBasicDescription structure is 40 bytes::

            Offset  Size  Type     Field
            ------  ----  -------  -------------------
            0-7     8     Float64  mSampleRate
            8-11    4     UInt32   mFormatID (FourCC)
            12-15   4     UInt32   mFormatFlags
            16-19   4     UInt32   mBytesPerPacket
            20-23   4     UInt32   mFramesPerPacket
            24-27   4     UInt32   mBytesPerFrame
            28-31   4     UInt32   mChannelsPerFrame
            32-35   4     UInt32   mBitsPerChannel
            36-39   4     UInt32   mReserved
    """
    if len(format_data) != 40:
        raise ValueError(
            f"AudioStreamBasicDescription must be exactly 40 bytes, got {len(format_data)}"
        )

    # Parse sample rate (Float64, little-endian)
    sample_rate: float = struct.unpack("<d", format_data[0:8])[0]

    # Parse format ID (FourCC, stored as big-endian UInt32)
    # Reverse bytes to get correct FourCC string representation
    format_id_bytes = format_data[8:12]
    format_id: str = format_id_bytes[::-1].decode("ascii")

    # Parse remaining UInt32 fields (little-endian)
    (
        format_flags,
        bytes_per_packet,
        frames_per_packet,
        bytes_per_frame,
        channels_per_frame,
        bits_per_channel,
        reserved,
    ) = struct.unpack("<7I", format_data[12:40])

    return {
        "sample_rate": sample_rate,
        "format_id": format_id,
        "format_flags": format_flags,
        "bytes_per_packet": bytes_per_packet,
        "frames_per_packet": frames_per_packet,
        "bytes_per_frame": bytes_per_frame,
        "channels_per_frame": channels_per_frame,
        "bits_per_channel": bits_per_channel,
        "reserved": reserved,
    }


# ============================================================================
# NOTE: AudioAnalyzer class has been moved to coremusic.audio.analysis
# Use: from coremusic.audio.analysis import AudioAnalyzer
# Or:  import coremusic as cm; cm.AudioAnalyzer.detect_silence(...)
# ============================================================================

# ============================================================================
# Batch Processing Utilities
# ============================================================================


def batch_convert(
    input_pattern: str,
    output_format: AudioFormat,
    output_dir: Optional[str] = None,
    output_extension: str = "wav",
    overwrite: bool = False,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> List[str]:
    """Batch convert audio files to a specified format.

    Args:
        input_pattern: Glob pattern for input files (e.g., "*.mp3", "audio/*.wav")
        output_format: Target AudioFormat for conversion
        output_dir: Output directory (default: same as input)
        output_extension: Output file extension (default: "wav")
        overwrite: Whether to overwrite existing files (default: False)
        progress_callback: Optional callback(filename, current, total)

    Returns:
        List of successfully converted output file paths

    Example:
        ```python
        import coremusic as cm

        # Convert all MP3 files to 16-bit 44.1kHz WAV
        output_format = cm.AudioFormat(
            sample_rate=44100.0,
            format_id='lpcm',
            format_flags=12,
            bytes_per_packet=4,
            frames_per_packet=1,
            bytes_per_frame=4,
            channels_per_frame=2,
            bits_per_channel=16
        )

        converted = cm.batch_convert(
            input_pattern="input/*.mp3",
            output_format=output_format,
            output_dir="output/",
            progress_callback=lambda f, c, t: print(f"Converting {f} ({c}/{t})")
        )
        print(f"Converted {len(converted)} files")
        ```
    """
    # Find all matching files
    input_files = glob.glob(input_pattern)
    if not input_files:
        return []

    total_files = len(input_files)
    converted_files = []

    for i, input_path_str in enumerate(input_files, 1):
        input_path = Path(input_path_str)

        # Determine output path
        if output_dir:
            output_path_obj = Path(output_dir) / f"{input_path.stem}.{output_extension}"
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            output_path = str(output_path_obj)
        else:
            output_path = str(input_path.with_suffix(f".{output_extension}"))

        # Skip if file exists and not overwriting
        output_path_check = Path(output_path)
        if output_path_check.exists() and not overwrite:
            continue

        # Call progress callback
        if progress_callback:
            progress_callback(str(input_path), i, total_files)

        # Convert file
        try:
            convert_audio_file(str(input_path), str(output_path), output_format)
            converted_files.append(str(output_path))
        except Exception as e:
            logger.warning(f"Failed to convert {input_path}: {e}")
            continue

    return converted_files


def convert_audio_file(
    input_path: str, output_path: str, output_format: AudioFormat
) -> None:
    """Convert a single audio file to a different format.

    Supports ALL conversion types:
    - Channel count (stereo <-> mono)
    - Sample rate (e.g., 44.1kHz -> 48kHz)
    - Bit depth (e.g., 16-bit -> 24-bit)
    - Combinations of the above

    Args:
        input_path: Input file path
        output_path: Output file path
        output_format: Target AudioFormat

    Example:
        ```python
        import coremusic as cm

        # Convert to different sample rate
        cm.convert_audio_file(
            "input_44100.wav",
            "output_48000.wav",
            cm.AudioFormat(48000.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)
        )

        # Convert to mono AND change sample rate
        cm.convert_audio_file(
            "stereo_44100.wav",
            "mono_48000.wav",
            cm.AudioFormat(48000.0, 'lpcm', channels_per_frame=1, bits_per_channel=16)
        )
        ```
    """
    # Read source file
    with AudioFile(input_path) as input_file:
        source_format = input_file.format

        # If formats match exactly, just copy
        if _formats_match(source_format, output_format):
            import shutil

            shutil.copy(input_path, output_path)
            return

        # Read all audio data
        audio_data, packet_count = input_file.read_packets(0, 999999999)  # type: ignore[call-arg]

        # Determine which conversion method to use
        needs_complex_conversion = (
            source_format.sample_rate != output_format.sample_rate
            or source_format.bits_per_channel != output_format.bits_per_channel
        )

        # Convert using AudioConverter
        with AudioConverter(source_format, output_format) as converter:
            if needs_complex_conversion:
                # Use callback-based API for complex conversions
                converted_data = converter.convert_with_callback(  # type: ignore[attr-defined]
                    audio_data, packet_count
                )
            else:
                # Use simple buffer API for channel-only conversions
                converted_data = converter.convert(audio_data)

        # Calculate number of frames from converted data
        num_frames = len(converted_data) // output_format.bytes_per_frame

        # Write to output file
        from .. import capi

        output_ext_file = ExtendedAudioFile.create(
            output_path, capi.get_audio_file_wave_type(), output_format
        )
        try:
            output_ext_file.write(num_frames, converted_data)
        finally:
            output_ext_file.close()


def _formats_match(fmt1: AudioFormat, fmt2: AudioFormat) -> bool:
    """Check if two formats are identical"""
    return (
        fmt1.sample_rate == fmt2.sample_rate
        and fmt1.channels_per_frame == fmt2.channels_per_frame
        and fmt1.bits_per_channel == fmt2.bits_per_channel
        and fmt1.format_id == fmt2.format_id
    )


# ============================================================================
# Format Conversion Helpers
# ============================================================================


class AudioFormatPresets:
    """Common audio format presets for easy format conversion.

    Example:
        ```python
        import coremusic as cm

        # Use predefined formats
        cm.convert_audio_file("input.mp3", "output.wav", cm.AudioFormatPresets.wav_44100_stereo())
        cm.convert_audio_file("input.wav", "output_mono.wav", cm.AudioFormatPresets.wav_44100_mono())
        ```
    """

    @staticmethod
    def wav_44100_stereo() -> AudioFormat:
        """Standard CD quality WAV: 44.1kHz, 16-bit, stereo"""
        return AudioFormat(
            sample_rate=44100.0,
            format_id="lpcm",
            format_flags=12,  # kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked
            bytes_per_packet=4,
            frames_per_packet=1,
            bytes_per_frame=4,
            channels_per_frame=2,
            bits_per_channel=16,
        )

    @staticmethod
    def wav_44100_mono() -> AudioFormat:
        """Mono WAV: 44.1kHz, 16-bit, mono"""
        return AudioFormat(
            sample_rate=44100.0,
            format_id="lpcm",
            format_flags=12,
            bytes_per_packet=2,
            frames_per_packet=1,
            bytes_per_frame=2,
            channels_per_frame=1,
            bits_per_channel=16,
        )

    @staticmethod
    def wav_48000_stereo() -> AudioFormat:
        """Pro audio WAV: 48kHz, 16-bit, stereo"""
        return AudioFormat(
            sample_rate=48000.0,
            format_id="lpcm",
            format_flags=12,
            bytes_per_packet=4,
            frames_per_packet=1,
            bytes_per_frame=4,
            channels_per_frame=2,
            bits_per_channel=16,
        )

    @staticmethod
    def wav_96000_stereo() -> AudioFormat:
        """High-res audio WAV: 96kHz, 24-bit, stereo"""
        return AudioFormat(
            sample_rate=96000.0,
            format_id="lpcm",
            format_flags=12,
            bytes_per_packet=6,
            frames_per_packet=1,
            bytes_per_frame=6,
            channels_per_frame=2,
            bits_per_channel=24,
        )


# ============================================================================
# Audio File Operations
# ============================================================================


def trim_audio(
    input_path: str,
    output_path: str,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
) -> None:
    """Trim an audio file to a specific time range.

    Args:
        input_path: Input file path
        output_path: Output file path
        start_time: Start time in seconds (default: 0.0)
        end_time: End time in seconds (default: end of file)

    Example:
        ```python
        # Extract first 30 seconds of audio
        cm.trim_audio("input.wav", "output.wav", start_time=0.0, end_time=30.0)

        # Skip first 10 seconds
        cm.trim_audio("input.wav", "output.wav", start_time=10.0)
        ```
    """
    with AudioFile(input_path) as input_file:
        format = input_file.format
        sample_rate = format.sample_rate

        # Calculate packet range
        start_packet = int(start_time * sample_rate)

        if end_time is not None:
            end_packet = int(end_time * sample_rate)
            packet_count = end_packet - start_packet
        else:
            packet_count = None  # Read to end

        # Read trimmed data
        data, actual_count = input_file.read_packets(  # type: ignore[call-arg]
            start_packet, packet_count or 999999999
        )

        # Write to output file using ExtendedAudioFile
        from .. import capi

        output_ext_file = ExtendedAudioFile.create(
            output_path,
            capi.get_audio_file_wave_type(),  # WAV file
            format,
        )
        try:
            output_ext_file.write(actual_count, data)  # type: ignore[arg-type]
        finally:
            output_ext_file.close()


# ============================================================================
# Audio Effects Chain
# ============================================================================


class AudioEffectsChain:
    """High-level audio effects chain using AUGraph.

    Provides a simplified API for creating and managing chains of AudioUnit
    effects for real-time audio processing.

    Example:
        ```python
        import coremusic as cm

        # Create an effects chain
        chain = cm.AudioEffectsChain()

        # Add effects (example effect types)
        reverb_node = chain.add_effect('aumu', 'rvb2', 'appl')  # Reverb
        eq_node = chain.add_effect('aufx', 'eqal', 'appl')      # EQ

        # Connect to output
        output_node = chain.add_output()
        chain.connect(reverb_node, eq_node)
        chain.connect(eq_node, output_node)

        # Initialize and start
        chain.initialize()
        chain.start()

        # ... process audio ...

        # Cleanup
        chain.stop()
        chain.dispose()
        ```
    """

    def __init__(self):
        """Create a new audio effects chain"""
        from ..objects import AUGraph

        self._graph = AUGraph()
        self._nodes = {}  # Map of node_id -> description

    @property
    def graph(self):
        """Get the underlying AUGraph"""
        return self._graph

    def add_effect(
        self,
        effect_type: str,
        effect_subtype: str,
        manufacturer: str = "appl",
        flags: int = 0,
    ) -> int:
        """Add an audio effect to the chain.

        Args:
            effect_type: AudioUnit type (4-char code or string)
            effect_subtype: AudioUnit subtype (4-char code or string)
            manufacturer: Manufacturer code (default: 'appl' for Apple)
            flags: Component flags (default: 0)

        Returns:
            Node ID for this effect

        Example:
            ```python
            # Add a reverb effect
            reverb = chain.add_effect('aumu', 'rvb2', 'appl')

            # Add an EQ effect
            eq = chain.add_effect('aufx', 'eqal', 'appl')

            # Add a dynamics processor
            dynamics = chain.add_effect('aufx', 'dcmp', 'appl')
            ```
        """
        from ..objects import AudioComponentDescription

        desc = AudioComponentDescription(  # type: ignore[call-arg]
            type=effect_type,
            subtype=effect_subtype,
            manufacturer=manufacturer,
            flags=flags,
            flags_mask=0,
        )

        node_id = self._graph.add_node(desc)
        self._nodes[node_id] = desc
        return node_id

    def add_effect_by_name(self, name: str) -> Optional[int]:
        """Add an audio effect to the chain by name.

        This searches for an AudioUnit matching the given name and adds it
        to the chain.

        Args:
            name: AudioUnit name (e.g., 'AUDelay', 'Reverb', 'AUGraphicEQ')

        Returns:
            Node ID for this effect, or None if not found

        Example:
            ```python
            # Add effects by name instead of FourCC codes
            delay = chain.add_effect_by_name('AUDelay')
            reverb = chain.add_effect_by_name('Reverb')
            eq = chain.add_effect_by_name('AUGraphicEQ')
            ```
        """
        component = find_audio_unit_by_name(name)
        if component is None:
            return None

        # Extract FourCC codes from the component
        desc = component._description
        return self.add_effect(desc.type, desc.subtype, desc.manufacturer)

    def add_output(
        self, output_type: str = "auou", output_subtype: str = "def "
    ) -> int:
        """Add an output node to the chain.

        Args:
            output_type: Output type (default: 'auou' for AudioUnit Output)
            output_subtype: Output subtype (default: 'def ' for default output)

        Returns:
            Node ID for the output

        Example:
            ```python
            # Add default output
            output = chain.add_output()

            # Add system output
            output = chain.add_output('auou', 'sys ')
            ```
        """
        return self.add_effect(output_type, output_subtype, "appl")

    def connect(
        self, source_node: int, dest_node: int, source_bus: int = 0, dest_bus: int = 0
    ) -> None:
        """Connect two nodes in the effects chain.

        Args:
            source_node: Source node ID
            dest_node: Destination node ID
            source_bus: Source output bus (default: 0)
            dest_bus: Destination input bus (default: 0)

        Example:
            ```python
            # Connect reverb to EQ
            chain.connect(reverb_node, eq_node)

            # Connect EQ to output
            chain.connect(eq_node, output_node)
            ```
        """
        self._graph.connect(source_node, source_bus, dest_node, dest_bus)  # type: ignore[attr-defined]

    def disconnect(self, dest_node: int, dest_bus: int = 0) -> None:
        """Disconnect a node input.

        Args:
            dest_node: Destination node ID
            dest_bus: Destination input bus (default: 0)
        """
        self._graph.disconnect(dest_node, dest_bus)  # type: ignore[attr-defined]

    def remove_node(self, node_id: int) -> None:
        """Remove a node from the chain.

        Args:
            node_id: Node ID to remove
        """
        if node_id in self._nodes:
            del self._nodes[node_id]
        self._graph.remove_node(node_id)

    def open(self) -> "AudioEffectsChain":
        """Open the graph (opens all AudioUnits).

        Returns:
            Self for method chaining
        """
        self._graph.open()
        return self

    def initialize(self) -> "AudioEffectsChain":
        """Initialize the graph (prepares for rendering).

        Returns:
            Self for method chaining
        """
        self._graph.initialize()
        return self

    def start(self) -> None:
        """Start audio rendering"""
        self._graph.start()

    def stop(self) -> None:
        """Stop audio rendering"""
        self._graph.stop()

    def dispose(self) -> None:
        """Dispose of the graph and all resources"""
        try:
            if self._graph.is_running:
                self.stop()
        except Exception:
            pass
        self._graph.dispose()

    def __enter__(self) -> "AudioEffectsChain":
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.dispose()

    @property
    def is_open(self) -> bool:
        """Check if the graph is open"""
        return self._graph.is_open

    @property
    def is_initialized(self) -> bool:
        """Check if the graph is initialized"""
        return self._graph.is_initialized

    @property
    def is_running(self) -> bool:
        """Check if the graph is running"""
        return self._graph.is_running

    @property
    def node_count(self) -> int:
        """Get the number of nodes in the chain"""
        return len(self._nodes)


def create_simple_effect_chain(
    effect_types: List[Tuple[str, str, str]], auto_connect: bool = True
) -> AudioEffectsChain:
    """Create a simple linear effects chain.

    Args:
        effect_types: List of (type, subtype, manufacturer) tuples
        auto_connect: Automatically connect effects in order (default: True)

    Returns:
        Configured AudioEffectsChain

    Example:
        ```python
        import coremusic as cm

        # Create a reverb -> EQ -> output chain
        chain = cm.create_simple_effect_chain([
            ('aumu', 'rvb2', 'appl'),  # Reverb
            ('aufx', 'eqal', 'appl'),  # EQ
        ])

        # Initialize and start
        chain.open().initialize().start()

        # ... process audio ...

        chain.dispose()
        ```
    """
    chain = AudioEffectsChain()

    # Add all effects
    nodes = []
    for effect_type, effect_subtype, manufacturer in effect_types:
        node_id = chain.add_effect(effect_type, effect_subtype, manufacturer)
        nodes.append(node_id)

    # Add output
    output_node = chain.add_output()
    nodes.append(output_node)

    # Auto-connect in linear chain
    if auto_connect and len(nodes) > 1:
        for i in range(len(nodes) - 1):
            chain.connect(nodes[i], nodes[i + 1])

    return chain


# ============================================================================
# AudioUnit Discovery by Name
# ============================================================================


def find_audio_unit_by_name(name: str, case_sensitive: bool = False):
    """Find an AudioUnit by name (searches all available AudioComponents).

    This function iterates through all available AudioComponents and matches
    by name. Returns an AudioComponent object that can be used to create
    AudioUnit instances.

    Args:
        name: Name or partial name to search for (e.g., 'AUDelay', 'Reverb')
        case_sensitive: Whether to do case-sensitive matching (default: False)

    Returns:
        AudioComponent object, or None if not found

    Example:
        ```python
        import coremusic as cm

        # Find AUDelay and create an instance
        component = cm.find_audio_unit_by_name('AUDelay')
        if component:
            audio_unit = component.create_instance()
            audio_unit.initialize()
            # Use the audio unit...
            audio_unit.dispose()

        # Or get the FourCC codes if needed
        if component:
            desc = component._description
            print(f"Type: {desc.type}, Subtype: {desc.subtype}")
        ```
    """
    from .. import capi
    from ..objects import AudioComponent, AudioComponentDescription

    # Wildcard description to iterate through all components
    desc_dict = {
        "type": 0,
        "subtype": 0,
        "manufacturer": 0,
        "flags": 0,
        "flags_mask": 0,
    }

    component_id: Optional[int] = 0  # Start with NULL
    search_name = name if case_sensitive else name.lower()

    while True:
        # Find next component (passing previous component for iteration)
        component_id = capi.audio_component_find_next(desc_dict, component_id or 0)

        if component_id is None or component_id == 0:
            break

        # Get component name
        component_name = capi.audio_component_copy_name(component_id)
        if component_name:
            match_name = component_name if case_sensitive else component_name.lower()

            # Check if name matches (substring match)
            if search_name in match_name:
                # Get the description
                desc_dict_result = capi.audio_component_get_description(component_id)

                # Convert to FourCC strings and create description
                type_fourcc = capi.int_to_fourchar(desc_dict_result["type"])
                subtype_fourcc = capi.int_to_fourchar(desc_dict_result["subtype"])
                manufacturer_fourcc = capi.int_to_fourchar(
                    desc_dict_result["manufacturer"]
                )

                # Create AudioComponent object
                desc = AudioComponentDescription(  # type: ignore[call-arg]
                    type=type_fourcc,
                    subtype=subtype_fourcc,
                    manufacturer=manufacturer_fourcc,
                    flags=desc_dict_result["flags"],
                    flags_mask=desc_dict_result["flags_mask"],
                )

                component = AudioComponent(desc)
                component._set_object_id(component_id)
                return component

    return None


def list_available_audio_units(
    filter_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List all available AudioUnits with their names and FourCC codes.

    Args:
        filter_type: Optional type filter (e.g., 'aumu', 'aufx', 'auou')

    Returns:
        List of dictionaries with keys: 'name', 'type', 'subtype', 'manufacturer'

    Example:
        ```python
        import coremusic as cm

        # List all AudioUnits
        units = cm.list_available_audio_units()
        for unit in units:
            print(f"{unit['name']}: {unit['type']}/{unit['subtype']}/{unit['manufacturer']}")

        # List only music effects
        effects = cm.list_available_audio_units(filter_type='aumu')
        ```
    """
    from .. import capi

    results = []
    type_int = capi.fourchar_to_int(filter_type) if filter_type else 0

    desc_dict = {
        "type": type_int,
        "subtype": 0,
        "manufacturer": 0,
        "flags": 0,
        "flags_mask": 0,
    }

    # Iterate through all components
    component_id: Optional[int] = 0  # Start with NULL

    while True:
        component_id = capi.audio_component_find_next(desc_dict, component_id or 0)

        if component_id is None or component_id == 0:
            break

        component_name = capi.audio_component_copy_name(component_id)
        if component_name:
            desc = capi.audio_component_get_description(component_id)

            results.append(
                {
                    "name": component_name,
                    "type": capi.int_to_fourchar(desc["type"]),
                    "subtype": capi.int_to_fourchar(desc["subtype"]),
                    "manufacturer": capi.int_to_fourchar(desc["manufacturer"]),
                    "flags": desc["flags"],
                }
            )

    return results


def get_audiounit_names(filter_type: Optional[str] = None) -> List[str]:
    """Get a list of all available AudioUnit names.

    This is a convenience function that returns just the names of AudioUnits,
    making it easy to see what's available or search for specific units.

    Args:
        filter_type: Optional type filter (e.g., 'aumu', 'aufx', 'auou')

    Returns:
        List of AudioUnit names as strings

    Example:
        ```python
        import coremusic as cm

        # Get all AudioUnit names
        names = cm.get_audiounit_names()
        print(f"Found {len(names)} AudioUnits:")
        for name in names[:10]:
            print(f"  - {name}")

        # Filter by type
        effects = cm.get_audiounit_names(filter_type='aufx')
        print(f"Audio effects: {effects[:5]}")

        # Check if specific unit is available
        if 'AUDelay' in names:
            print("AUDelay is available!")

        # Search for units containing 'Reverb'
        reverbs = [name for name in names if 'reverb' in name.lower()]
        print(f"Found {len(reverbs)} reverb units: {reverbs}")
        ```
    """
    units = list_available_audio_units(filter_type)
    return [unit["name"] for unit in units]
