#!/usr/bin/env python3
"""High-level MIDI utilities for file I/O, sequencing, and routing.

This module provides high-level MIDI operations beyond CoreMIDI basics:
- MIDI file reading/writing (Standard MIDI File format)
- MIDI sequencing and playback
- MIDI routing matrix
- MIDI message builders
- MIDI transformations

Example:
    >>> seq = MIDISequence(tempo=120.0)
    >>> track = seq.add_track("Melody")
    >>> track.add_note(0.0, 60, 100, 0.5)  # C4 for 0.5 seconds
    >>> seq.save("output.mid")
"""

import logging
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================


class MIDIFileFormat(IntEnum):
    """Standard MIDI File format types."""

    SINGLE_TRACK = 0  # Single multi-channel track
    MULTI_TRACK = 1  # Multiple simultaneous tracks
    MULTI_SONG = 2  # Multiple sequential tracks


class MIDIStatus(IntEnum):
    """MIDI status byte values."""

    NOTE_OFF = 0x80
    NOTE_ON = 0x90
    POLY_AFTERTOUCH = 0xA0
    CONTROL_CHANGE = 0xB0
    PROGRAM_CHANGE = 0xC0
    CHANNEL_AFTERTOUCH = 0xD0
    PITCH_BEND = 0xE0
    SYSTEM = 0xF0


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class MIDIEvent:
    """MIDI event in a sequence.

    Attributes:
        time: Time in seconds (or ticks if using delta_time)
        status: MIDI status byte (upper nibble)
        channel: MIDI channel (0-15)
        data1: First data byte (0-127)
        data2: Second data byte (0-127)
    """

    time: float
    status: int
    channel: int
    data1: int
    data2: int = 0

    @property
    def is_note_on(self) -> bool:
        """Check if this is a note on event."""
        return self.status == MIDIStatus.NOTE_ON and self.data2 > 0

    @property
    def is_note_off(self) -> bool:
        """Check if this is a note off event."""
        return self.status == MIDIStatus.NOTE_OFF or (
            self.status == MIDIStatus.NOTE_ON and self.data2 == 0
        )

    @property
    def is_control_change(self) -> bool:
        """Check if this is a control change event."""
        return self.status == MIDIStatus.CONTROL_CHANGE

    @property
    def is_program_change(self) -> bool:
        """Check if this is a program change event."""
        return self.status == MIDIStatus.PROGRAM_CHANGE

    def to_bytes(self) -> bytes:
        """Convert to MIDI byte representation.

        Returns:
            MIDI message as bytes
        """
        status_byte = (self.status & 0xF0) | (self.channel & 0x0F)
        if self.status == MIDIStatus.PROGRAM_CHANGE or self.status == MIDIStatus.CHANNEL_AFTERTOUCH:
            return bytes([status_byte, self.data1 & 0x7F])
        else:
            return bytes([status_byte, self.data1 & 0x7F, self.data2 & 0x7F])

    @classmethod
    def from_bytes(cls, data: bytes, time: float = 0.0) -> "MIDIEvent":
        """Create MIDIEvent from MIDI message bytes.

        Args:
            data: MIDI message bytes
            time: Event time in seconds

        Returns:
            MIDIEvent instance
        """
        if len(data) < 1:
            raise ValueError("MIDI message must have at least 1 byte")

        status = data[0] & 0xF0
        channel = data[0] & 0x0F
        data1 = data[1] if len(data) > 1 else 0
        data2 = data[2] if len(data) > 2 else 0

        return cls(time, status, channel, data1, data2)


# ============================================================================
# MIDI Track
# ============================================================================


class MIDITrack:
    """MIDI track with events.

    A track contains a sequence of MIDI events that can be played back
    or saved to a MIDI file.

    Example:
        >>> track = MIDITrack("Melody")
        >>> track.add_note(0.0, 60, 100, 0.5)  # C4
        >>> track.add_note(0.5, 64, 100, 0.5)  # E4
        >>> track.add_note(1.0, 67, 100, 0.5)  # G4
    """

    def __init__(self, name: str = ""):
        """Initialize MIDI track.

        Args:
            name: Track name
        """
        self.name = name
        self.events: List[MIDIEvent] = []
        self.program: int = 0  # MIDI program/patch
        self.channel: int = 0  # Default channel

    def add_note(
        self,
        time: float,
        note: int,
        velocity: int,
        duration: float,
        channel: Optional[int] = None,
    ) -> None:
        """Add note on/off events.

        Args:
            time: Start time in seconds
            note: MIDI note number (0-127)
            velocity: Note velocity (0-127)
            duration: Note duration in seconds
            channel: MIDI channel (0-15), or None to use track default
        """
        ch = channel if channel is not None else self.channel

        # Validate parameters
        if not 0 <= note <= 127:
            raise ValueError(f"Note must be 0-127, got {note}")
        if not 0 <= velocity <= 127:
            raise ValueError(f"Velocity must be 0-127, got {velocity}")
        if not 0 <= ch <= 15:
            raise ValueError(f"Channel must be 0-15, got {ch}")
        if duration < 0:
            raise ValueError(f"Duration must be >= 0, got {duration}")

        # Note On
        self.events.append(
            MIDIEvent(time, MIDIStatus.NOTE_ON, ch, note, velocity)
        )

        # Note Off
        self.events.append(
            MIDIEvent(time + duration, MIDIStatus.NOTE_OFF, ch, note, 0)
        )

        # Keep events sorted by time
        self.events.sort(key=lambda e: e.time)

    def add_control_change(
        self,
        time: float,
        controller: int,
        value: int,
        channel: Optional[int] = None,
    ) -> None:
        """Add control change event.

        Args:
            time: Time in seconds
            controller: Controller number (0-127)
            value: Controller value (0-127)
            channel: MIDI channel (0-15), or None to use track default
        """
        ch = channel if channel is not None else self.channel

        if not 0 <= controller <= 127:
            raise ValueError(f"Controller must be 0-127, got {controller}")
        if not 0 <= value <= 127:
            raise ValueError(f"Value must be 0-127, got {value}")
        if not 0 <= ch <= 15:
            raise ValueError(f"Channel must be 0-15, got {ch}")

        self.events.append(
            MIDIEvent(time, MIDIStatus.CONTROL_CHANGE, ch, controller, value)
        )
        self.events.sort(key=lambda e: e.time)

    def add_program_change(
        self,
        time: float,
        program: int,
        channel: Optional[int] = None,
    ) -> None:
        """Add program change event.

        Args:
            time: Time in seconds
            program: Program number (0-127)
            channel: MIDI channel (0-15), or None to use track default
        """
        ch = channel if channel is not None else self.channel

        if not 0 <= program <= 127:
            raise ValueError(f"Program must be 0-127, got {program}")
        if not 0 <= ch <= 15:
            raise ValueError(f"Channel must be 0-15, got {ch}")

        self.events.append(
            MIDIEvent(time, MIDIStatus.PROGRAM_CHANGE, ch, program, 0)
        )
        self.program = program
        self.events.sort(key=lambda e: e.time)

    def add_pitch_bend(
        self,
        time: float,
        value: int,
        channel: Optional[int] = None,
    ) -> None:
        """Add pitch bend event.

        Args:
            time: Time in seconds
            value: Pitch bend value (0-16383, 8192 = center)
            channel: MIDI channel (0-15), or None to use track default
        """
        ch = channel if channel is not None else self.channel

        if not 0 <= value <= 16383:
            raise ValueError(f"Pitch bend value must be 0-16383, got {value}")
        if not 0 <= ch <= 15:
            raise ValueError(f"Channel must be 0-15, got {ch}")

        # Split 14-bit value into LSB and MSB
        lsb = value & 0x7F
        msb = (value >> 7) & 0x7F

        self.events.append(
            MIDIEvent(time, MIDIStatus.PITCH_BEND, ch, lsb, msb)
        )
        self.events.sort(key=lambda e: e.time)

    @property
    def duration(self) -> float:
        """Total track duration in seconds."""
        if not self.events:
            return 0.0
        return max(e.time for e in self.events)

    def clear(self) -> None:
        """Clear all events from the track."""
        self.events.clear()

    def __len__(self) -> int:
        """Return number of events in track."""
        return len(self.events)

    def __repr__(self) -> str:
        """String representation of track."""
        return f"MIDITrack(name={self.name!r}, events={len(self.events)}, duration={self.duration:.2f}s)"


# ============================================================================
# MIDI Sequence
# ============================================================================


class MIDISequence:
    """MIDI sequence (collection of tracks).

    A sequence represents a complete MIDI composition with multiple tracks,
    tempo, and time signature information. Can be saved to or loaded from
    Standard MIDI Files.

    Example:
        >>> seq = MIDISequence(tempo=120.0)
        >>> track = seq.add_track("Piano")
        >>> track.add_note(0.0, 60, 100, 0.5)
        >>> seq.save("output.mid")
    """

    def __init__(
        self,
        tempo: float = 120.0,
        time_signature: Tuple[int, int] = (4, 4),
    ):
        """Initialize MIDI sequence.

        Args:
            tempo: Tempo in BPM (beats per minute)
            time_signature: Time signature (numerator, denominator)
        """
        if tempo <= 0:
            raise ValueError(f"Tempo must be > 0, got {tempo}")
        if time_signature[0] <= 0 or time_signature[1] <= 0:
            raise ValueError(f"Invalid time signature: {time_signature}")

        self.tempo = tempo
        self.time_signature = time_signature
        self.tracks: List[MIDITrack] = []
        self.ppq = 480  # Pulses per quarter note (MIDI resolution)

    def add_track(self, name: str = "") -> MIDITrack:
        """Add new track to sequence.

        Args:
            name: Track name

        Returns:
            The newly created track
        """
        track = MIDITrack(name)
        self.tracks.append(track)
        return track

    def _write_variable_length(self, value: int) -> bytes:
        """Write variable-length quantity (VLQ) for MIDI.

        Args:
            value: Integer value to encode

        Returns:
            Encoded bytes
        """
        result = bytearray()
        result.append(value & 0x7F)

        value >>= 7
        while value > 0:
            result.insert(0, (value & 0x7F) | 0x80)
            value >>= 7

        return bytes(result)

    def _write_track(self, track: MIDITrack) -> bytes:
        """Write track as MIDI track chunk.

        Args:
            track: Track to write

        Returns:
            Track chunk bytes (MTrk + data)
        """
        data = bytearray()

        # Write track name meta event if present
        if track.name:
            data.extend(self._write_variable_length(0))  # Delta time
            data.append(0xFF)  # Meta event
            data.append(0x03)  # Track name
            name_bytes = track.name.encode('utf-8')
            data.extend(self._write_variable_length(len(name_bytes)))
            data.extend(name_bytes)

        # Sort events by time
        sorted_events = sorted(track.events, key=lambda e: e.time)

        # Write events
        last_time = 0
        for event in sorted_events:
            # Convert time to ticks
            ticks = int(event.time * self.ppq * (self.tempo / 60.0))
            delta_ticks = max(0, ticks - last_time)
            last_time = ticks

            # Write delta time
            data.extend(self._write_variable_length(delta_ticks))

            # Write MIDI event
            data.extend(event.to_bytes())

        # End of track meta event
        data.extend(self._write_variable_length(0))  # Delta time
        data.append(0xFF)  # Meta event
        data.append(0x2F)  # End of track
        data.append(0x00)  # Length

        # Build track chunk
        chunk = bytearray()
        chunk.extend(b'MTrk')
        chunk.extend(struct.pack('>I', len(data)))
        chunk.extend(data)

        return bytes(chunk)

    def save(
        self,
        filename: str,
        format: MIDIFileFormat = MIDIFileFormat.MULTI_TRACK,
    ) -> None:
        """Save sequence as Standard MIDI File.

        Args:
            filename: Output file path
            format: MIDI file format (0=single track, 1=multi track, 2=multi song)
        """
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            # Write MThd header
            # For format 1, we add a tempo track, so total is len(tracks) + 1
            num_tracks = len(self.tracks)
            if format == MIDIFileFormat.MULTI_TRACK and self.tracks:
                num_tracks += 1  # Account for tempo track

            f.write(b'MThd')
            f.write(struct.pack('>I', 6))  # Header length
            f.write(struct.pack('>H', int(format)))  # Format
            f.write(struct.pack('>H', num_tracks))  # Number of tracks
            f.write(struct.pack('>H', self.ppq))  # Ticks per quarter note

            # Write tempo track (track 0 for format 1)
            if format == MIDIFileFormat.MULTI_TRACK and self.tracks:
                tempo_track_data = bytearray()

                # Time signature meta event
                tempo_track_data.extend(self._write_variable_length(0))
                tempo_track_data.append(0xFF)  # Meta event
                tempo_track_data.append(0x58)  # Time signature
                tempo_track_data.append(0x04)  # Length
                tempo_track_data.append(self.time_signature[0])  # Numerator
                # Denominator as power of 2
                denom_power = 0
                denom = self.time_signature[1]
                while denom > 1:
                    denom >>= 1
                    denom_power += 1
                tempo_track_data.append(denom_power)
                tempo_track_data.append(24)  # MIDI clocks per metronome click
                tempo_track_data.append(8)  # 32nd notes per quarter note

                # Tempo meta event
                tempo_track_data.extend(self._write_variable_length(0))
                tempo_track_data.append(0xFF)  # Meta event
                tempo_track_data.append(0x51)  # Tempo
                tempo_track_data.append(0x03)  # Length
                microseconds_per_quarter = int(60_000_000 / self.tempo)
                tempo_track_data.extend(struct.pack('>I', microseconds_per_quarter)[1:])

                # End of track
                tempo_track_data.extend(self._write_variable_length(0))
                tempo_track_data.append(0xFF)
                tempo_track_data.append(0x2F)
                tempo_track_data.append(0x00)

                # Write tempo track chunk
                f.write(b'MTrk')
                f.write(struct.pack('>I', len(tempo_track_data)))
                f.write(tempo_track_data)

            # Write tracks
            for track in self.tracks:
                f.write(self._write_track(track))

        logger.info(f"Saved MIDI file: {filename} ({len(self.tracks)} tracks)")

    @classmethod
    def load(cls, filename: str) -> "MIDISequence":
        """Load Standard MIDI File.

        Args:
            filename: MIDI file path

        Returns:
            Loaded MIDISequence

        Raises:
            ValueError: If file format is invalid
        """
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"MIDI file not found: {filename}")

        sequence = cls()

        with open(path, 'rb') as f:
            # Parse MThd header
            chunk_type = f.read(4)
            if chunk_type != b'MThd':
                raise ValueError(f"Invalid MIDI file: expected MThd, got {chunk_type!r}")

            header_length = struct.unpack('>I', f.read(4))[0]
            if header_length != 6:
                raise ValueError(f"Invalid header length: {header_length}")

            format_type = struct.unpack('>H', f.read(2))[0]
            num_tracks = struct.unpack('>H', f.read(2))[0]
            division = struct.unpack('>H', f.read(2))[0]

            # Handle division (we only support ticks per quarter note)
            if division & 0x8000:
                raise ValueError("SMPTE time division not supported")
            sequence.ppq = division

            logger.info(f"Loading MIDI file: format={format_type}, tracks={num_tracks}, ppq={division}")

            # Parse MTrk chunks
            for track_num in range(num_tracks):
                chunk_type = f.read(4)
                if chunk_type != b'MTrk':
                    logger.warning(f"Expected MTrk, got {chunk_type!r}, skipping")
                    continue

                track_length = struct.unpack('>I', f.read(4))[0]
                track_data = f.read(track_length)

                # Parse track (simplified - just extract note events)
                track = sequence.add_track(f"Track {track_num + 1}")
                sequence._parse_track_data(track, track_data)

        logger.info(f"Loaded MIDI file: {filename}")
        return sequence

    def _parse_track_data(self, track: MIDITrack, data: bytes) -> None:
        """Parse track data and add events to track.

        Args:
            track: Track to add events to
            data: Track data bytes
        """
        pos = 0
        current_ticks = 0
        running_status = 0

        def read_variable_length() -> int:
            nonlocal pos
            value = 0
            while True:
                if pos >= len(data):
                    return value
                byte = data[pos]
                pos += 1
                value = (value << 7) | (byte & 0x7F)
                if not (byte & 0x80):
                    break
            return value

        while pos < len(data):
            # Read delta time
            delta = read_variable_length()
            current_ticks += delta

            if pos >= len(data):
                break

            # Read event
            status_byte = data[pos]

            # Handle running status
            if status_byte & 0x80:
                pos += 1
                running_status = status_byte
            else:
                status_byte = running_status

            # Check for meta events (0xFF) before masking
            if status_byte == 0xFF:
                status = 0xFF
                channel = 0
            else:
                status = status_byte & 0xF0
                channel = status_byte & 0x0F

            # Convert ticks to seconds
            time_seconds = current_ticks / (self.ppq * (self.tempo / 60.0))

            # Parse different message types
            if status == 0xFF:  # Meta event
                if pos >= len(data):
                    break
                meta_type = data[pos]
                pos += 1
                length = read_variable_length()

                if meta_type == 0x51 and length == 3:  # Tempo
                    microseconds = (data[pos] << 16) | (data[pos + 1] << 8) | data[pos + 2]
                    self.tempo = 60_000_000 / microseconds
                elif meta_type == 0x03:  # Track name
                    track.name = data[pos:pos + length].decode('utf-8', errors='ignore')

                pos += length

            elif status in (0x80, 0x90, 0xA0, 0xB0, 0xE0):  # Two-byte messages
                if pos + 1 >= len(data):
                    break
                data1 = data[pos]
                data2 = data[pos + 1]
                pos += 2

                event = MIDIEvent(time_seconds, status, channel, data1, data2)
                track.events.append(event)

            elif status in (0xC0, 0xD0):  # One-byte messages
                if pos >= len(data):
                    break
                data1 = data[pos]
                pos += 1

                event = MIDIEvent(time_seconds, status, channel, data1, 0)
                track.events.append(event)

            else:
                # Unknown status, skip
                pos += 1

    @property
    def duration(self) -> float:
        """Total sequence duration in seconds."""
        if not self.tracks:
            return 0.0
        return max(track.duration for track in self.tracks)

    def __repr__(self) -> str:
        """String representation of sequence."""
        return f"MIDISequence(tempo={self.tempo:.1f}, tracks={len(self.tracks)}, duration={self.duration:.2f}s)"


# ============================================================================
# MIDI Router
# ============================================================================


@dataclass
class Route:
    """MIDI routing configuration."""

    source: str  # Source identifier
    destination: str  # Destination identifier
    channel_map: Dict[int, int] = field(default_factory=dict)
    transform: Optional[str] = None
    filter_func: Optional[Callable[[MIDIEvent], bool]] = None


class MIDIRouter:
    """MIDI routing matrix.

    Routes MIDI events from sources to destinations with optional
    transformations, channel mapping, and filtering.

    Example:
        >>> router = MIDIRouter()
        >>> router.add_transform("transpose", transpose_transform(12))
        >>> router.add_route("keyboard", "synth", transform="transpose")
        >>> router.process_event("keyboard", event)
    """

    def __init__(self):
        """Initialize MIDI router."""
        self.routes: List[Route] = []
        self.transforms: Dict[str, Callable[[MIDIEvent], MIDIEvent]] = {}

    def add_route(
        self,
        source: str,
        destination: str,
        channel_map: Optional[Dict[int, int]] = None,
        transform: Optional[str] = None,
        filter_func: Optional[Callable[[MIDIEvent], bool]] = None,
    ) -> None:
        """Add MIDI route.

        Args:
            source: Source identifier
            destination: Destination identifier
            channel_map: Optional channel remapping {src_ch: dst_ch}
            transform: Optional transform name to apply
            filter_func: Optional filter function (return True to pass event)
        """
        route = Route(
            source=source,
            destination=destination,
            channel_map=channel_map or {},
            transform=transform,
            filter_func=filter_func,
        )
        self.routes.append(route)
        logger.info(f"Added route: {source} -> {destination}")

    def add_transform(
        self,
        name: str,
        func: Callable[[MIDIEvent], MIDIEvent],
    ) -> None:
        """Register MIDI transform function.

        Args:
            name: Transform name
            func: Transform function(MIDIEvent) -> MIDIEvent
        """
        self.transforms[name] = func
        logger.info(f"Registered transform: {name}")

    def remove_route(self, source: str, destination: str) -> bool:
        """Remove route from source to destination.

        Args:
            source: Source identifier
            destination: Destination identifier

        Returns:
            True if route was removed, False if not found
        """
        for i, route in enumerate(self.routes):
            if route.source == source and route.destination == destination:
                self.routes.pop(i)
                logger.info(f"Removed route: {source} -> {destination}")
                return True
        return False

    def process_event(self, source: str, event: MIDIEvent) -> List[Tuple[str, MIDIEvent]]:
        """Process MIDI event through routing matrix.

        Args:
            source: Source identifier
            event: MIDI event to route

        Returns:
            List of (destination, event) tuples
        """
        results = []

        for route in self.routes:
            if route.source != source:
                continue

            # Make a copy for transformation
            routed_event = MIDIEvent(
                time=event.time,
                status=event.status,
                channel=event.channel,
                data1=event.data1,
                data2=event.data2,
            )

            # Apply filter
            if route.filter_func and not route.filter_func(routed_event):
                continue

            # Apply channel mapping
            if routed_event.channel in route.channel_map:
                routed_event.channel = route.channel_map[routed_event.channel]

            # Apply transform
            if route.transform and route.transform in self.transforms:
                routed_event = self.transforms[route.transform](routed_event)

            results.append((route.destination, routed_event))

        return results

    def clear(self) -> None:
        """Clear all routes."""
        self.routes.clear()
        logger.info("Cleared all routes")

    def __repr__(self) -> str:
        """String representation of router."""
        return f"MIDIRouter(routes={len(self.routes)}, transforms={len(self.transforms)})"


# ============================================================================
# Transform Functions
# ============================================================================


def transpose_transform(semitones: int) -> Callable[[MIDIEvent], MIDIEvent]:
    """Create transpose transform.

    Args:
        semitones: Number of semitones to transpose (positive or negative)

    Returns:
        Transform function
    """

    def transform(event: MIDIEvent) -> MIDIEvent:
        if event.is_note_on or event.is_note_off:
            event.data1 = max(0, min(127, event.data1 + semitones))
        return event

    return transform


def velocity_scale_transform(factor: float) -> Callable[[MIDIEvent], MIDIEvent]:
    """Create velocity scaling transform.

    Args:
        factor: Scale factor (e.g., 0.5 = half velocity, 2.0 = double velocity)

    Returns:
        Transform function
    """

    def transform(event: MIDIEvent) -> MIDIEvent:
        if event.is_note_on:
            event.data2 = int(max(1, min(127, event.data2 * factor)))
        return event

    return transform


def velocity_curve_transform(curve: Callable[[int], int]) -> Callable[[MIDIEvent], MIDIEvent]:
    """Create velocity curve transform.

    Args:
        curve: Function mapping input velocity (0-127) to output velocity (0-127)

    Returns:
        Transform function
    """

    def transform(event: MIDIEvent) -> MIDIEvent:
        if event.is_note_on and event.data2 > 0:
            event.data2 = max(1, min(127, curve(event.data2)))
        return event

    return transform


def channel_remap_transform(channel_map: Dict[int, int]) -> Callable[[MIDIEvent], MIDIEvent]:
    """Create channel remapping transform.

    Args:
        channel_map: Dictionary mapping source channels to destination channels

    Returns:
        Transform function
    """

    def transform(event: MIDIEvent) -> MIDIEvent:
        if event.channel in channel_map:
            event.channel = channel_map[event.channel]
        return event

    return transform


def quantize_transform(grid: float) -> Callable[[MIDIEvent], MIDIEvent]:
    """Create time quantization transform.

    Args:
        grid: Quantization grid in seconds (e.g., 0.25 for 16th notes at 120 BPM)

    Returns:
        Transform function
    """

    def transform(event: MIDIEvent) -> MIDIEvent:
        event.time = round(event.time / grid) * grid
        return event

    return transform
