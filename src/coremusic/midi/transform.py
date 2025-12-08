#!/usr/bin/env python3
"""MIDI Transformation Pipeline.

This module provides a composable pipeline for transforming MIDI sequences.
Transformers can be chained together to create complex processing workflows.

Example:
    >>> from coremusic.midi.utilities import MIDISequence
    >>> from coremusic.midi.transform import Pipeline, Transpose, Quantize, Humanize
    >>>
    >>> # Load MIDI file
    >>> seq = MIDISequence.load("input.mid")
    >>>
    >>> # Create transformation pipeline
    >>> pipeline = Pipeline([
    ...     Transpose(semitones=5),
    ...     Quantize(grid=1/16, strength=0.8),
    ...     Humanize(timing=0.01, velocity=5),
    ... ])
    >>>
    >>> # Apply transformations
    >>> transformed = pipeline.apply(seq)
    >>> transformed.save("output.mid")
"""

import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import (TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple,
                    Union)

from .utilities import MIDIEvent, MIDISequence, MIDIStatus

if TYPE_CHECKING:
    from ..music.theory import Scale


# ============================================================================
# Base Classes
# ============================================================================


class MIDITransformer(ABC):
    """Abstract base class for MIDI transformers.

    All transformers must implement the transform() method which takes
    a MIDISequence and returns a transformed copy.
    """

    @abstractmethod
    def transform(self, sequence: MIDISequence) -> MIDISequence:
        """Transform a MIDI sequence.

        Args:
            sequence: Input MIDI sequence

        Returns:
            Transformed MIDI sequence (copy)
        """
        pass

    def __call__(self, sequence: MIDISequence) -> MIDISequence:
        """Allow transformer to be called directly."""
        return self.transform(sequence)

    def _copy_sequence(self, sequence: MIDISequence) -> MIDISequence:
        """Create a deep copy of a sequence."""
        return deepcopy(sequence)

    def _copy_event(self, event: MIDIEvent) -> MIDIEvent:
        """Create a copy of an event."""
        return MIDIEvent(
            time=event.time,
            status=event.status,
            channel=event.channel,
            data1=event.data1,
            data2=event.data2,
        )


class Pipeline:
    """Chain of MIDI transformers applied in sequence.

    Example:
        >>> pipeline = Pipeline([
        ...     Transpose(5),
        ...     Quantize(1/16),
        ...     VelocityScale(0.8, 1.0),
        ... ])
        >>> result = pipeline.apply(sequence)
    """

    def __init__(self, transformers: Optional[List[MIDITransformer]] = None):
        """Initialize pipeline.

        Args:
            transformers: List of transformers to apply in order
        """
        self.transformers: List[MIDITransformer] = transformers or []

    def add(self, transformer: MIDITransformer) -> "Pipeline":
        """Add transformer to pipeline.

        Args:
            transformer: Transformer to add

        Returns:
            Self for chaining
        """
        self.transformers.append(transformer)
        return self

    def apply(self, sequence: MIDISequence) -> MIDISequence:
        """Apply all transformers in sequence.

        Args:
            sequence: Input MIDI sequence

        Returns:
            Transformed MIDI sequence
        """
        result = sequence
        for transformer in self.transformers:
            result = transformer.transform(result)
        return result

    def __call__(self, sequence: MIDISequence) -> MIDISequence:
        """Allow pipeline to be called directly."""
        return self.apply(sequence)

    def __len__(self) -> int:
        """Return number of transformers in pipeline."""
        return len(self.transformers)

    def __repr__(self) -> str:
        names = [t.__class__.__name__ for t in self.transformers]
        return f"Pipeline([{', '.join(names)}])"


# ============================================================================
# Pitch Transformers
# ============================================================================


class Transpose(MIDITransformer):
    """Transpose all notes by a fixed number of semitones.

    Example:
        >>> transposed = Transpose(5).transform(sequence)  # Up a fourth
        >>> transposed = Transpose(-12).transform(sequence)  # Down an octave
    """

    def __init__(self, semitones: int):
        """Initialize transpose transformer.

        Args:
            semitones: Number of semitones to transpose (positive or negative)
        """
        self.semitones = semitones

    def transform(self, sequence: MIDISequence) -> MIDISequence:
        result = self._copy_sequence(sequence)

        for track in result.tracks:
            for event in track.events:
                if event.status in (MIDIStatus.NOTE_ON, MIDIStatus.NOTE_OFF):
                    new_note = event.data1 + self.semitones
                    event.data1 = max(0, min(127, new_note))

        return result

    def __repr__(self) -> str:
        return f"Transpose({self.semitones})"


class Invert(MIDITransformer):
    """Invert melody around a pivot note.

    Intervals are mirrored: notes above the pivot go below and vice versa.

    Example:
        >>> inverted = Invert(60).transform(sequence)  # Invert around middle C
    """

    def __init__(self, pivot: int = 60):
        """Initialize invert transformer.

        Args:
            pivot: MIDI note number to invert around (default: middle C)
        """
        self.pivot = pivot

    def transform(self, sequence: MIDISequence) -> MIDISequence:
        result = self._copy_sequence(sequence)

        for track in result.tracks:
            for event in track.events:
                if event.status in (MIDIStatus.NOTE_ON, MIDIStatus.NOTE_OFF):
                    # Mirror around pivot: new_note = pivot - (note - pivot) = 2*pivot - note
                    new_note = 2 * self.pivot - event.data1
                    event.data1 = max(0, min(127, new_note))

        return result

    def __repr__(self) -> str:
        return f"Invert(pivot={self.pivot})"


class Harmonize(MIDITransformer):
    """Add parallel intervals to create harmonies.

    Example:
        >>> # Add a third and fifth above each note
        >>> harmonized = Harmonize([4, 7]).transform(sequence)
    """

    def __init__(
        self,
        intervals: List[int],
        velocity_scale: float = 0.8,
    ):
        """Initialize harmonize transformer.

        Args:
            intervals: List of intervals in semitones to add (e.g., [4, 7] for major chord)
            velocity_scale: Velocity multiplier for harmony notes (0.0-1.0)
        """
        self.intervals = intervals
        self.velocity_scale = velocity_scale

    def transform(self, sequence: MIDISequence) -> MIDISequence:
        result = self._copy_sequence(sequence)

        for track in result.tracks:
            new_events = []
            for event in track.events:
                if event.status == MIDIStatus.NOTE_ON and event.data2 > 0:
                    # Add harmony notes
                    for interval in self.intervals:
                        new_note = event.data1 + interval
                        if 0 <= new_note <= 127:
                            harmony_velocity = int(event.data2 * self.velocity_scale)
                            harmony_velocity = max(1, min(127, harmony_velocity))
                            new_events.append(MIDIEvent(
                                time=event.time,
                                status=MIDIStatus.NOTE_ON,
                                channel=event.channel,
                                data1=new_note,
                                data2=harmony_velocity,
                            ))
                elif event.status == MIDIStatus.NOTE_OFF or (
                    event.status == MIDIStatus.NOTE_ON and event.data2 == 0
                ):
                    # Add note offs for harmony notes
                    for interval in self.intervals:
                        new_note = event.data1 + interval
                        if 0 <= new_note <= 127:
                            new_events.append(MIDIEvent(
                                time=event.time,
                                status=MIDIStatus.NOTE_OFF,
                                channel=event.channel,
                                data1=new_note,
                                data2=0,
                            ))

            track.events.extend(new_events)
            track.events.sort(key=lambda e: e.time)

        return result

    def __repr__(self) -> str:
        return f"Harmonize({self.intervals})"


# ============================================================================
# Time Transformers
# ============================================================================


class Quantize(MIDITransformer):
    """Quantize note timing to a grid.

    Example:
        >>> # Quantize to 16th notes at 120 BPM (0.125s grid)
        >>> quantized = Quantize(grid=0.125, strength=1.0).transform(sequence)
        >>> # Partial quantize (swing-friendly)
        >>> quantized = Quantize(grid=0.125, strength=0.5).transform(sequence)
    """

    def __init__(
        self,
        grid: float,
        strength: float = 1.0,
        swing: float = 0.0,
    ):
        """Initialize quantize transformer.

        Args:
            grid: Quantization grid in seconds (e.g., 0.125 for 16th notes at 120 BPM)
            strength: Quantization strength 0.0-1.0 (1.0 = full quantize)
            swing: Swing amount 0.0-1.0 (shifts every other grid point)
        """
        if grid <= 0:
            raise ValueError(f"Grid must be > 0, got {grid}")
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"Strength must be 0.0-1.0, got {strength}")
        if not 0.0 <= swing <= 1.0:
            raise ValueError(f"Swing must be 0.0-1.0, got {swing}")

        self.grid = grid
        self.strength = strength
        self.swing = swing

    def _quantize_time(self, time: float) -> float:
        """Quantize a single time value."""
        grid_position = round(time / self.grid)
        quantized = grid_position * self.grid

        # Apply swing to odd grid positions
        if self.swing > 0 and grid_position % 2 == 1:
            quantized += self.grid * self.swing * 0.5

        # Apply strength (blend between original and quantized)
        return time + (quantized - time) * self.strength

    def transform(self, sequence: MIDISequence) -> MIDISequence:
        result = self._copy_sequence(sequence)

        for track in result.tracks:
            for event in track.events:
                event.time = self._quantize_time(event.time)
            track.events.sort(key=lambda e: e.time)

        return result

    def __repr__(self) -> str:
        return f"Quantize(grid={self.grid}, strength={self.strength})"


class TimeStretch(MIDITransformer):
    """Stretch or compress timing by a factor.

    Example:
        >>> stretched = TimeStretch(2.0).transform(sequence)  # Double tempo
        >>> compressed = TimeStretch(0.5).transform(sequence)  # Half tempo
    """

    def __init__(self, factor: float):
        """Initialize time stretch transformer.

        Args:
            factor: Time multiplier (>1 = slower, <1 = faster)
        """
        if factor <= 0:
            raise ValueError(f"Factor must be > 0, got {factor}")
        self.factor = factor

    def transform(self, sequence: MIDISequence) -> MIDISequence:
        result = self._copy_sequence(sequence)

        for track in result.tracks:
            for event in track.events:
                event.time *= self.factor

        return result

    def __repr__(self) -> str:
        return f"TimeStretch({self.factor})"


class TimeShift(MIDITransformer):
    """Shift all events by a fixed time offset.

    Example:
        >>> shifted = TimeShift(1.0).transform(sequence)  # Delay by 1 second
        >>> shifted = TimeShift(-0.5).transform(sequence)  # Earlier by 0.5s
    """

    def __init__(self, offset: float):
        """Initialize time shift transformer.

        Args:
            offset: Time offset in seconds (positive = later, negative = earlier)
        """
        self.offset = offset

    def transform(self, sequence: MIDISequence) -> MIDISequence:
        result = self._copy_sequence(sequence)

        for track in result.tracks:
            for event in track.events:
                event.time = max(0.0, event.time + self.offset)
            track.events.sort(key=lambda e: e.time)

        return result

    def __repr__(self) -> str:
        return f"TimeShift({self.offset})"


class Reverse(MIDITransformer):
    """Reverse the sequence (retrograde).

    Note durations are preserved but note order is reversed.

    Example:
        >>> reversed_seq = Reverse().transform(sequence)
    """

    def transform(self, sequence: MIDISequence) -> MIDISequence:
        result = self._copy_sequence(sequence)

        for track in result.tracks:
            if not track.events:
                continue

            # Find total duration
            max_time = max(e.time for e in track.events)

            # Build note pairs (note on -> note off)
            note_pairs: Dict[Tuple[int, int], List[Tuple[float, float, int]]] = {}
            other_events: List[MIDIEvent] = []

            note_on_times: Dict[Tuple[int, int, int], float] = {}

            for event in track.events:
                key = (event.data1, event.channel)
                if event.is_note_on:
                    note_on_times[(event.data1, event.channel, event.data2)] = event.time
                elif event.is_note_off:
                    # Find matching note on
                    for (note, ch, vel), on_time in list(note_on_times.items()):
                        if note == event.data1 and ch == event.channel:
                            duration = event.time - on_time
                            if key not in note_pairs:
                                note_pairs[key] = []
                            note_pairs[key].append((on_time, duration, vel))
                            del note_on_times[(note, ch, vel)]
                            break
                elif event.status not in (MIDIStatus.NOTE_ON, MIDIStatus.NOTE_OFF):
                    other_events.append(event)

            # Clear and rebuild events
            track.events.clear()

            # Reverse note timings
            for (note, channel), pairs in note_pairs.items():
                for on_time, duration, velocity in pairs:
                    # Calculate reversed start time
                    reversed_start = max_time - on_time - duration

                    track.events.append(MIDIEvent(
                        time=max(0.0, reversed_start),
                        status=MIDIStatus.NOTE_ON,
                        channel=channel,
                        data1=note,
                        data2=velocity,
                    ))
                    track.events.append(MIDIEvent(
                        time=max(0.0, reversed_start + duration),
                        status=MIDIStatus.NOTE_OFF,
                        channel=channel,
                        data1=note,
                        data2=0,
                    ))

            # Reverse other events too
            for event in other_events:
                event.time = max(0.0, max_time - event.time)
                track.events.append(event)

            track.events.sort(key=lambda e: e.time)

        return result

    def __repr__(self) -> str:
        return "Reverse()"


# ============================================================================
# Velocity Transformers
# ============================================================================


class VelocityScale(MIDITransformer):
    """Scale velocities to a range or by a factor.

    Example:
        >>> # Compress velocity range
        >>> scaled = VelocityScale(min_vel=40, max_vel=100).transform(sequence)
        >>> # Scale by factor
        >>> scaled = VelocityScale(factor=0.8).transform(sequence)
    """

    def __init__(
        self,
        min_vel: Optional[int] = None,
        max_vel: Optional[int] = None,
        factor: Optional[float] = None,
    ):
        """Initialize velocity scale transformer.

        Args:
            min_vel: Minimum output velocity (for range scaling)
            max_vel: Maximum output velocity (for range scaling)
            factor: Velocity multiplier (alternative to min/max)
        """
        if factor is not None and (min_vel is not None or max_vel is not None):
            raise ValueError("Use either factor OR min_vel/max_vel, not both")

        self.min_vel = min_vel
        self.max_vel = max_vel
        self.factor = factor

    def transform(self, sequence: MIDISequence) -> MIDISequence:
        result = self._copy_sequence(sequence)

        # Collect all velocities for range scaling
        if self.min_vel is not None or self.max_vel is not None:
            all_velocities = []
            for track in result.tracks:
                for event in track.events:
                    if event.is_note_on:
                        all_velocities.append(event.data2)

            if not all_velocities:
                return result

            orig_min = min(all_velocities)
            orig_max = max(all_velocities)
            orig_range = orig_max - orig_min if orig_max > orig_min else 1

            target_min = self.min_vel if self.min_vel is not None else orig_min
            target_max = self.max_vel if self.max_vel is not None else orig_max
            target_range = target_max - target_min

            for track in result.tracks:
                for event in track.events:
                    if event.is_note_on:
                        # Normalize to 0-1, then scale to target range
                        normalized = (event.data2 - orig_min) / orig_range
                        new_vel = int(target_min + normalized * target_range)
                        event.data2 = max(1, min(127, new_vel))

        elif self.factor is not None:
            for track in result.tracks:
                for event in track.events:
                    if event.is_note_on:
                        new_vel = int(event.data2 * self.factor)
                        event.data2 = max(1, min(127, new_vel))

        return result

    def __repr__(self) -> str:
        if self.factor is not None:
            return f"VelocityScale(factor={self.factor})"
        return f"VelocityScale(min_vel={self.min_vel}, max_vel={self.max_vel})"


class VelocityCurve(MIDITransformer):
    """Apply a velocity curve (compression/expansion).

    Example:
        >>> # Logarithmic curve for softer feel
        >>> curved = VelocityCurve(curve='log').transform(sequence)
        >>> # Exponential for more dynamic range
        >>> curved = VelocityCurve(curve='exp').transform(sequence)
    """

    CURVES = {
        'linear': lambda x: x,
        'log': lambda x: (x ** 0.5),  # Square root for softer feel
        'exp': lambda x: (x ** 2),    # Square for more dynamic
        'soft': lambda x: (x ** 0.7),
        'hard': lambda x: (x ** 1.5),
    }

    def __init__(
        self,
        curve: Union[str, Callable[[float], float]] = 'linear',
    ):
        """Initialize velocity curve transformer.

        Args:
            curve: Curve name ('linear', 'log', 'exp', 'soft', 'hard')
                   or custom function(0-1) -> 0-1
        """
        if isinstance(curve, str):
            if curve not in self.CURVES:
                raise ValueError(f"Unknown curve: {curve}. Available: {list(self.CURVES.keys())}")
            self.curve_func = self.CURVES[curve]
            self.curve_name = curve
        else:
            self.curve_func = curve
            self.curve_name = 'custom'

    def transform(self, sequence: MIDISequence) -> MIDISequence:
        result = self._copy_sequence(sequence)

        for track in result.tracks:
            for event in track.events:
                if event.is_note_on:
                    # Normalize to 0-1, apply curve, scale back
                    normalized = event.data2 / 127.0
                    curved = self.curve_func(normalized)
                    event.data2 = max(1, min(127, int(curved * 127)))

        return result

    def __repr__(self) -> str:
        return f"VelocityCurve(curve={self.curve_name!r})"


# ============================================================================
# Humanize Transformer
# ============================================================================


class Humanize(MIDITransformer):
    """Add human-like timing and velocity variations.

    Example:
        >>> humanized = Humanize(timing=0.02, velocity=10).transform(sequence)
    """

    def __init__(
        self,
        timing: float = 0.01,
        velocity: int = 5,
        seed: Optional[int] = None,
    ):
        """Initialize humanize transformer.

        Args:
            timing: Maximum timing deviation in seconds
            velocity: Maximum velocity deviation (+/-)
            seed: Random seed for reproducibility
        """
        if timing < 0:
            raise ValueError(f"Timing must be >= 0, got {timing}")
        if velocity < 0:
            raise ValueError(f"Velocity must be >= 0, got {velocity}")

        self.timing = timing
        self.velocity = velocity
        self.seed = seed

    def transform(self, sequence: MIDISequence) -> MIDISequence:
        result = self._copy_sequence(sequence)

        rng = random.Random(self.seed)

        # Track note on times to apply same shift to corresponding note offs
        note_shifts: Dict[Tuple[int, int, float], float] = {}

        for track in result.tracks:
            for event in track.events:
                if event.is_note_on:
                    # Apply timing variation
                    time_shift = rng.uniform(-self.timing, self.timing)
                    event.time = max(0.0, event.time + time_shift)

                    # Store shift for matching note off
                    note_shifts[(event.data1, event.channel, event.time - time_shift)] = time_shift

                    # Apply velocity variation
                    vel_shift = rng.randint(-self.velocity, self.velocity)
                    event.data2 = max(1, min(127, event.data2 + vel_shift))

                elif event.is_note_off:
                    # Find and apply same shift as note on
                    for (note, ch, orig_time), shift in list(note_shifts.items()):
                        if note == event.data1 and ch == event.channel:
                            event.time = max(0.0, event.time + shift)
                            del note_shifts[(note, ch, orig_time)]
                            break

            track.events.sort(key=lambda e: e.time)

        return result

    def __repr__(self) -> str:
        return f"Humanize(timing={self.timing}, velocity={self.velocity})"


# ============================================================================
# Filter Transformers
# ============================================================================


class NoteFilter(MIDITransformer):
    """Filter notes by pitch range, velocity, or channel.

    Example:
        >>> # Keep only notes C3-C5
        >>> filtered = NoteFilter(min_note=48, max_note=72).transform(sequence)
        >>> # Keep only loud notes
        >>> filtered = NoteFilter(min_velocity=80).transform(sequence)
    """

    def __init__(
        self,
        min_note: Optional[int] = None,
        max_note: Optional[int] = None,
        min_velocity: Optional[int] = None,
        max_velocity: Optional[int] = None,
        channels: Optional[Set[int]] = None,
        invert: bool = False,
    ):
        """Initialize note filter.

        Args:
            min_note: Minimum note to keep (inclusive)
            max_note: Maximum note to keep (inclusive)
            min_velocity: Minimum velocity to keep (inclusive)
            max_velocity: Maximum velocity to keep (inclusive)
            channels: Set of channels to keep (None = all)
            invert: If True, remove matching notes instead of keeping them
        """
        self.min_note = min_note
        self.max_note = max_note
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.channels = channels
        self.invert = invert

    def _matches(self, event: MIDIEvent) -> bool:
        """Check if event matches filter criteria."""
        if self.min_note is not None and event.data1 < self.min_note:
            return False
        if self.max_note is not None and event.data1 > self.max_note:
            return False
        if event.is_note_on:
            if self.min_velocity is not None and event.data2 < self.min_velocity:
                return False
            if self.max_velocity is not None and event.data2 > self.max_velocity:
                return False
        if self.channels is not None and event.channel not in self.channels:
            return False
        return True

    def transform(self, sequence: MIDISequence) -> MIDISequence:
        result = self._copy_sequence(sequence)

        for track in result.tracks:
            # Track which notes to filter
            notes_to_keep: Set[Tuple[int, int]] = set()
            notes_to_remove: Set[Tuple[int, int]] = set()

            # First pass: determine which notes pass the filter
            for event in track.events:
                if event.status in (MIDIStatus.NOTE_ON, MIDIStatus.NOTE_OFF):
                    key = (event.data1, event.channel)
                    if event.is_note_on:
                        matches = self._matches(event)
                        keep = matches if not self.invert else not matches
                        if keep:
                            notes_to_keep.add(key)
                        else:
                            notes_to_remove.add(key)

            # Second pass: filter events
            filtered_events = []
            for event in track.events:
                if event.status in (MIDIStatus.NOTE_ON, MIDIStatus.NOTE_OFF):
                    key = (event.data1, event.channel)
                    if key in notes_to_keep:
                        filtered_events.append(event)
                else:
                    # Keep non-note events
                    filtered_events.append(event)

            track.events = filtered_events

        return result

    def __repr__(self) -> str:
        parts = []
        if self.min_note is not None:
            parts.append(f"min_note={self.min_note}")
        if self.max_note is not None:
            parts.append(f"max_note={self.max_note}")
        if self.min_velocity is not None:
            parts.append(f"min_velocity={self.min_velocity}")
        if self.max_velocity is not None:
            parts.append(f"max_velocity={self.max_velocity}")
        if self.channels is not None:
            parts.append(f"channels={self.channels}")
        if self.invert:
            parts.append("invert=True")
        return f"NoteFilter({', '.join(parts)})"


class ScaleFilter(MIDITransformer):
    """Filter notes to only allow those in a given scale.

    Notes that are not in the scale are removed from the sequence.
    This acts as a "scale mask" that only lets through notes belonging
    to the specified scale.

    Example:
        >>> from coremusic.music.theory import Note, Scale, ScaleType
        >>> # Only keep notes in C major
        >>> c_major = Scale(Note.from_name("C4"), ScaleType.MAJOR)
        >>> filtered = ScaleFilter(c_major).transform(sequence)
        >>>
        >>> # Only keep notes in A minor pentatonic
        >>> a_pent = Scale(Note.from_name("A3"), ScaleType.MINOR_PENTATONIC)
        >>> filtered = ScaleFilter(a_pent).transform(sequence)
    """

    def __init__(self, scale: "Scale"):
        """Initialize scale filter.

        Args:
            scale: Scale object defining which notes to allow through.
                   Notes are matched by pitch class (octave-independent).
        """
        # Import here to avoid circular imports at module level
        from ..music.theory import Scale as ScaleClass
        if not isinstance(scale, ScaleClass):
            raise TypeError(f"Expected Scale, got {type(scale).__name__}")
        self.scale = scale
        # Pre-compute allowed pitch classes for efficiency
        root_pc = scale.root.pitch_class
        self._allowed_pitch_classes: Set[int] = {
            (root_pc + interval) % 12 for interval in scale.intervals
        }

    def _note_in_scale(self, midi_note: int) -> bool:
        """Check if a MIDI note number is in the scale."""
        pitch_class = midi_note % 12
        return pitch_class in self._allowed_pitch_classes

    def transform(self, sequence: MIDISequence) -> MIDISequence:
        result = self._copy_sequence(sequence)

        for track in result.tracks:
            # Track which notes to keep (those in scale)
            notes_to_keep: Set[Tuple[int, int]] = set()

            # First pass: determine which notes are in scale
            for event in track.events:
                if event.is_note_on:
                    key = (event.data1, event.channel)
                    if self._note_in_scale(event.data1):
                        notes_to_keep.add(key)

            # Second pass: filter events
            filtered_events = []
            for event in track.events:
                if event.status in (MIDIStatus.NOTE_ON, MIDIStatus.NOTE_OFF):
                    key = (event.data1, event.channel)
                    if key in notes_to_keep:
                        filtered_events.append(event)
                else:
                    # Keep non-note events
                    filtered_events.append(event)

            track.events = filtered_events

        return result

    def __repr__(self) -> str:
        return f"ScaleFilter(scale={self.scale})"


class EventTypeFilter(MIDITransformer):
    """Filter events by MIDI message type.

    Example:
        >>> # Keep only note events
        >>> filtered = EventTypeFilter(keep=[MIDIStatus.NOTE_ON, MIDIStatus.NOTE_OFF])
        >>> # Remove control changes
        >>> filtered = EventTypeFilter(remove=[MIDIStatus.CONTROL_CHANGE])
    """

    def __init__(
        self,
        keep: Optional[List[int]] = None,
        remove: Optional[List[int]] = None,
    ):
        """Initialize event type filter.

        Args:
            keep: List of status types to keep (None = keep all)
            remove: List of status types to remove
        """
        if keep is not None and remove is not None:
            raise ValueError("Use either keep OR remove, not both")

        self.keep = set(keep) if keep else None
        self.remove = set(remove) if remove else None

    def transform(self, sequence: MIDISequence) -> MIDISequence:
        result = self._copy_sequence(sequence)

        for track in result.tracks:
            if self.keep is not None:
                track.events = [e for e in track.events if e.status in self.keep]
            elif self.remove is not None:
                track.events = [e for e in track.events if e.status not in self.remove]

        return result

    def __repr__(self) -> str:
        if self.keep is not None:
            return f"EventTypeFilter(keep={self.keep})"
        return f"EventTypeFilter(remove={self.remove})"


# ============================================================================
# Track Transformers
# ============================================================================


class ChannelRemap(MIDITransformer):
    """Remap MIDI channels.

    Example:
        >>> # Move all events from channel 0 to channel 9 (drums)
        >>> remapped = ChannelRemap({0: 9}).transform(sequence)
    """

    def __init__(self, channel_map: Dict[int, int]):
        """Initialize channel remap transformer.

        Args:
            channel_map: Dictionary mapping source channel to destination channel
        """
        for src, dst in channel_map.items():
            if not 0 <= src <= 15:
                raise ValueError(f"Source channel must be 0-15, got {src}")
            if not 0 <= dst <= 15:
                raise ValueError(f"Destination channel must be 0-15, got {dst}")

        self.channel_map = channel_map

    def transform(self, sequence: MIDISequence) -> MIDISequence:
        result = self._copy_sequence(sequence)

        for track in result.tracks:
            for event in track.events:
                if event.channel in self.channel_map:
                    event.channel = self.channel_map[event.channel]

        return result

    def __repr__(self) -> str:
        return f"ChannelRemap({self.channel_map})"


class TrackMerge(MIDITransformer):
    """Merge all tracks into a single track.

    Example:
        >>> merged = TrackMerge(name="Combined").transform(sequence)
    """

    def __init__(self, name: str = "Merged"):
        """Initialize track merge transformer.

        Args:
            name: Name for the merged track
        """
        self.name = name

    def transform(self, sequence: MIDISequence) -> MIDISequence:
        result = MIDISequence(
            tempo=sequence.tempo,
            time_signature=sequence.time_signature,
        )
        result.ppq = sequence.ppq

        merged_track = result.add_track(self.name)

        for track in sequence.tracks:
            for event in track.events:
                merged_track.events.append(self._copy_event(event))

        merged_track.events.sort(key=lambda e: e.time)

        return result

    def __repr__(self) -> str:
        return f"TrackMerge(name={self.name!r})"


# ============================================================================
# Arpeggio Transformer
# ============================================================================


class Arpeggiate(MIDITransformer):
    """Convert chords into arpeggios.

    Example:
        >>> arpeggiated = Arpeggiate(pattern='up', note_duration=0.1).transform(sequence)
    """

    PATTERNS = ['up', 'down', 'up_down', 'down_up', 'random']

    def __init__(
        self,
        pattern: str = 'up',
        note_duration: float = 0.1,
        seed: Optional[int] = None,
    ):
        """Initialize arpeggiate transformer.

        Args:
            pattern: Arpeggio pattern ('up', 'down', 'up_down', 'down_up', 'random')
            note_duration: Duration of each arpeggiated note in seconds
            seed: Random seed for 'random' pattern
        """
        if pattern not in self.PATTERNS:
            raise ValueError(f"Unknown pattern: {pattern}. Available: {self.PATTERNS}")

        self.pattern = pattern
        self.note_duration = note_duration
        self.seed = seed

    def _get_pattern_order(self, notes: list, rng: random.Random) -> List[int]:
        """Get indices for pattern."""
        n = len(notes)
        if self.pattern == 'up':
            return list(range(n))
        elif self.pattern == 'down':
            return list(range(n - 1, -1, -1))
        elif self.pattern == 'up_down':
            return list(range(n)) + list(range(n - 2, 0, -1))
        elif self.pattern == 'down_up':
            return list(range(n - 1, -1, -1)) + list(range(1, n - 1))
        elif self.pattern == 'random':
            indices = list(range(n))
            rng.shuffle(indices)
            return indices
        return list(range(n))

    def transform(self, sequence: MIDISequence) -> MIDISequence:
        result = self._copy_sequence(sequence)
        rng = random.Random(self.seed)

        for track in result.tracks:
            # Group simultaneous note-ons as chords
            chord_times: Dict[float, List[MIDIEvent]] = {}
            non_note_events: List[MIDIEvent] = []
            note_off_times: Dict[Tuple[int, int], float] = {}

            for event in track.events:
                if event.is_note_on:
                    if event.time not in chord_times:
                        chord_times[event.time] = []
                    chord_times[event.time].append(event)
                elif event.is_note_off:
                    note_off_times[(event.data1, event.channel)] = event.time
                else:
                    non_note_events.append(event)

            # Rebuild events with arpeggios
            track.events = non_note_events.copy()

            for chord_time, chord_events in sorted(chord_times.items()):
                if len(chord_events) <= 1:
                    # Single note, keep as is
                    for event in chord_events:
                        track.events.append(event)
                        off_time = note_off_times.get((event.data1, event.channel), chord_time + 0.5)
                        track.events.append(MIDIEvent(
                            time=off_time,
                            status=MIDIStatus.NOTE_OFF,
                            channel=event.channel,
                            data1=event.data1,
                            data2=0,
                        ))
                else:
                    # Arpeggiate the chord
                    sorted_notes = sorted(chord_events, key=lambda e: e.data1)
                    pattern_indices = self._get_pattern_order(sorted_notes, rng)

                    for i, idx in enumerate(pattern_indices):
                        event = sorted_notes[idx]
                        start_time = chord_time + i * self.note_duration

                        track.events.append(MIDIEvent(
                            time=start_time,
                            status=MIDIStatus.NOTE_ON,
                            channel=event.channel,
                            data1=event.data1,
                            data2=event.data2,
                        ))
                        track.events.append(MIDIEvent(
                            time=start_time + self.note_duration * 0.9,
                            status=MIDIStatus.NOTE_OFF,
                            channel=event.channel,
                            data1=event.data1,
                            data2=0,
                        ))

            track.events.sort(key=lambda e: e.time)

        return result

    def __repr__(self) -> str:
        return f"Arpeggiate(pattern={self.pattern!r}, note_duration={self.note_duration})"


# ============================================================================
# Convenience Functions
# ============================================================================


def transpose(sequence: MIDISequence, semitones: int) -> MIDISequence:
    """Convenience function to transpose a sequence."""
    return Transpose(semitones).transform(sequence)


def quantize(
    sequence: MIDISequence,
    grid: float,
    strength: float = 1.0,
) -> MIDISequence:
    """Convenience function to quantize a sequence."""
    return Quantize(grid, strength).transform(sequence)


def humanize(
    sequence: MIDISequence,
    timing: float = 0.01,
    velocity: int = 5,
) -> MIDISequence:
    """Convenience function to humanize a sequence."""
    return Humanize(timing, velocity).transform(sequence)


def reverse(sequence: MIDISequence) -> MIDISequence:
    """Convenience function to reverse a sequence."""
    return Reverse().transform(sequence)


def scale_velocity(
    sequence: MIDISequence,
    factor: Optional[float] = None,
    min_vel: Optional[int] = None,
    max_vel: Optional[int] = None,
) -> MIDISequence:
    """Convenience function to scale velocity."""
    return VelocityScale(min_vel, max_vel, factor).transform(sequence)


def filter_to_scale(sequence: MIDISequence, scale: "Scale") -> MIDISequence:
    """Convenience function to filter notes to a scale.

    Args:
        sequence: Input MIDI sequence
        scale: Scale object defining which notes to allow

    Returns:
        Sequence with only notes from the scale
    """
    return ScaleFilter(scale).transform(sequence)
