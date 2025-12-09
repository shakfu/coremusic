#!/usr/bin/env python3
"""MIDI data encoding and preprocessing for neural network training.

This module provides:
- BaseEncoder: Abstract base class for MIDI event encoding
- NoteEncoder: Simple note-only encoding (MIDI notes 0-127)
- MIDIDataset: Dataset class for loading and preprocessing MIDI files

Example:
    >>> from coremusic.music.neural.data import NoteEncoder, MIDIDataset
    >>>
    >>> encoder = NoteEncoder()
    >>> dataset = MIDIDataset(encoder, seq_length=32)
    >>> dataset.load_file('music.mid')
    >>> dataset.augment(transpose_range=(-5, 7))
    >>>
    >>> x_train, y_train = dataset.prepare_training_data()
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from coremusic.midi.utilities import MIDIEvent, MIDISequence, MIDIStatus
from coremusic.music.theory import Note, Scale, ScaleType

if TYPE_CHECKING:
    from coremusic.kann import Array2D

# ============================================================================
# Base Encoder
# ============================================================================


class BaseEncoder(ABC):
    """Abstract base class for MIDI event encoding.

    Encoders convert between MIDI events and integer token sequences
    suitable for neural network training.
    """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Size of the vocabulary (number of unique tokens)."""
        pass

    @abstractmethod
    def encode(self, events: List[MIDIEvent]) -> List[int]:
        """Encode MIDI events to a sequence of integer tokens.

        Args:
            events: List of MIDIEvent objects

        Returns:
            List of integer token indices
        """
        pass

    @abstractmethod
    def decode(
        self,
        tokens: List[int],
        tempo: float = 120.0,
        channel: int = 0,
        **kwargs,
    ) -> List[MIDIEvent]:
        """Decode integer tokens back to MIDI events.

        Args:
            tokens: List of integer token indices
            tempo: Tempo for timing (BPM), used by some encoders
            channel: MIDI channel (0-15), used by some encoders
            **kwargs: Additional encoder-specific options

        Returns:
            List of MIDIEvent objects
        """
        pass

    def encode_file(
        self,
        path: Union[str, Path],
        channels: Optional[List[int]] = None,
        tracks: Optional[List[int]] = None,
        track_names: Optional[List[str]] = None,
        pitch_range: Optional[Tuple[int, int]] = None,
        exclude_drums: bool = False,
    ) -> List[int]:
        """Encode a MIDI file to tokens with optional filtering.

        Args:
            path: Path to MIDI file
            channels: Only include events from these MIDI channels (0-15)
            tracks: Only include events from these track indices
            track_names: Only include tracks whose names contain these substrings
            pitch_range: Only include notes within (min_pitch, max_pitch) range
            exclude_drums: If True, exclude channel 9 (GM drums)

        Returns:
            List of integer tokens
        """
        sequence = MIDISequence.load(str(path))
        all_events = []

        for track_idx, track in enumerate(sequence.tracks):
            # Filter by track index
            if tracks is not None and track_idx not in tracks:
                continue

            # Filter by track name
            if track_names is not None:
                track_name = getattr(track, 'name', '') or ''
                if not any(name.lower() in track_name.lower() for name in track_names):
                    continue

            for event in track.events:
                # Filter by channel
                if channels is not None and event.channel not in channels:
                    continue

                # Exclude drums (channel 9 in 0-indexed, channel 10 in 1-indexed)
                if exclude_drums and event.channel == 9:
                    continue

                # Filter by pitch range (only for note events)
                if pitch_range is not None and (event.is_note_on or event.is_note_off):
                    min_pitch, max_pitch = pitch_range
                    if not (min_pitch <= event.data1 <= max_pitch):
                        continue

                all_events.append(event)

        # Sort by time
        all_events.sort(key=lambda e: e.time)
        return self.encode(all_events)

    def _decode_sequential_notes(
        self,
        tokens: List[int],
        step_duration: float,
        velocity: int,
        channel: int = 0,
    ) -> List[MIDIEvent]:
        """Decode tokens as sequential notes with uniform timing.

        This is a common decode pattern used by simple encoders that treat
        each token as a note played at sequential time steps.

        Args:
            tokens: List of note numbers (0-127)
            step_duration: Duration of each step in seconds
            velocity: Note velocity (1-127)
            channel: MIDI channel (0-15)

        Returns:
            List of MIDIEvent objects with note-on and note-off pairs
        """
        events = []

        for i, note in enumerate(tokens):
            if not 0 <= note <= 127:
                continue

            time = i * step_duration

            # Note On
            events.append(
                MIDIEvent(
                    time=time,
                    status=MIDIStatus.NOTE_ON,
                    channel=channel,
                    data1=note,
                    data2=velocity,
                )
            )

            # Note Off
            events.append(
                MIDIEvent(
                    time=time + step_duration,
                    status=MIDIStatus.NOTE_OFF,
                    channel=channel,
                    data1=note,
                    data2=0,
                )
            )

        # Sort by time
        events.sort(key=lambda e: e.time)
        return events


# ============================================================================
# Note Encoder (Simplest)
# ============================================================================


class NoteEncoder(BaseEncoder):
    """Simple encoder that extracts note-on events as MIDI note numbers (0-127).

    This is the simplest encoding scheme - it captures only the pitch
    of notes, discarding timing, velocity, and duration information.

    Attributes:
        vocab_size: 128 (MIDI note range)

    Example:
        >>> encoder = NoteEncoder()
        >>> tokens = encoder.encode(midi_events)
        >>> events = encoder.decode(tokens)
    """

    def __init__(self, default_velocity: int = 100, default_duration: float = 0.25):
        """Initialize the encoder.

        Args:
            default_velocity: Velocity to use when decoding (1-127)
            default_duration: Duration in beats for decoded notes
        """
        self._default_velocity = default_velocity
        self._default_duration = default_duration

    @property
    def vocab_size(self) -> int:
        """Vocabulary size is 128 (MIDI note range 0-127)."""
        return 128

    def encode(self, events: List[MIDIEvent]) -> List[int]:
        """Extract note-on events as MIDI note numbers.

        Args:
            events: List of MIDI events

        Returns:
            List of note numbers (0-127)
        """
        notes = []
        for event in events:
            if event.is_note_on:
                notes.append(event.data1)  # data1 is note number
        return notes

    def decode(
        self,
        tokens: List[int],
        tempo: float = 120.0,
        channel: int = 0,
        **kwargs,
    ) -> List[MIDIEvent]:
        """Convert note tokens back to MIDI events.

        Args:
            tokens: List of note numbers (0-127)
            tempo: Tempo for timing (BPM)
            channel: MIDI channel (0-15)
            **kwargs: Unused, for compatibility with base class

        Returns:
            List of MIDIEvent objects with note-on and note-off pairs
        """
        beat_duration = 60.0 / tempo  # Duration of one beat in seconds
        step_duration = self._default_duration * beat_duration
        return self._decode_sequential_notes(
            tokens, step_duration, self._default_velocity, channel
        )

    def transpose(self, tokens: List[int], semitones: int) -> List[int]:
        """Transpose a sequence of note tokens.

        Args:
            tokens: List of note numbers
            semitones: Number of semitones to transpose (can be negative)

        Returns:
            Transposed list of note numbers, clipped to 0-127
        """
        return [max(0, min(127, note + semitones)) for note in tokens]


# ============================================================================
# Event Encoder (Most Expressive)
# ============================================================================


class EventEncoder(BaseEncoder):
    """Event-based encoder that preserves timing, velocity, and note-off events.

    This encoder tokenizes MIDI events into a sequence that captures:
    - NOTE_ON events (128 tokens: notes 0-127)
    - NOTE_OFF events (128 tokens: notes 0-127)
    - TIME_SHIFT events (100 tokens: time steps)
    - VELOCITY events (32 tokens: velocity buckets)

    Token layout:
    - 0-127: NOTE_ON_0 to NOTE_ON_127
    - 128-255: NOTE_OFF_0 to NOTE_OFF_127
    - 256-355: TIME_SHIFT_1 to TIME_SHIFT_100 (in 10ms increments)
    - 356-387: VELOCITY_1 to VELOCITY_32 (velocity buckets)

    Attributes:
        vocab_size: 388 total tokens
        time_step_ms: Time resolution in milliseconds (default: 10)
        max_time_steps: Maximum time shift tokens (default: 100)
        velocity_bins: Number of velocity buckets (default: 32)

    Example:
        >>> encoder = EventEncoder()
        >>> tokens = encoder.encode(midi_events)
        >>> events = encoder.decode(tokens)
    """

    # Token type offsets
    NOTE_ON_OFFSET = 0
    NOTE_OFF_OFFSET = 128
    TIME_SHIFT_OFFSET = 256
    VELOCITY_OFFSET = 356

    def __init__(
        self,
        time_step_ms: int = 10,
        max_time_steps: int = 100,
        velocity_bins: int = 32,
    ):
        """Initialize the encoder.

        Args:
            time_step_ms: Time resolution in milliseconds
            max_time_steps: Maximum time shift value (longer gaps use multiple tokens)
            velocity_bins: Number of velocity buckets (1-127 mapped to bins)
        """
        self.time_step_ms = time_step_ms
        self.max_time_steps = max_time_steps
        self.velocity_bins = velocity_bins
        self._current_velocity_bin = velocity_bins // 2  # Default mid velocity

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        return 128 + 128 + self.max_time_steps + self.velocity_bins

    def _velocity_to_bin(self, velocity: int) -> int:
        """Convert MIDI velocity (1-127) to bin index (0 to velocity_bins-1)."""
        if velocity <= 0:
            return 0
        return min(self.velocity_bins - 1, (velocity - 1) * self.velocity_bins // 127)

    def _bin_to_velocity(self, bin_idx: int) -> int:
        """Convert bin index back to MIDI velocity."""
        if bin_idx <= 0:
            return 1
        return min(127, 1 + (bin_idx * 127) // self.velocity_bins)

    def _time_to_steps(self, time_seconds: float) -> int:
        """Convert time in seconds to time steps."""
        return int(time_seconds * 1000 / self.time_step_ms)

    def _steps_to_time(self, steps: int) -> float:
        """Convert time steps to seconds."""
        return steps * self.time_step_ms / 1000.0

    def encode(self, events: List[MIDIEvent]) -> List[int]:
        """Encode MIDI events to token sequence.

        Args:
            events: List of MIDI events (should be sorted by time)

        Returns:
            List of token indices
        """
        if not events:
            return []

        # Sort by time
        sorted_events = sorted(events, key=lambda e: e.time)

        tokens = []
        current_time = 0.0
        current_velocity_bin = -1  # Force first velocity token

        for event in sorted_events:
            # Skip non-note events
            if not (event.is_note_on or event.is_note_off):
                continue

            # Add time shift tokens if needed
            time_diff = event.time - current_time
            if time_diff > 0:
                steps = self._time_to_steps(time_diff)
                while steps > 0:
                    shift = min(steps, self.max_time_steps)
                    tokens.append(self.TIME_SHIFT_OFFSET + shift - 1)
                    steps -= shift
                current_time = event.time

            if event.is_note_on:
                # Add velocity token if velocity changed
                vel_bin = self._velocity_to_bin(event.data2)
                if vel_bin != current_velocity_bin:
                    tokens.append(self.VELOCITY_OFFSET + vel_bin)
                    current_velocity_bin = vel_bin

                # Add note-on token
                tokens.append(self.NOTE_ON_OFFSET + event.data1)

            elif event.is_note_off:
                # Add note-off token
                tokens.append(self.NOTE_OFF_OFFSET + event.data1)

        return tokens

    def decode(
        self,
        tokens: List[int],
        tempo: float = 120.0,
        channel: int = 0,
        **kwargs,
    ) -> List[MIDIEvent]:
        """Decode tokens back to MIDI events.

        Args:
            tokens: List of token indices
            tempo: Tempo (not used directly, time is absolute)
            channel: MIDI channel (0-15)
            **kwargs: Unused, for compatibility with base class

        Returns:
            List of MIDIEvent objects
        """
        events = []
        current_time = 0.0
        current_velocity = 100

        for token in tokens:
            if token < 0 or token >= self.vocab_size:
                continue

            if token < self.NOTE_OFF_OFFSET:
                # NOTE_ON token
                note = token - self.NOTE_ON_OFFSET
                events.append(
                    MIDIEvent(
                        time=current_time,
                        status=MIDIStatus.NOTE_ON,
                        channel=channel,
                        data1=note,
                        data2=current_velocity,
                    )
                )

            elif token < self.TIME_SHIFT_OFFSET:
                # NOTE_OFF token
                note = token - self.NOTE_OFF_OFFSET
                events.append(
                    MIDIEvent(
                        time=current_time,
                        status=MIDIStatus.NOTE_OFF,
                        channel=channel,
                        data1=note,
                        data2=0,
                    )
                )

            elif token < self.VELOCITY_OFFSET:
                # TIME_SHIFT token
                steps = token - self.TIME_SHIFT_OFFSET + 1
                current_time += self._steps_to_time(steps)

            else:
                # VELOCITY token
                vel_bin = token - self.VELOCITY_OFFSET
                current_velocity = self._bin_to_velocity(vel_bin)

        # Sort by time
        events.sort(key=lambda e: e.time)
        return events

    def transpose(self, tokens: List[int], semitones: int) -> List[int]:
        """Transpose note tokens in the sequence.

        Args:
            tokens: List of tokens
            semitones: Number of semitones to transpose

        Returns:
            Transposed token list
        """
        result = []
        for token in tokens:
            if token < self.NOTE_OFF_OFFSET:
                # NOTE_ON - transpose and clip
                note = token - self.NOTE_ON_OFFSET
                new_note = max(0, min(127, note + semitones))
                result.append(self.NOTE_ON_OFFSET + new_note)
            elif token < self.TIME_SHIFT_OFFSET:
                # NOTE_OFF - transpose and clip
                note = token - self.NOTE_OFF_OFFSET
                new_note = max(0, min(127, note + semitones))
                result.append(self.NOTE_OFF_OFFSET + new_note)
            else:
                # TIME_SHIFT or VELOCITY - keep unchanged
                result.append(token)
        return result

    def token_type(self, token: int) -> str:
        """Get the type of a token.

        Args:
            token: Token index

        Returns:
            Token type string: 'note_on', 'note_off', 'time_shift', or 'velocity'
        """
        if token < self.NOTE_OFF_OFFSET:
            return "note_on"
        elif token < self.TIME_SHIFT_OFFSET:
            return "note_off"
        elif token < self.VELOCITY_OFFSET:
            return "time_shift"
        else:
            return "velocity"

    def token_value(self, token: int) -> int:
        """Get the value encoded in a token.

        Args:
            token: Token index

        Returns:
            The note number, time steps, or velocity bin
        """
        if token < self.NOTE_OFF_OFFSET:
            return token - self.NOTE_ON_OFFSET
        elif token < self.TIME_SHIFT_OFFSET:
            return token - self.NOTE_OFF_OFFSET
        elif token < self.VELOCITY_OFFSET:
            return token - self.TIME_SHIFT_OFFSET + 1
        else:
            return token - self.VELOCITY_OFFSET


# ============================================================================
# Piano Roll Encoder
# ============================================================================


class PianoRollEncoder(BaseEncoder):
    """Encode MIDI as piano roll (time steps x 128 notes).

    This encoder quantizes MIDI events into a binary piano roll representation
    where each time step is a 128-dimensional vector indicating which notes
    are active.

    Attributes:
        resolution: Time steps per beat (default: 16, i.e., 16th notes)
        vocab_size: 128 (one token per MIDI note for output prediction)

    Note:
        For training, use prepare_piano_roll_data() which returns the full
        piano roll matrix. The encode/decode methods work with flattened
        sequences for compatibility with the base interface.

    Example:
        >>> encoder = PianoRollEncoder(resolution=16)
        >>> piano_roll = encoder.encode_to_piano_roll(midi_events, num_beats=8)
        >>> events = encoder.decode_piano_roll(piano_roll)
    """

    def __init__(self, resolution: int = 16, default_velocity: int = 100):
        """Initialize the encoder.

        Args:
            resolution: Time steps per beat (e.g., 16 = 16th notes)
            default_velocity: Velocity to use when decoding (1-127)
        """
        self.resolution = resolution
        self._default_velocity = default_velocity

    @property
    def vocab_size(self) -> int:
        """Vocabulary size is 128 (MIDI note range)."""
        return 128

    def encode(self, events: List[MIDIEvent]) -> List[int]:
        """Encode MIDI events to a sequence of active notes per time step.

        This flattens the piano roll into a sequence of note events.
        For full piano roll access, use encode_to_piano_roll().

        Args:
            events: List of MIDI events

        Returns:
            List of note numbers representing active notes at each time step
        """
        if not events:
            return []

        # Find duration
        max_time = max(e.time for e in events) if events else 0
        num_steps = max(1, int(max_time * self.resolution) + 1)

        # Build piano roll
        piano_roll = self._build_piano_roll(events, num_steps)

        # Flatten to sequence of active notes (for each step, emit active notes)
        tokens = []
        for step in range(num_steps):
            for note in range(128):
                if piano_roll[step][note]:
                    tokens.append(note)

        return tokens

    def decode(
        self,
        tokens: List[int],
        tempo: float = 120.0,
        channel: int = 0,
        **kwargs,
    ) -> List[MIDIEvent]:
        """Decode note tokens back to MIDI events.

        This is a simplified decode that treats each token as a note at
        sequential time steps. For proper piano roll decoding, use
        decode_piano_roll().

        Args:
            tokens: List of note numbers
            tempo: Tempo in BPM
            channel: MIDI channel (0-15)
            **kwargs: Unused, for compatibility with base class

        Returns:
            List of MIDIEvent objects
        """
        beat_duration = 60.0 / tempo
        step_duration = beat_duration / self.resolution
        return self._decode_sequential_notes(
            tokens, step_duration, self._default_velocity, channel
        )

    def encode_to_piano_roll(
        self, events: List[MIDIEvent], num_beats: Optional[int] = None
    ) -> List[List[int]]:
        """Encode MIDI events to a full piano roll matrix.

        Args:
            events: List of MIDI events
            num_beats: Number of beats (auto-calculated if None)

        Returns:
            2D list [time_steps][128] with 1 for active notes, 0 otherwise
        """
        if not events:
            return []

        # Determine duration
        if num_beats is None:
            max_time = max(e.time for e in events)
            num_beats = int(max_time) + 1

        num_steps = num_beats * self.resolution
        return self._build_piano_roll(events, num_steps)

    def _build_piano_roll(
        self, events: List[MIDIEvent], num_steps: int
    ) -> List[List[int]]:
        """Build piano roll from events.

        Args:
            events: List of MIDI events
            num_steps: Number of time steps

        Returns:
            2D list [time_steps][128]
        """
        # Initialize piano roll
        piano_roll = [[0] * 128 for _ in range(num_steps)]

        # Track active notes and their start times
        active_notes = {}  # note -> start_step

        sorted_events = sorted(events, key=lambda e: e.time)

        for event in sorted_events:
            step = int(event.time * self.resolution)
            step = min(step, num_steps - 1)

            if event.is_note_on:
                active_notes[event.data1] = step
                piano_roll[step][event.data1] = 1
            elif event.is_note_off:
                if event.data1 in active_notes:
                    start_step = active_notes.pop(event.data1)
                    # Fill in all steps while note was held
                    for s in range(start_step, step + 1):
                        if s < num_steps:
                            piano_roll[s][event.data1] = 1

        return piano_roll

    def decode_piano_roll(
        self,
        piano_roll: List[List[int]],
        tempo: float = 120.0,
        channel: int = 0,
    ) -> List[MIDIEvent]:
        """Decode a piano roll matrix back to MIDI events.

        Args:
            piano_roll: 2D list [time_steps][128]
            tempo: Tempo in BPM
            channel: MIDI channel (0-15)

        Returns:
            List of MIDIEvent objects
        """
        events = []
        beat_duration = 60.0 / tempo
        step_duration = beat_duration / self.resolution

        num_steps = len(piano_roll)
        if num_steps == 0:
            return []

        # Track note states
        active_notes = set()

        for step in range(num_steps):
            time = step * step_duration

            for note in range(128):
                is_active = piano_roll[step][note] > 0

                if is_active and note not in active_notes:
                    # Note on
                    events.append(
                        MIDIEvent(
                            time=time,
                            status=MIDIStatus.NOTE_ON,
                            channel=channel,
                            data1=note,
                            data2=self._default_velocity,
                        )
                    )
                    active_notes.add(note)

                elif not is_active and note in active_notes:
                    # Note off
                    events.append(
                        MIDIEvent(
                            time=time,
                            status=MIDIStatus.NOTE_OFF,
                            channel=channel,
                            data1=note,
                            data2=0,
                        )
                    )
                    active_notes.discard(note)

        # Close any remaining notes
        final_time = num_steps * step_duration
        for note in active_notes:
            events.append(
                MIDIEvent(
                    time=final_time,
                    status=MIDIStatus.NOTE_OFF,
                    channel=channel,
                    data1=note,
                    data2=0,
                )
            )

        events.sort(key=lambda e: (e.time, e.status != MIDIStatus.NOTE_OFF))
        return events

    def transpose(self, tokens: List[int], semitones: int) -> List[int]:
        """Transpose a sequence of note tokens.

        Args:
            tokens: List of note numbers
            semitones: Number of semitones to transpose

        Returns:
            Transposed list of note numbers, clipped to 0-127
        """
        return [max(0, min(127, note + semitones)) for note in tokens]

    def transpose_piano_roll(
        self, piano_roll: List[List[int]], semitones: int
    ) -> List[List[int]]:
        """Transpose a piano roll by shifting notes.

        Args:
            piano_roll: 2D list [time_steps][128]
            semitones: Number of semitones to transpose

        Returns:
            Transposed piano roll
        """
        num_steps = len(piano_roll)
        result = [[0] * 128 for _ in range(num_steps)]

        for step in range(num_steps):
            for note in range(128):
                if piano_roll[step][note]:
                    new_note = note + semitones
                    if 0 <= new_note < 128:
                        result[step][new_note] = piano_roll[step][note]

        return result


# ============================================================================
# Relative Pitch Encoder
# ============================================================================


class RelativePitchEncoder(BaseEncoder):
    """Encode MIDI as relative pitch intervals from the previous note.

    This encoder is transposition-invariant since it captures intervals
    rather than absolute pitches. Special tokens handle rests and holds.

    Token layout:
    - 0-48: Intervals from -24 to +24 semitones (49 tokens)
    - 49: REST token (silence/gap)
    - 50: HOLD token (sustain previous note)
    - 51: START token (beginning of sequence)

    Attributes:
        vocab_size: 52 (49 intervals + 3 special tokens)
        max_interval: Maximum interval to encode (default: 24 semitones = 2 octaves)

    Example:
        >>> encoder = RelativePitchEncoder()
        >>> tokens = encoder.encode(midi_events)
        >>> events = encoder.decode(tokens, start_note=60)
    """

    # Special tokens
    REST_TOKEN = 49
    HOLD_TOKEN = 50
    START_TOKEN = 51

    def __init__(
        self,
        max_interval: int = 24,
        default_velocity: int = 100,
        default_duration: float = 0.25,
        quantize_time: bool = True,
        time_resolution: int = 16,
    ):
        """Initialize the encoder.

        Args:
            max_interval: Maximum interval in semitones (clipped beyond this)
            default_velocity: Velocity to use when decoding
            default_duration: Default note duration in beats
            quantize_time: Whether to quantize timing during encoding
            time_resolution: Time steps per beat for quantization
        """
        self.max_interval = max_interval
        self._default_velocity = default_velocity
        self._default_duration = default_duration
        self.quantize_time = quantize_time
        self.time_resolution = time_resolution

    @property
    def vocab_size(self) -> int:
        """Vocabulary size: intervals + special tokens."""
        return self.max_interval * 2 + 1 + 3  # -24 to +24 + REST + HOLD + START

    def _interval_to_token(self, interval: int) -> int:
        """Convert interval to token index.

        Args:
            interval: Interval in semitones

        Returns:
            Token index (0-48 for intervals -24 to +24)
        """
        # Clamp interval to range
        interval = max(-self.max_interval, min(self.max_interval, interval))
        return interval + self.max_interval

    def _token_to_interval(self, token: int) -> int:
        """Convert token index to interval.

        Args:
            token: Token index (0-48)

        Returns:
            Interval in semitones (-24 to +24)
        """
        return token - self.max_interval

    def encode(self, events: List[MIDIEvent]) -> List[int]:
        """Encode MIDI events as relative pitch intervals.

        Args:
            events: List of MIDI events

        Returns:
            List of interval tokens
        """
        if not events:
            return []

        # Extract note-on events
        note_events = [e for e in events if e.is_note_on]
        if not note_events:
            return []

        # Sort by time
        note_events.sort(key=lambda e: e.time)

        tokens = [self.START_TOKEN]
        prev_note = None
        prev_time = 0.0

        for event in note_events:
            # Check for significant time gap (rest)
            if self.quantize_time:
                step_duration = 1.0 / self.time_resolution
                time_diff = event.time - prev_time
                if time_diff > step_duration * 2 and prev_note is not None:
                    # Insert rest token(s) for gaps
                    num_rests = int(time_diff / step_duration) - 1
                    for _ in range(min(num_rests, 4)):  # Cap at 4 rests
                        tokens.append(self.REST_TOKEN)

            if prev_note is None:
                # First note - use interval from middle C (60)
                interval = event.data1 - 60
            else:
                interval = event.data1 - prev_note

            tokens.append(self._interval_to_token(interval))
            prev_note = event.data1
            prev_time = event.time

        return tokens

    def decode(
        self,
        tokens: List[int],
        tempo: float = 120.0,
        channel: int = 0,
        **kwargs,
    ) -> List[MIDIEvent]:
        """Decode interval tokens back to MIDI events.

        Args:
            tokens: List of interval tokens
            tempo: Tempo in BPM
            channel: MIDI channel (0-15)
            **kwargs: Additional options:
                - start_note: Starting note for reconstruction (default: 60)

        Returns:
            List of MIDIEvent objects
        """
        start_note = kwargs.get("start_note", 60)
        events = []
        beat_duration = 60.0 / tempo
        note_duration = self._default_duration * beat_duration

        current_note = start_note
        current_time = 0.0

        for token in tokens:
            if token == self.START_TOKEN:
                continue
            elif token == self.REST_TOKEN:
                # Add a rest (advance time)
                current_time += note_duration
            elif token == self.HOLD_TOKEN:
                # Extend previous note (just advance time, no new note)
                current_time += note_duration
            elif 0 <= token < self.max_interval * 2 + 1:
                # Interval token
                interval = self._token_to_interval(token)
                current_note = current_note + interval

                # Clamp to valid MIDI range
                current_note = max(0, min(127, current_note))

                # Note on
                events.append(
                    MIDIEvent(
                        time=current_time,
                        status=MIDIStatus.NOTE_ON,
                        channel=channel,
                        data1=current_note,
                        data2=self._default_velocity,
                    )
                )

                # Note off
                events.append(
                    MIDIEvent(
                        time=current_time + note_duration,
                        status=MIDIStatus.NOTE_OFF,
                        channel=channel,
                        data1=current_note,
                        data2=0,
                    )
                )

                current_time += note_duration

        events.sort(key=lambda e: e.time)
        return events

    def transpose(self, tokens: List[int], semitones: int) -> List[int]:
        """Transpose is a no-op for relative pitch encoding.

        Since the encoding is already relative/interval-based, transposition
        doesn't change the token sequence. The START_TOKEN's reference note
        would be adjusted during decoding instead.

        Args:
            tokens: List of tokens
            semitones: Ignored for relative encoding

        Returns:
            Same tokens unchanged
        """
        # Relative encoding is transposition invariant
        return tokens.copy()

    def decode_with_start(
        self,
        tokens: List[int],
        start_note: int,
        tempo: float = 120.0,
        channel: int = 0,
    ) -> List[MIDIEvent]:
        """Decode with a specific starting note.

        This is the same as decode() but makes the start_note parameter
        more explicit for transposition purposes.

        Args:
            tokens: List of interval tokens
            start_note: Starting note for reconstruction
            tempo: Tempo in BPM
            channel: MIDI channel (0-15)

        Returns:
            List of MIDIEvent objects
        """
        return self.decode(tokens, start_note=start_note, tempo=tempo, channel=channel)

    def tokens_to_notes(
        self, tokens: List[int], start_note: int = 60
    ) -> List[Optional[int]]:
        """Convert tokens to absolute note numbers.

        Args:
            tokens: List of interval tokens
            start_note: Starting note

        Returns:
            List of note numbers (None for REST/START tokens)
        """
        notes: List[Optional[int]] = []
        current_note = start_note

        for token in tokens:
            if token == self.START_TOKEN:
                notes.append(None)
            elif token == self.REST_TOKEN:
                notes.append(None)
            elif token == self.HOLD_TOKEN:
                notes.append(current_note if notes else None)
            elif 0 <= token < self.max_interval * 2 + 1:
                interval = self._token_to_interval(token)
                current_note = max(0, min(127, current_note + interval))
                notes.append(current_note)
            else:
                notes.append(None)

        return notes


# ============================================================================
# Scale Encoder
# ============================================================================


class ScaleEncoder(BaseEncoder):
    """Encode MIDI notes as scale degrees within a musical scale.

    This encoder maps MIDI notes to scale degree tokens, ensuring all generated
    notes are in-key. Notes outside the scale are mapped to the nearest scale
    tone or a special OUT_OF_SCALE token.

    Token layout (for 7-note scale like major, 6 octaves):
    - 0 to (degrees * octaves - 1): Scale degrees across octaves
    - degrees * octaves: REST token
    - degrees * octaves + 1: OUT_OF_SCALE token (chromatic notes)

    Benefits:
    - Smaller vocabulary (e.g., 44 tokens for major scale vs 128)
    - All generated notes guaranteed to be in-key
    - Model learns melodic patterns as scale degrees (more musical)
    - Works across keys (1-3-5 is always a triad)

    Attributes:
        scale: The Scale instance defining the key and mode
        octave_range: Tuple of (min_octave, max_octave) for encoding
        snap_to_scale: If True, snap out-of-scale notes to nearest scale tone
                       If False, use OUT_OF_SCALE token

    Example:
        >>> from coremusic.music.theory import Scale, Note, ScaleType
        >>> scale = Scale(Note('C', 4), ScaleType.MAJOR)
        >>> encoder = ScaleEncoder(scale, octave_range=(3, 6))
        >>> encoder.vocab_size  # 7 degrees * 4 octaves + 2 special = 30
        30
        >>> # Encode C major notes
        >>> encoder.midi_to_token(60)  # C4 -> degree 0, octave 1
        7
        >>> encoder.midi_to_token(64)  # E4 -> degree 2, octave 1
        9
    """

    def __init__(
        self,
        scale: Scale,
        octave_range: Tuple[int, int] = (2, 8),
        snap_to_scale: bool = True,
        default_velocity: int = 100,
        default_duration: float = 0.25,
    ):
        """Initialize the ScaleEncoder.

        Args:
            scale: Scale instance defining the key and mode
            octave_range: (min_octave, max_octave) inclusive range for encoding
            snap_to_scale: If True, snap chromatic notes to nearest scale tone.
                           If False, use OUT_OF_SCALE token for chromatic notes.
            default_velocity: Velocity for decoded notes (0-127)
            default_duration: Duration in beats for decoded notes
        """
        self.scale = scale
        self.octave_range = octave_range
        self.snap_to_scale = snap_to_scale
        self._default_velocity = default_velocity
        self._default_duration = default_duration

        # Calculate dimensions
        self.num_octaves = octave_range[1] - octave_range[0]
        self.degrees_per_octave = len(scale.intervals)

        # Special tokens (at the end of vocabulary)
        self._base_vocab = self.degrees_per_octave * self.num_octaves
        self.REST_TOKEN = self._base_vocab
        self.OUT_OF_SCALE_TOKEN = self._base_vocab + 1

        # Build lookup tables for efficient encoding/decoding
        self._build_lookup_tables()

    def _build_lookup_tables(self) -> None:
        """Build MIDI <-> token lookup tables."""
        # midi_to_degree: maps MIDI note -> (degree, octave_offset) or None
        self._midi_to_degree: Dict[int, Optional[Tuple[int, int]]] = {}

        # degree_octave_to_midi: maps (degree, octave) -> MIDI note
        self._token_to_midi: Dict[int, int] = {}

        root_pc = self.scale.root.pitch_class
        intervals = self.scale.intervals

        # Calculate the MIDI note of the root at each requested octave
        # For A minor at octave 4: root MIDI = 69 (A4)
        # For C major at octave 4: root MIDI = 60 (C4)
        root_octave = self.scale.root.octave
        root_midi = self.scale.root.midi

        # Build mappings for each requested scale octave
        for octave_idx in range(self.num_octaves):
            # The actual octave number for this scale octave
            actual_octave = self.octave_range[0] + octave_idx

            # Calculate the root note for this scale octave
            # Offset from the original root's octave
            octave_offset = actual_octave - root_octave
            base_midi = root_midi + (octave_offset * 12)

            # Add all scale degrees for this octave
            for degree, interval in enumerate(intervals):
                midi = base_midi + interval
                # Ensure MIDI is in valid range
                if 0 <= midi <= 127:
                    self._midi_to_degree[midi] = (degree, octave_idx)

        # Build reverse mapping (token -> MIDI)
        for midi, deg_oct in self._midi_to_degree.items():
            if deg_oct is not None:
                degree, octave_idx = deg_oct
                token = octave_idx * self.degrees_per_octave + degree
                self._token_to_midi[token] = midi

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary (scale degrees * octaves + special tokens)."""
        return self._base_vocab + 2  # +REST, +OUT_OF_SCALE

    def midi_to_token(self, midi: int) -> int:
        """Convert a MIDI note number to a token.

        Args:
            midi: MIDI note number (0-127)

        Returns:
            Token index
        """
        if midi in self._midi_to_degree:
            deg_oct = self._midi_to_degree[midi]
            if deg_oct is not None:
                degree, octave_idx = deg_oct
                return octave_idx * self.degrees_per_octave + degree

        # Note is out of scale
        if self.snap_to_scale:
            # Find nearest scale tone
            return self._snap_to_nearest_token(midi)
        else:
            return self.OUT_OF_SCALE_TOKEN

    def _snap_to_nearest_token(self, midi: int) -> int:
        """Snap a chromatic note to the nearest scale tone.

        Args:
            midi: MIDI note number

        Returns:
            Token for nearest scale tone
        """
        # Try notes above and below until we find a scale tone
        for offset in range(1, 7):
            # Try below
            below = midi - offset
            if below in self._midi_to_degree and self._midi_to_degree[below] is not None:
                deg_oct = self._midi_to_degree[below]
                degree, octave_idx = deg_oct
                return octave_idx * self.degrees_per_octave + degree

            # Try above
            above = midi + offset
            if above in self._midi_to_degree and self._midi_to_degree[above] is not None:
                deg_oct = self._midi_to_degree[above]
                degree, octave_idx = deg_oct
                return octave_idx * self.degrees_per_octave + degree

        # Fallback to OUT_OF_SCALE if no nearby scale tone found
        return self.OUT_OF_SCALE_TOKEN

    def token_to_midi(self, token: int) -> Optional[int]:
        """Convert a token back to MIDI note number.

        Args:
            token: Token index

        Returns:
            MIDI note number, or None for special tokens
        """
        if token == self.REST_TOKEN or token == self.OUT_OF_SCALE_TOKEN:
            return None

        if token in self._token_to_midi:
            return self._token_to_midi[token]

        return None

    def encode(self, events: List[MIDIEvent]) -> List[int]:
        """Encode MIDI events as scale degree tokens.

        Args:
            events: List of MIDI events

        Returns:
            List of scale degree tokens
        """
        if not events:
            return []

        # Extract note-on events
        note_events = [e for e in events if e.is_note_on]
        if not note_events:
            return []

        # Sort by time
        note_events.sort(key=lambda e: e.time)

        tokens = []
        for event in note_events:
            token = self.midi_to_token(event.data1)
            # Skip OUT_OF_SCALE tokens if not snapping
            if token != self.OUT_OF_SCALE_TOKEN or not self.snap_to_scale:
                tokens.append(token)

        return tokens

    def decode(
        self,
        tokens: List[int],
        tempo: float = 120.0,
        channel: int = 0,
        **kwargs,
    ) -> List[MIDIEvent]:
        """Decode scale degree tokens back to MIDI events.

        Args:
            tokens: List of scale degree tokens
            tempo: Tempo in BPM
            channel: MIDI channel (0-15)
            **kwargs: Additional options (unused)

        Returns:
            List of MIDIEvent objects
        """
        events = []
        beat_duration = 60.0 / tempo
        note_duration = self._default_duration * beat_duration
        current_time = 0.0

        for token in tokens:
            if token == self.REST_TOKEN:
                current_time += note_duration
                continue
            if token == self.OUT_OF_SCALE_TOKEN:
                current_time += note_duration
                continue

            midi = self.token_to_midi(token)
            if midi is None:
                current_time += note_duration
                continue

            # Note on
            events.append(
                MIDIEvent(
                    time=current_time,
                    status=MIDIStatus.NOTE_ON,
                    channel=channel,
                    data1=midi,
                    data2=self._default_velocity,
                )
            )

            # Note off
            events.append(
                MIDIEvent(
                    time=current_time + note_duration,
                    status=MIDIStatus.NOTE_OFF,
                    channel=channel,
                    data1=midi,
                    data2=0,
                )
            )

            current_time += note_duration

        events.sort(key=lambda e: e.time)
        return events

    def transpose(self, tokens: List[int], semitones: int) -> List[int]:
        """Transpose tokens by semitones.

        For scale encoding, transposition by scale intervals is more meaningful.
        This method transposes by the nearest scale degree.

        Args:
            tokens: List of tokens
            semitones: Number of semitones to transpose

        Returns:
            Transposed tokens
        """
        if semitones == 0:
            return tokens.copy()

        # Calculate degree offset (approximate)
        # For diatonic scales, 1-2 semitones ~= 1 degree
        degree_offset = round(semitones * len(self.scale.intervals) / 12)

        transposed = []
        for token in tokens:
            if token >= self._base_vocab:
                # Special token - keep as is
                transposed.append(token)
            else:
                # Scale degree token
                octave_idx = token // self.degrees_per_octave
                degree = token % self.degrees_per_octave

                # Apply transposition
                new_degree = degree + degree_offset
                new_octave = octave_idx + (new_degree // self.degrees_per_octave)
                new_degree = new_degree % self.degrees_per_octave

                # Check bounds
                if 0 <= new_octave < self.num_octaves:
                    new_token = new_octave * self.degrees_per_octave + new_degree
                    transposed.append(new_token)
                else:
                    # Out of range - use REST
                    transposed.append(self.REST_TOKEN)

        return transposed

    def transpose_by_degree(self, tokens: List[int], degrees: int) -> List[int]:
        """Transpose tokens by scale degrees.

        This is more musically meaningful than semitone transposition for
        scale-encoded sequences.

        Args:
            tokens: List of tokens
            degrees: Number of scale degrees to transpose (positive = up)

        Returns:
            Transposed tokens
        """
        if degrees == 0:
            return tokens.copy()

        transposed = []
        for token in tokens:
            if token >= self._base_vocab:
                # Special token - keep as is
                transposed.append(token)
            else:
                # Scale degree token
                octave_idx = token // self.degrees_per_octave
                degree = token % self.degrees_per_octave

                # Apply transposition
                new_degree = degree + degrees
                new_octave = octave_idx + (new_degree // self.degrees_per_octave)
                new_degree = new_degree % self.degrees_per_octave

                # Check bounds
                if 0 <= new_octave < self.num_octaves:
                    new_token = new_octave * self.degrees_per_octave + new_degree
                    transposed.append(new_token)
                else:
                    # Out of range - use REST
                    transposed.append(self.REST_TOKEN)

        return transposed

    def get_scale_notes(self) -> List[Tuple[int, int, str]]:
        """Get all scale notes with their tokens and names.

        Returns:
            List of (token, midi, note_name) tuples
        """
        notes = []
        for token, midi in sorted(self._token_to_midi.items()):
            note = Note.from_midi(midi)
            notes.append((token, midi, str(note)))
        return notes

    def __repr__(self) -> str:
        return (
            f"ScaleEncoder({self.scale}, octaves={self.octave_range}, "
            f"vocab_size={self.vocab_size})"
        )


# ============================================================================
# MIDI Dataset
# ============================================================================


@dataclass
class MIDIDataset:
    """Dataset for loading and preprocessing MIDI files for neural network training.

    This class handles:
    - Loading MIDI files and encoding them to token sequences
    - Data augmentation (transposition)
    - Preparing training data (X, Y arrays) for sequence prediction
    - Track/channel filtering for cleaner training data

    Example:
        >>> encoder = NoteEncoder()
        >>> dataset = MIDIDataset(encoder, seq_length=32)
        >>> dataset.load_file('music.mid')
        >>> dataset.augment(transpose_range=(-5, 7))
        >>> x_train, y_train = dataset.prepare_training_data()

    Example with filtering (train on melody only):
        >>> encoder = NoteEncoder()
        >>> dataset = MIDIDataset(
        ...     encoder,
        ...     seq_length=32,
        ...     channels=[0],           # Only channel 0
        ...     pitch_range=(48, 84),   # C3-C6 melodic range
        ...     exclude_drums=True,
        ... )
        >>> dataset.load_directory('midi_files/')

    Using presets:
        >>> dataset = MIDIDataset.for_melody(NoteEncoder(), seq_length=32)
        >>> dataset = MIDIDataset.for_drums(NoteEncoder(), seq_length=16)
        >>> dataset = MIDIDataset.for_bass(NoteEncoder(), seq_length=32)
    """

    encoder: BaseEncoder
    seq_length: int
    sequences: List[List[int]] = field(default_factory=list)
    # Filter options (applied to all loaded files)
    channels: Optional[List[int]] = None
    tracks: Optional[List[int]] = None
    track_names: Optional[List[str]] = None
    pitch_range: Optional[Tuple[int, int]] = None
    exclude_drums: bool = False

    @classmethod
    def for_melody(
        cls,
        encoder: BaseEncoder,
        seq_length: int,
        pitch_range: Tuple[int, int] = (48, 84),
        **kwargs,
    ) -> "MIDIDataset":
        """Create dataset preset for melodic training.

        Filters out drums and limits to melodic pitch range (C3-C6 by default).
        Best for training models on melody lines, lead instruments, etc.

        Args:
            encoder: Encoder instance
            seq_length: Sequence length for training
            pitch_range: Min/max pitch to include (default: 48-84, C3-C6)
            **kwargs: Additional MIDIDataset arguments

        Returns:
            Configured MIDIDataset instance
        """
        return cls(
            encoder=encoder,
            seq_length=seq_length,
            exclude_drums=True,
            pitch_range=pitch_range,
            **kwargs,
        )

    @classmethod
    def for_drums(
        cls,
        encoder: BaseEncoder,
        seq_length: int,
        **kwargs,
    ) -> "MIDIDataset":
        """Create dataset preset for drum/rhythm training.

        No filtering applied - suitable for drum pattern MIDI files where
        note numbers represent different percussion instruments.

        Args:
            encoder: Encoder instance
            seq_length: Sequence length for training
            **kwargs: Additional MIDIDataset arguments

        Returns:
            Configured MIDIDataset instance
        """
        return cls(
            encoder=encoder,
            seq_length=seq_length,
            exclude_drums=False,
            **kwargs,
        )

    @classmethod
    def for_bass(
        cls,
        encoder: BaseEncoder,
        seq_length: int,
        pitch_range: Tuple[int, int] = (28, 60),
        **kwargs,
    ) -> "MIDIDataset":
        """Create dataset preset for bass line training.

        Filters out drums and limits to bass pitch range (E1-C4 by default).
        Best for training models on bass lines.

        Args:
            encoder: Encoder instance
            seq_length: Sequence length for training
            pitch_range: Min/max pitch to include (default: 28-60, E1-C4)
            **kwargs: Additional MIDIDataset arguments

        Returns:
            Configured MIDIDataset instance
        """
        return cls(
            encoder=encoder,
            seq_length=seq_length,
            exclude_drums=True,
            pitch_range=pitch_range,
            **kwargs,
        )

    @classmethod
    def for_chords(
        cls,
        encoder: BaseEncoder,
        seq_length: int,
        pitch_range: Tuple[int, int] = (36, 72),
        **kwargs,
    ) -> "MIDIDataset":
        """Create dataset preset for chord/harmony training.

        Filters out drums and uses a mid-range pitch range (C2-C5 by default).
        Best for training models on chord progressions and harmony.

        Args:
            encoder: Encoder instance
            seq_length: Sequence length for training
            pitch_range: Min/max pitch to include (default: 36-72, C2-C5)
            **kwargs: Additional MIDIDataset arguments

        Returns:
            Configured MIDIDataset instance
        """
        return cls(
            encoder=encoder,
            seq_length=seq_length,
            exclude_drums=True,
            pitch_range=pitch_range,
            **kwargs,
        )

    def load_file(self, path: Union[str, Path], **filter_overrides) -> int:
        """Load a single MIDI file and add its encoded sequence.

        Args:
            path: Path to MIDI file
            **filter_overrides: Override default filter options for this file:
                - channels: Only include events from these MIDI channels
                - tracks: Only include events from these track indices
                - track_names: Only include tracks with matching names
                - pitch_range: Only include notes in (min, max) range
                - exclude_drums: Exclude channel 9 (GM drums)

        Returns:
            Number of tokens extracted from the file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"MIDI file not found: {path}")

        # Merge default filters with overrides
        filters = {
            'channels': self.channels,
            'tracks': self.tracks,
            'track_names': self.track_names,
            'pitch_range': self.pitch_range,
            'exclude_drums': self.exclude_drums,
        }
        filters.update(filter_overrides)

        tokens = self.encoder.encode_file(path, **filters)
        if tokens:
            self.sequences.append(tokens)
        return len(tokens)

    def load_directory(
        self, path: Union[str, Path], pattern: str = "*.mid", recursive: bool = False
    ) -> int:
        """Load all MIDI files from a directory.

        Args:
            path: Directory path
            pattern: Glob pattern for matching files
            recursive: Whether to search recursively

        Returns:
            Total number of tokens loaded
        """
        path = Path(path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        total_tokens = 0
        glob_method = path.rglob if recursive else path.glob

        for midi_file in glob_method(pattern):
            try:
                total_tokens += self.load_file(midi_file)
            except Exception as e:
                # Log but continue loading other files
                print(f"Warning: Failed to load {midi_file}: {e}")

        return total_tokens

    def add_sequence(self, tokens: List[int]) -> None:
        """Add a pre-encoded token sequence directly.

        Args:
            tokens: List of integer tokens
        """
        if tokens:
            self.sequences.append(tokens)

    def clear(self) -> None:
        """Clear all loaded sequences."""
        self.sequences.clear()

    @property
    def n_sequences(self) -> int:
        """Number of loaded sequences."""
        return len(self.sequences)

    @property
    def total_tokens(self) -> int:
        """Total number of tokens across all sequences."""
        return sum(len(seq) for seq in self.sequences)

    def augment(
        self,
        transpose_range: Tuple[int, int] = (-6, 6),
        include_original: bool = True,
    ) -> int:
        """Augment data by transposition.

        Creates transposed copies of all sequences within the specified
        semitone range.

        Args:
            transpose_range: (min, max) semitones for transposition
            include_original: Whether to keep original (non-transposed) sequences

        Returns:
            Number of new sequences created
        """
        # Check if encoder supports transposition
        if not hasattr(self.encoder, 'transpose'):
            raise NotImplementedError(
                f"Augmentation not supported for {self.encoder.__class__.__name__} "
                "(no transpose method)"
            )

        min_t, max_t = transpose_range
        original_sequences = self.sequences.copy()
        new_sequences = []

        for seq in original_sequences:
            for semitones in range(min_t, max_t + 1):
                if semitones == 0 and include_original:
                    continue  # Skip 0 transposition if keeping original
                transposed = self.encoder.transpose(seq, semitones)
                new_sequences.append(transposed)

        if not include_original:
            self.sequences.clear()

        self.sequences.extend(new_sequences)
        return len(new_sequences)

    def prepare_training_data(
        self, one_hot: bool = True, use_numpy: bool = True
    ) -> Tuple:
        """Prepare training data for sequence prediction.

        Creates (X, Y) pairs where X is a sequence of tokens and Y is the
        next token to predict.

        Args:
            one_hot: Whether to one-hot encode the data
            use_numpy: If True, return numpy arrays; if False, return Array2D

        Returns:
            Tuple of (x_train, y_train) as numpy arrays or Array2D objects

        Raises:
            ValueError: If no valid sequences available
        """
        x_list = []
        y_list = []
        vocab_size = self.encoder.vocab_size

        for seq in self.sequences:
            if len(seq) <= self.seq_length:
                continue

            for i in range(len(seq) - self.seq_length):
                if one_hot:
                    # Input: one-hot encoded sequence (flattened)
                    x_seq = [0.0] * (self.seq_length * vocab_size)
                    for j in range(self.seq_length):
                        val = seq[i + j]
                        if 0 <= val < vocab_size:
                            x_seq[j * vocab_size + val] = 1.0
                    x_list.append(x_seq)

                    # Target: next token (one-hot)
                    y_seq = [0.0] * vocab_size
                    val = seq[i + self.seq_length]
                    if 0 <= val < vocab_size:
                        y_seq[val] = 1.0
                    y_list.append(y_seq)
                else:
                    # Raw integer sequences
                    x_list.append(list(seq[i : i + self.seq_length]))
                    y_list.append([seq[i + self.seq_length]])

        if not x_list:
            raise ValueError(
                "No valid sequences found. Ensure sequences are longer than seq_length."
            )

        if use_numpy:
            try:
                import numpy as np
                x_train = np.array(x_list, dtype=np.float32)
                y_train = np.array(y_list, dtype=np.float32)
                return x_train, y_train
            except ImportError:
                pass  # Fall back to Array2D

        # Convert to Array2D
        from coremusic.kann import Array2D

        x_rows = len(x_list)
        x_cols = len(x_list[0])
        y_rows = len(y_list)
        y_cols = len(y_list[0])

        x_flat = []
        for row in x_list:
            x_flat.extend(row)

        y_flat = []
        for row in y_list:
            y_flat.extend(row)

        x_train = Array2D(x_rows, x_cols, x_flat)
        y_train = Array2D(y_rows, y_cols, y_flat)

        return x_train, y_train

    def prepare_rnn_training_data(
        self,
    ) -> Tuple["Array2D", "Array2D"]:
        """Prepare training data for RNN models.

        For RNNs, input is one-hot encoded at each timestep (not flattened).

        Returns:
            Tuple of (x_train, y_train) as Array2D objects
        """
        from coremusic.kann import Array2D

        x_list = []
        y_list = []
        vocab_size = self.encoder.vocab_size

        for seq in self.sequences:
            if len(seq) <= self.seq_length:
                continue

            for i in range(len(seq) - self.seq_length):
                # For RNNs, we typically unroll and process one step at a time
                # Input: single token one-hot encoded
                for j in range(self.seq_length):
                    x_seq = [0.0] * vocab_size
                    val = seq[i + j]
                    if 0 <= val < vocab_size:
                        x_seq[val] = 1.0
                    x_list.append(x_seq)

                    # Target: next token (one-hot)
                    y_seq = [0.0] * vocab_size
                    next_val = seq[i + j + 1]
                    if 0 <= next_val < vocab_size:
                        y_seq[next_val] = 1.0
                    y_list.append(y_seq)

        if not x_list:
            raise ValueError(
                "No valid sequences found. Ensure sequences are longer than seq_length."
            )

        # Convert to Array2D
        x_rows = len(x_list)
        x_cols = vocab_size
        y_rows = len(y_list)
        y_cols = vocab_size

        x_flat = []
        for row in x_list:
            x_flat.extend(row)

        y_flat = []
        for row in y_list:
            y_flat.extend(row)

        x_train = Array2D(x_rows, x_cols, x_flat)
        y_train = Array2D(y_rows, y_cols, y_flat)

        return x_train, y_train

    def get_sample_sequence(self, length: Optional[int] = None) -> List[int]:
        """Get a random sample sequence for seeding generation.

        Args:
            length: Desired sequence length (defaults to seq_length)

        Returns:
            List of tokens
        """
        if not self.sequences:
            raise ValueError("No sequences loaded")

        length = length or self.seq_length

        # Find a sequence long enough
        valid_seqs = [s for s in self.sequences if len(s) >= length]
        if not valid_seqs:
            raise ValueError(f"No sequences of length >= {length}")

        seq = random.choice(valid_seqs)
        start = random.randint(0, len(seq) - length)
        return seq[start : start + length]

    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences)

    def __repr__(self) -> str:
        return (
            f"MIDIDataset(encoder={self.encoder.__class__.__name__}, "
            f"seq_length={self.seq_length}, "
            f"sequences={len(self.sequences)}, "
            f"total_tokens={self.total_tokens})"
        )
