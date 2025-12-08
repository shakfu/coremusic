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
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from coremusic.midi.utilities import MIDIEvent, MIDISequence, MIDIStatus

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

    def encode_file(self, path: Union[str, Path]) -> List[int]:
        """Encode a MIDI file to tokens.

        Args:
            path: Path to MIDI file

        Returns:
            List of integer tokens
        """
        sequence = MIDISequence.load(str(path))
        all_events = []
        for track in sequence.tracks:
            all_events.extend(track.events)
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
            default_duration: Duration in seconds for decoded notes
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
# MIDI Dataset
# ============================================================================


@dataclass
class MIDIDataset:
    """Dataset for loading and preprocessing MIDI files for neural network training.

    This class handles:
    - Loading MIDI files and encoding them to token sequences
    - Data augmentation (transposition)
    - Preparing training data (X, Y arrays) for sequence prediction

    Example:
        >>> encoder = NoteEncoder()
        >>> dataset = MIDIDataset(encoder, seq_length=32)
        >>> dataset.load_file('music.mid')
        >>> dataset.augment(transpose_range=(-5, 7))
        >>> x_train, y_train = dataset.prepare_training_data()
    """

    encoder: BaseEncoder
    seq_length: int
    sequences: List[List[int]] = field(default_factory=list)

    def load_file(self, path: Union[str, Path]) -> int:
        """Load a single MIDI file and add its encoded sequence.

        Args:
            path: Path to MIDI file

        Returns:
            Number of tokens extracted from the file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"MIDI file not found: {path}")

        tokens = self.encoder.encode_file(path)
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
        if not isinstance(self.encoder, NoteEncoder):
            raise NotImplementedError(
                "Augmentation currently only supported for NoteEncoder"
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
