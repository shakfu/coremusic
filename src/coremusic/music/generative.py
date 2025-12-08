#!/usr/bin/env python3
"""Generative music algorithms for MIDI composition.

This module provides generators for creating MIDI note sequences using
various algorithmic composition techniques:

- Arpeggiator: Flexible arpeggio patterns from chords
- EuclideanGenerator: Euclidean rhythm patterns
- MarkovGenerator: Markov chain-based melody generation
- ProbabilisticGenerator: Weighted random note selection
- SequenceGenerator: Step sequencer patterns
- MelodyGenerator: Rule-based melodic generation
- PolyrhythmGenerator: Multiple simultaneous rhythms

All generators produce MIDIEvent objects compatible with coremusic.midi.

Example:
    >>> from coremusic.music import theory, generative
    >>>
    >>> # Create an arpeggiator
    >>> chord = theory.Chord(theory.Note('C', 4), theory.ChordType.MAJOR_7)
    >>> arp = generative.Arpeggiator(chord, generative.ArpPattern.UP_DOWN)
    >>> events = arp.generate(num_cycles=2)
    >>>
    >>> # Create a Euclidean rhythm
    >>> euclid = generative.EuclideanGenerator(pulses=5, steps=8, pitch=36)
    >>> events = euclid.generate(cycles=4)
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

from ..midi.utilities import MIDIEvent, MIDIStatus
from .theory import Chord, Note, Scale

# ============================================================================
# Base Classes
# ============================================================================


@dataclass
class GeneratorConfig:
    """Base configuration for all generators.

    Attributes:
        tempo: Tempo in BPM (for time calculations)
        channel: MIDI channel (0-15)
        velocity: Default velocity (0-127)
        swing: Swing amount (0.0 = none, 1.0 = full triplet swing)
        humanize_timing: Random timing variation in seconds
        humanize_velocity: Random velocity variation (0-127)
        seed: Random seed for reproducibility (None = random)
    """
    tempo: float = 120.0
    channel: int = 0
    velocity: int = 100
    swing: float = 0.0
    humanize_timing: float = 0.0
    humanize_velocity: int = 0
    seed: Optional[int] = None

    def __post_init__(self):
        if not 0 <= self.channel <= 15:
            raise ValueError(f"Channel must be 0-15, got {self.channel}")
        if not 0 <= self.velocity <= 127:
            raise ValueError(f"Velocity must be 0-127, got {self.velocity}")
        if not 0.0 <= self.swing <= 1.0:
            raise ValueError(f"Swing must be 0.0-1.0, got {self.swing}")


class Generator(ABC):
    """Abstract base class for all generators.

    Subclasses must implement the generate() method.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """Initialize generator.

        Args:
            config: Generator configuration
        """
        self.config = config or GeneratorConfig()
        self._rng = random.Random(self.config.seed)

    def _apply_swing(self, time: float, step: int) -> float:
        """Apply swing to a time value.

        Args:
            time: Original time in seconds
            step: Step number (swing affects odd steps)

        Returns:
            Time with swing applied
        """
        if self.config.swing == 0.0 or step % 2 == 0:
            return time

        # Swing delays odd steps toward the next beat
        beat_duration = 60.0 / self.config.tempo
        swing_delay = beat_duration * 0.333 * self.config.swing  # Max = triplet feel
        return time + swing_delay

    def _apply_humanize(self, time: float, velocity: int) -> Tuple[float, int]:
        """Apply humanization to timing and velocity.

        Args:
            time: Original time
            velocity: Original velocity

        Returns:
            Tuple of (humanized_time, humanized_velocity)
        """
        # Timing humanization
        if self.config.humanize_timing > 0:
            time += self._rng.uniform(-self.config.humanize_timing, self.config.humanize_timing)
            time = max(0.0, time)

        # Velocity humanization
        if self.config.humanize_velocity > 0:
            velocity += self._rng.randint(-self.config.humanize_velocity, self.config.humanize_velocity)
            velocity = max(1, min(127, velocity))

        return time, velocity

    def _create_note_events(
        self,
        time: float,
        pitch: int,
        velocity: int,
        duration: float,
    ) -> List[MIDIEvent]:
        """Create note on/off events.

        Args:
            time: Start time in seconds
            pitch: MIDI note number
            velocity: Note velocity
            duration: Note duration in seconds

        Returns:
            List of [NoteOn, NoteOff] events
        """
        return [
            MIDIEvent(time, MIDIStatus.NOTE_ON, self.config.channel, pitch, velocity),
            MIDIEvent(time + duration, MIDIStatus.NOTE_OFF, self.config.channel, pitch, 0),
        ]

    @abstractmethod
    def generate(self, **kwargs: Any) -> List[MIDIEvent]:
        """Generate MIDI events.

        Subclasses override this with specific signatures.

        Returns:
            List of MIDIEvent objects
        """
        pass

    def generate_to_track(self, track, **kwargs) -> None:
        """Generate events directly to a MIDITrack.

        Args:
            track: MIDITrack to add events to
            **kwargs: Generator-specific arguments
        """
        events = self.generate(**kwargs)
        for event in events:
            if event.status == MIDIStatus.NOTE_ON and event.data2 > 0:
                # Find matching note off
                note_off_time = None
                for off_event in events:
                    if (off_event.status == MIDIStatus.NOTE_OFF and
                        off_event.data1 == event.data1 and
                        off_event.time > event.time):
                        note_off_time = off_event.time
                        break
                if note_off_time:
                    duration = note_off_time - event.time
                    track.add_note(event.time, event.data1, event.data2, duration, event.channel)


# ============================================================================
# Arpeggiator
# ============================================================================


class ArpPattern(Enum):
    """Arpeggiator pattern types."""
    UP = auto()              # Ascending
    DOWN = auto()            # Descending
    UP_DOWN = auto()         # Ascending then descending
    DOWN_UP = auto()         # Descending then ascending
    UP_DOWN_INCLUSIVE = auto()  # Up/down including top/bottom notes twice
    RANDOM = auto()          # Random order
    RANDOM_WALK = auto()     # Random steps (adjacent notes)
    AS_PLAYED = auto()       # Order as defined in chord
    OUTSIDE_IN = auto()      # Lowest, highest, 2nd lowest, 2nd highest...
    INSIDE_OUT = auto()      # Middle outward


@dataclass
class ArpConfig(GeneratorConfig):
    """Arpeggiator configuration.

    Attributes:
        note_duration: Duration of each note in beats
        gate: Gate time as fraction of note duration (0.0-1.0)
        octave_range: Number of octaves to span
        rate: Rate in beats per note (1.0 = quarter, 0.5 = eighth, etc.)
        latch: If True, hold notes until next chord
        velocity_pattern: Optional list of velocities to cycle through
        accent_pattern: Optional list of accent positions (1-indexed)
        accent_velocity: Velocity for accented notes
    """
    note_duration: float = 0.5  # In beats
    gate: float = 0.9
    octave_range: int = 1
    rate: float = 0.25  # 16th notes
    latch: bool = False
    velocity_pattern: Optional[List[int]] = None
    accent_pattern: Optional[List[int]] = None
    accent_velocity: int = 127


class Arpeggiator(Generator):
    """Versatile arpeggiator for chord-based patterns.

    Creates arpeggiated sequences from chords using various patterns,
    with support for multiple octaves, swing, and humanization.

    Example:
        >>> chord = Chord(Note('C', 4), ChordType.MAJOR_7)
        >>> arp = Arpeggiator(chord, ArpPattern.UP_DOWN)
        >>> events = arp.generate(num_cycles=2)
        >>>
        >>> # With configuration
        >>> config = ArpConfig(tempo=140, swing=0.3, octave_range=2)
        >>> arp = Arpeggiator(chord, ArpPattern.RANDOM, config=config)
    """

    def __init__(
        self,
        chord: Optional[Chord] = None,
        pattern: ArpPattern = ArpPattern.UP,
        config: Optional[ArpConfig] = None,
    ):
        """Initialize arpeggiator.

        Args:
            chord: Chord to arpeggiate
            pattern: Arpeggio pattern type
            config: Arpeggiator configuration
        """
        super().__init__(config or ArpConfig())
        self.chord = chord
        self.pattern = pattern
        self._current_notes: List[int] = []
        self._pattern_index = 0

    @property  # type: ignore[override]
    def config(self) -> ArpConfig:
        return self._config

    @config.setter
    def config(self, value: ArpConfig):
        self._config = value

    def set_chord(self, chord: Chord) -> None:
        """Set new chord to arpeggiate.

        Args:
            chord: New chord
        """
        self.chord = chord
        self._current_notes = self._build_note_sequence()
        self._pattern_index = 0

    def _build_note_sequence(self) -> List[int]:
        """Build the sequence of MIDI notes based on pattern."""
        if not self.chord:
            return []

        # Get base notes from chord
        base_notes = self.chord.get_midi_notes()

        # Extend across octaves
        all_notes = []
        for octave in range(self.config.octave_range):
            for note in base_notes:
                transposed = note + (octave * 12)
                if 0 <= transposed <= 127:
                    all_notes.append(transposed)

        # Apply pattern
        if self.pattern == ArpPattern.UP:
            return sorted(all_notes)

        elif self.pattern == ArpPattern.DOWN:
            return sorted(all_notes, reverse=True)

        elif self.pattern == ArpPattern.UP_DOWN:
            up = sorted(all_notes)
            down = sorted(all_notes, reverse=True)[1:-1]  # Exclude endpoints
            return up + down

        elif self.pattern == ArpPattern.DOWN_UP:
            down = sorted(all_notes, reverse=True)
            up = sorted(all_notes)[1:-1]
            return down + up

        elif self.pattern == ArpPattern.UP_DOWN_INCLUSIVE:
            up = sorted(all_notes)
            down = sorted(all_notes, reverse=True)
            return up + down

        elif self.pattern == ArpPattern.RANDOM:
            notes = all_notes.copy()
            self._rng.shuffle(notes)
            return notes

        elif self.pattern == ArpPattern.RANDOM_WALK:
            # Start from middle, random adjacent steps
            sorted_notes = sorted(all_notes)
            idx = len(sorted_notes) // 2
            result = [sorted_notes[idx]]
            for _ in range(len(all_notes) - 1):
                idx += self._rng.choice([-1, 1])
                idx = max(0, min(len(sorted_notes) - 1, idx))
                result.append(sorted_notes[idx])
            return result

        elif self.pattern == ArpPattern.AS_PLAYED:
            return all_notes

        elif self.pattern == ArpPattern.OUTSIDE_IN:
            sorted_notes = sorted(all_notes)
            result = []
            left, right = 0, len(sorted_notes) - 1
            while left <= right:
                result.append(sorted_notes[left])
                if left != right:
                    result.append(sorted_notes[right])
                left += 1
                right -= 1
            return result

        elif self.pattern == ArpPattern.INSIDE_OUT:
            sorted_notes = sorted(all_notes)
            result = []
            mid = len(sorted_notes) // 2
            for i in range(len(sorted_notes)):
                if i % 2 == 0:
                    idx = mid + (i // 2)
                else:
                    idx = mid - ((i + 1) // 2)
                if 0 <= idx < len(sorted_notes):
                    result.append(sorted_notes[idx])
            return result

        return all_notes  # type: ignore[unreachable]

    def generate(  # type: ignore[override]
        self,
        num_cycles: int = 1,
        start_time: float = 0.0,
        duration: Optional[float] = None,
    ) -> List[MIDIEvent]:
        """Generate arpeggio events.

        Args:
            num_cycles: Number of pattern cycles
            start_time: Start time in seconds
            duration: Total duration (overrides num_cycles if set)

        Returns:
            List of MIDIEvent objects
        """
        if not self.chord:
            return []

        notes = self._build_note_sequence()
        if not notes:
            return []

        events = []
        beat_duration = 60.0 / self.config.tempo
        note_time = self.config.rate * beat_duration
        note_duration = self.config.note_duration * beat_duration * self.config.gate

        current_time = start_time
        step = 0

        if duration is not None:
            # Generate for specific duration
            end_time = start_time + duration
            while current_time < end_time:
                note_idx = step % len(notes)
                pitch = notes[note_idx]

                # Apply swing
                time = self._apply_swing(current_time, step)

                # Get velocity
                velocity = self._get_velocity(step)

                # Apply humanization
                time, velocity = self._apply_humanize(time, velocity)

                events.extend(self._create_note_events(time, pitch, velocity, note_duration))

                current_time += note_time
                step += 1
        else:
            # Generate for num_cycles
            total_notes = len(notes) * num_cycles
            for i in range(total_notes):
                note_idx = i % len(notes)
                pitch = notes[note_idx]

                time = self._apply_swing(current_time, step)
                velocity = self._get_velocity(step)
                time, velocity = self._apply_humanize(time, velocity)

                events.extend(self._create_note_events(time, pitch, velocity, note_duration))

                current_time += note_time
                step += 1

        return sorted(events, key=lambda e: e.time)

    def _get_velocity(self, step: int) -> int:
        """Get velocity for a step, considering patterns and accents."""
        velocity = self.config.velocity

        # Check velocity pattern
        if self.config.velocity_pattern:
            pattern_idx = step % len(self.config.velocity_pattern)
            velocity = self.config.velocity_pattern[pattern_idx]

        # Check accent pattern
        if self.config.accent_pattern:
            if (step + 1) in self.config.accent_pattern or \
               ((step % len(self.config.accent_pattern or [1])) + 1) in self.config.accent_pattern:
                velocity = self.config.accent_velocity

        return velocity

    def __repr__(self) -> str:
        return f"Arpeggiator({self.chord}, {self.pattern.name})"


# ============================================================================
# Euclidean Generator
# ============================================================================


@dataclass
class EuclideanConfig(GeneratorConfig):
    """Euclidean rhythm generator configuration.

    Attributes:
        note_duration: Duration of each note in beats
        gate: Gate time as fraction
        rotation: Pattern rotation (shift pattern start)
    """
    note_duration: float = 0.5
    gate: float = 0.8
    rotation: int = 0


class EuclideanGenerator(Generator):
    """Euclidean rhythm pattern generator.

    Creates rhythms using the Euclidean algorithm, which distributes
    pulses as evenly as possible across steps. This produces many
    traditional rhythmic patterns (e.g., 3 over 8 = Cuban tresillo).

    Example:
        >>> euclid = EuclideanGenerator(pulses=3, steps=8, pitch=36)
        >>> pattern = euclid.get_pattern()  # [1, 0, 0, 1, 0, 0, 1, 0]
        >>> events = euclid.generate(cycles=4)
    """

    def __init__(
        self,
        pulses: int,
        steps: int,
        pitch: Union[int, Note] = 60,
        config: Optional[EuclideanConfig] = None,
    ):
        """Initialize Euclidean generator.

        Args:
            pulses: Number of hits/pulses
            steps: Total number of steps
            pitch: MIDI note number or Note
            config: Generator configuration
        """
        super().__init__(config or EuclideanConfig())
        self.pulses = pulses
        self.steps = steps
        self.pitch = pitch.midi if isinstance(pitch, Note) else pitch
        self._pattern = self._compute_pattern()

    @property  # type: ignore[override]
    def config(self) -> EuclideanConfig:
        return self._config

    @config.setter
    def config(self, value: EuclideanConfig):
        self._config = value

    def _compute_pattern(self) -> List[int]:
        """Compute Euclidean pattern using Bjorklund's algorithm."""
        if self.pulses > self.steps:
            raise ValueError(f"Pulses ({self.pulses}) cannot exceed steps ({self.steps})")

        if self.pulses == 0:
            return [0] * self.steps
        if self.pulses == self.steps:
            return [1] * self.steps

        # Bjorklund's algorithm
        pattern = [[1]] * self.pulses + [[0]] * (self.steps - self.pulses)

        while True:
            # Count trailing zeros
            zeros_count = sum(1 for p in pattern if p == [0])
            if zeros_count <= 1:
                break

            # Distribute zeros
            ones_count = len(pattern) - zeros_count
            iterations = min(ones_count, zeros_count)

            new_pattern = []
            for i in range(iterations):
                new_pattern.append(pattern[i] + pattern[-(i + 1)])
            new_pattern.extend(pattern[iterations:-iterations or None])

            pattern = new_pattern

        # Flatten
        result = [item for sublist in pattern for item in sublist]

        # Apply rotation
        rotation = self.config.rotation % len(result)
        if rotation:
            result = result[rotation:] + result[:rotation]

        return result

    def get_pattern(self) -> List[int]:
        """Get the binary pattern.

        Returns:
            List of 1s (hit) and 0s (rest)
        """
        return self._pattern.copy()

    def set_parameters(self, pulses: int, steps: int) -> None:
        """Update pattern parameters.

        Args:
            pulses: Number of pulses
            steps: Number of steps
        """
        self.pulses = pulses
        self.steps = steps
        self._pattern = self._compute_pattern()

    def generate(  # type: ignore[override]
        self,
        cycles: int = 1,
        start_time: float = 0.0,
        step_duration: Optional[float] = None,
    ) -> List[MIDIEvent]:
        """Generate Euclidean rhythm events.

        Args:
            cycles: Number of pattern cycles
            start_time: Start time in seconds
            step_duration: Duration per step (default: beat / 4 = 16th notes)

        Returns:
            List of MIDIEvent objects
        """
        events = []
        beat_duration = 60.0 / self.config.tempo

        if step_duration is None:
            step_duration = beat_duration / 4  # 16th notes by default

        note_duration = self.config.note_duration * beat_duration * self.config.gate
        current_time = start_time

        for cycle in range(cycles):
            for step, hit in enumerate(self._pattern):
                if hit:
                    time = self._apply_swing(current_time, step + cycle * len(self._pattern))
                    velocity = self.config.velocity
                    time, velocity = self._apply_humanize(time, velocity)

                    events.extend(self._create_note_events(time, self.pitch, velocity, note_duration))

                current_time += step_duration

        return sorted(events, key=lambda e: e.time)

    def __repr__(self) -> str:
        return f"EuclideanGenerator(E({self.pulses}, {self.steps}))"


# ============================================================================
# Markov Chain Generator
# ============================================================================


@dataclass
class MarkovConfig(GeneratorConfig):
    """Markov chain generator configuration.

    Attributes:
        note_duration: Duration of each note in beats
        gate: Gate time as fraction
        order: Markov chain order (1 = first order, 2 = second order, etc.)
    """
    note_duration: float = 0.5
    gate: float = 0.9
    order: int = 1


class MarkovGenerator(Generator):
    """Markov chain-based melodic generator.

    Uses transition probabilities learned from input sequences
    or manually defined to generate new melodies.

    Example:
        >>> markov = MarkovGenerator()
        >>> # Train from melody
        >>> markov.train([60, 62, 64, 62, 60, 62, 64, 67])
        >>> events = markov.generate(num_notes=16, start_note=60)
        >>>
        >>> # Or define transitions manually
        >>> markov.set_transitions({
        ...     60: {62: 0.5, 64: 0.3, 67: 0.2},
        ...     62: {60: 0.4, 64: 0.6},
        ... })
    """

    def __init__(
        self,
        transitions: Optional[Dict[int, Dict[int, float]]] = None,
        scale: Optional[Scale] = None,
        config: Optional[MarkovConfig] = None,
    ):
        """Initialize Markov generator.

        Args:
            transitions: Transition probability matrix {from_note: {to_note: prob}}
            scale: Optional scale to constrain notes to
            config: Generator configuration
        """
        super().__init__(config or MarkovConfig())
        self.transitions = transitions or {}
        self.scale = scale
        self._history: List[int] = []

    @property  # type: ignore[override]
    def config(self) -> MarkovConfig:
        return self._config

    @config.setter
    def config(self, value: MarkovConfig):
        self._config = value

    def train(self, sequence: List[Union[int, Note]], order: Optional[int] = None) -> None:
        """Train transition matrix from a sequence.

        Args:
            sequence: List of MIDI notes or Note objects
            order: Override config order for training
        """
        if order is None:
            order = self.config.order

        # Convert Notes to MIDI numbers
        notes = [n.midi if isinstance(n, Note) else n for n in sequence]

        if len(notes) <= order:
            return

        # Build transition counts
        counts: Dict[tuple, Dict[int, int]] = {}

        for i in range(len(notes) - order):
            state = tuple(notes[i:i + order])
            next_note = notes[i + order]

            if state not in counts:
                counts[state] = {}
            if next_note not in counts[state]:
                counts[state][next_note] = 0
            counts[state][next_note] += 1

        # Convert counts to probabilities
        self.transitions = {}
        for state, next_counts in counts.items():
            total = sum(next_counts.values())
            # For first-order, use single int key
            key = state[0] if order == 1 else state
            self.transitions[key] = {note: count / total for note, count in next_counts.items()}  # type: ignore[index]

    def set_transitions(self, transitions: Dict[int, Dict[int, float]]) -> None:
        """Set transition probabilities manually.

        Args:
            transitions: {from_note: {to_note: probability}}
        """
        # Normalize probabilities
        self.transitions = {}
        for from_note, to_probs in transitions.items():
            total = sum(to_probs.values())
            self.transitions[from_note] = {n: p / total for n, p in to_probs.items()}

    def _get_next_note(self, current: int) -> Optional[int]:
        """Get next note based on transition probabilities."""
        if current not in self.transitions:
            # Fall back to random from all known states
            if self.transitions:
                current = self._rng.choice(list(self.transitions.keys()))
            else:
                return None

        probs = self.transitions[current]
        if not probs:
            return None

        notes = list(probs.keys())
        weights = list(probs.values())
        next_note = self._rng.choices(notes, weights)[0]

        # Constrain to scale if defined
        if self.scale and not self.scale.contains(Note.from_midi(next_note)):
            # Find nearest scale note
            scale_notes = self.scale.get_midi_notes(octaves=3)
            next_note = min(scale_notes, key=lambda n: abs(n - next_note))

        return next_note

    def generate(  # type: ignore[override]
        self,
        num_notes: int = 16,
        start_note: Optional[Union[int, Note]] = None,
        start_time: float = 0.0,
    ) -> List[MIDIEvent]:
        """Generate melody using Markov chain.

        Args:
            num_notes: Number of notes to generate
            start_note: Starting note (random if None)
            start_time: Start time in seconds

        Returns:
            List of MIDIEvent objects
        """
        if not self.transitions:
            return []

        events = []
        beat_duration = 60.0 / self.config.tempo
        note_duration = self.config.note_duration * beat_duration * self.config.gate

        # Determine start note
        current: Optional[int]
        if start_note is None:
            current = self._rng.choice(list(self.transitions.keys()))
        else:
            current = start_note.midi if isinstance(start_note, Note) else start_note

        current_time = start_time

        for step in range(num_notes):
            if current is None:
                break

            time = self._apply_swing(current_time, step)
            velocity = self.config.velocity
            time, velocity = self._apply_humanize(time, velocity)

            events.extend(self._create_note_events(time, current, velocity, note_duration))

            current = self._get_next_note(current)
            current_time += self.config.note_duration * beat_duration

        return sorted(events, key=lambda e: e.time)

    def __repr__(self) -> str:
        return f"MarkovGenerator(states={len(self.transitions)})"


# ============================================================================
# Probabilistic Generator
# ============================================================================


@dataclass
class ProbabilisticConfig(GeneratorConfig):
    """Probabilistic generator configuration.

    Attributes:
        note_duration: Duration of each note in beats
        gate: Gate time as fraction
        rest_probability: Probability of rest (0.0-1.0)
    """
    note_duration: float = 0.5
    gate: float = 0.9
    rest_probability: float = 0.0


class ProbabilisticGenerator(Generator):
    """Weighted random note generator.

    Generates notes based on weighted probabilities, with support
    for scales, custom weights, and rests.

    Example:
        >>> prob = ProbabilisticGenerator(Scale(Note('C', 4), ScaleType.MAJOR))
        >>> events = prob.generate(num_notes=16)
        >>>
        >>> # With custom weights
        >>> prob.set_weights({60: 3, 64: 2, 67: 1})  # Favor root and third
    """

    def __init__(
        self,
        scale: Optional[Scale] = None,
        weights: Optional[Dict[int, float]] = None,
        config: Optional[ProbabilisticConfig] = None,
    ):
        """Initialize probabilistic generator.

        Args:
            scale: Scale to draw notes from
            weights: Custom note weights {midi: weight}
            config: Generator configuration
        """
        super().__init__(config or ProbabilisticConfig())
        self.scale = scale
        self._weights = weights or {}
        self._notes: List[int] = []
        self._probabilities: List[float] = []
        self._update_distributions()

    @property  # type: ignore[override]
    def config(self) -> ProbabilisticConfig:
        return self._config

    @config.setter
    def config(self, value: ProbabilisticConfig):
        self._config = value

    def _update_distributions(self) -> None:
        """Update internal note and probability arrays."""
        if self._weights:
            self._notes = list(self._weights.keys())
            total = sum(self._weights.values())
            self._probabilities = [w / total for w in self._weights.values()]
        elif self.scale:
            self._notes = self.scale.get_midi_notes(octaves=2)
            self._probabilities = [1.0 / len(self._notes)] * len(self._notes)
        else:
            self._notes = list(range(60, 72))  # Default C4-B4
            self._probabilities = [1.0 / 12] * 12

    def set_scale(self, scale: Scale) -> None:
        """Set scale for note selection.

        Args:
            scale: Scale to use
        """
        self.scale = scale
        self._weights = {}
        self._update_distributions()

    def set_weights(self, weights: Dict[int, float]) -> None:
        """Set custom note weights.

        Args:
            weights: {midi_note: weight}
        """
        self._weights = weights
        self._update_distributions()

    def generate(  # type: ignore[override]
        self,
        num_notes: int = 16,
        start_time: float = 0.0,
    ) -> List[MIDIEvent]:
        """Generate notes with weighted random selection.

        Args:
            num_notes: Number of notes to generate
            start_time: Start time in seconds

        Returns:
            List of MIDIEvent objects
        """
        if not self._notes:
            return []

        events = []
        beat_duration = 60.0 / self.config.tempo
        note_duration = self.config.note_duration * beat_duration * self.config.gate
        current_time = start_time

        for step in range(num_notes):
            # Check for rest
            if self._rng.random() < self.config.rest_probability:
                current_time += self.config.note_duration * beat_duration
                continue

            # Select note
            pitch = self._rng.choices(self._notes, self._probabilities)[0]

            time = self._apply_swing(current_time, step)
            velocity = self.config.velocity
            time, velocity = self._apply_humanize(time, velocity)

            events.extend(self._create_note_events(time, pitch, velocity, note_duration))
            current_time += self.config.note_duration * beat_duration

        return sorted(events, key=lambda e: e.time)

    def __repr__(self) -> str:
        return f"ProbabilisticGenerator(notes={len(self._notes)})"


# ============================================================================
# Sequence Generator (Step Sequencer)
# ============================================================================


@dataclass
class SequenceConfig(GeneratorConfig):
    """Sequence generator configuration.

    Attributes:
        step_duration: Duration of each step in beats
        gate: Gate time as fraction
    """
    step_duration: float = 0.25  # 16th notes
    gate: float = 0.8


@dataclass
class Step:
    """Single step in a sequence.

    Attributes:
        pitch: MIDI note number (or None for rest)
        velocity: Note velocity (or None to use default)
        gate: Gate time multiplier (or None to use default)
        probability: Probability of triggering (0.0-1.0)
        slide: If True, glide to next note
    """
    pitch: Optional[int] = None
    velocity: Optional[int] = None
    gate: Optional[float] = None
    probability: float = 1.0
    slide: bool = False


class SequenceGenerator(Generator):
    """Step sequencer pattern generator.

    Classic step sequencer with support for per-step parameters,
    probability, and slides.

    Example:
        >>> seq = SequenceGenerator(steps=16)
        >>> seq.set_step(0, Step(pitch=60, velocity=120))
        >>> seq.set_step(4, Step(pitch=64))
        >>> seq.set_step(8, Step(pitch=67))
        >>> events = seq.generate(cycles=4)
    """

    def __init__(
        self,
        steps: int = 16,
        config: Optional[SequenceConfig] = None,
    ):
        """Initialize sequence generator.

        Args:
            steps: Number of steps in the sequence
            config: Generator configuration
        """
        super().__init__(config or SequenceConfig())
        self.num_steps = steps
        self._steps: List[Optional[Step]] = [None] * steps

    @property  # type: ignore[override]
    def config(self) -> SequenceConfig:
        return self._config

    @config.setter
    def config(self, value: SequenceConfig):
        self._config = value

    def set_step(self, index: int, step: Step) -> None:
        """Set a step in the sequence.

        Args:
            index: Step index (0-based)
            step: Step configuration
        """
        if not 0 <= index < self.num_steps:
            raise ValueError(f"Step index {index} out of range (0-{self.num_steps - 1})")
        self._steps[index] = step

    def clear_step(self, index: int) -> None:
        """Clear a step (make it a rest).

        Args:
            index: Step index
        """
        self._steps[index] = None

    def set_pattern(self, pattern: List[Optional[int]], velocity: int = 100) -> None:
        """Set pattern from list of pitches.

        Args:
            pattern: List of MIDI notes (None for rest)
            velocity: Default velocity
        """
        for i, pitch in enumerate(pattern[:self.num_steps]):
            if pitch is not None:
                self._steps[i] = Step(pitch=pitch, velocity=velocity)
            else:
                self._steps[i] = None

    def generate(  # type: ignore[override]
        self,
        cycles: int = 1,
        start_time: float = 0.0,
    ) -> List[MIDIEvent]:
        """Generate sequence events.

        Args:
            cycles: Number of sequence cycles
            start_time: Start time in seconds

        Returns:
            List of MIDIEvent objects
        """
        events = []
        beat_duration = 60.0 / self.config.tempo
        step_duration = self.config.step_duration * beat_duration
        current_time = start_time

        for cycle in range(cycles):
            for step_idx, step in enumerate(self._steps):
                global_step = step_idx + cycle * self.num_steps

                if step is not None and step.pitch is not None:
                    # Check probability
                    if self._rng.random() > step.probability:
                        current_time += step_duration
                        continue

                    pitch = step.pitch
                    velocity = step.velocity if step.velocity is not None else self.config.velocity
                    gate = step.gate if step.gate is not None else self.config.gate

                    note_duration = step_duration * gate

                    time = self._apply_swing(current_time, global_step)
                    time, velocity = self._apply_humanize(time, velocity)

                    events.extend(self._create_note_events(time, pitch, velocity, note_duration))

                current_time += step_duration

        return sorted(events, key=lambda e: e.time)

    def __repr__(self) -> str:
        active_steps = sum(1 for s in self._steps if s is not None)
        return f"SequenceGenerator(steps={self.num_steps}, active={active_steps})"


# ============================================================================
# Melody Generator
# ============================================================================


@dataclass
class MelodyConfig(GeneratorConfig):
    """Melody generator configuration.

    Attributes:
        note_duration: Base note duration in beats
        gate: Gate time as fraction
        max_jump: Maximum interval jump in semitones
        contour_tendency: Tendency for melodic direction (-1 to 1)
        phrase_length: Target phrase length in notes
        rest_probability: Probability of rest between phrases
    """
    note_duration: float = 0.5
    gate: float = 0.9
    max_jump: int = 7  # Perfect fifth
    contour_tendency: float = 0.0  # 0 = neutral, positive = ascending
    phrase_length: int = 4
    rest_probability: float = 0.2


class MelodyGenerator(Generator):
    """Rule-based melodic generator.

    Generates melodies following music theory rules for voice leading,
    contour, and phrase structure.

    Example:
        >>> scale = Scale(Note('C', 4), ScaleType.MAJOR)
        >>> melody = MelodyGenerator(scale)
        >>> events = melody.generate(num_notes=32)
    """

    def __init__(
        self,
        scale: Scale,
        config: Optional[MelodyConfig] = None,
    ):
        """Initialize melody generator.

        Args:
            scale: Scale to use for melody
            config: Generator configuration
        """
        super().__init__(config or MelodyConfig())
        self.scale = scale
        self._scale_notes: List[int] = []
        self._update_scale()

    @property  # type: ignore[override]
    def config(self) -> MelodyConfig:
        return self._config

    @config.setter
    def config(self, value: MelodyConfig):
        self._config = value

    def _update_scale(self) -> None:
        """Update internal scale note list."""
        self._scale_notes = self.scale.get_midi_notes(octaves=3)

    def _get_next_note(self, current: int, direction_hint: int = 0) -> int:
        """Get next melodic note following rules.

        Args:
            current: Current MIDI note
            direction_hint: -1 = prefer down, 0 = neutral, 1 = prefer up

        Returns:
            Next MIDI note
        """
        # Find current position in scale
        try:
            current_idx = self._scale_notes.index(current)
        except ValueError:
            # Find nearest scale note
            current_idx = min(
                range(len(self._scale_notes)),
                key=lambda i: abs(self._scale_notes[i] - current)
            )

        # Calculate step range
        max_step = self.config.max_jump // 2 + 1  # Convert semitones to approximate scale steps

        # Apply contour tendency
        bias = self.config.contour_tendency + direction_hint * 0.3
        if bias > 0:
            weights = [1.0 + bias * i for i in range(-max_step, max_step + 1)]
        elif bias < 0:
            weights = [1.0 - bias * i for i in range(max_step, -max_step - 1, -1)]
        else:
            weights = [1.0] * (max_step * 2 + 1)

        # Get valid indices
        candidates = []
        candidate_weights = []
        for step, weight in zip(range(-max_step, max_step + 1), weights):
            new_idx = current_idx + step
            if 0 <= new_idx < len(self._scale_notes):
                candidates.append(new_idx)
                candidate_weights.append(weight)

        if not candidates:
            return current

        # Select with weighted random
        chosen_idx = self._rng.choices(candidates, candidate_weights)[0]
        return self._scale_notes[chosen_idx]

    def generate(  # type: ignore[override]
        self,
        num_notes: int = 16,
        start_note: Optional[Union[int, Note]] = None,
        start_time: float = 0.0,
    ) -> List[MIDIEvent]:
        """Generate melody.

        Args:
            num_notes: Number of notes to generate
            start_note: Starting note (scale root if None)
            start_time: Start time in seconds

        Returns:
            List of MIDIEvent objects
        """
        events = []
        beat_duration = 60.0 / self.config.tempo
        note_duration = self.config.note_duration * beat_duration * self.config.gate

        # Determine start note
        if start_note is None:
            current = self.scale.root.midi
        else:
            current = start_note.midi if isinstance(start_note, Note) else start_note

        current_time = start_time
        phrase_position = 0
        direction_hint = 0

        for step in range(num_notes):
            # Check for phrase boundary rest
            if phrase_position >= self.config.phrase_length:
                if self._rng.random() < self.config.rest_probability:
                    current_time += note_duration / self.config.gate  # Add rest
                    phrase_position = 0
                    # Reset direction for new phrase
                    direction_hint = self._rng.choice([-1, 0, 1])
                    continue
                phrase_position = 0

            time = self._apply_swing(current_time, step)
            velocity = self.config.velocity
            time, velocity = self._apply_humanize(time, velocity)

            events.extend(self._create_note_events(time, current, velocity, note_duration))

            # Get next note
            current = self._get_next_note(current, direction_hint)
            current_time += note_duration / self.config.gate
            phrase_position += 1

        return sorted(events, key=lambda e: e.time)

    def __repr__(self) -> str:
        return f"MelodyGenerator({self.scale})"


# ============================================================================
# Polyrhythm Generator
# ============================================================================


@dataclass
class PolyrhythmConfig(GeneratorConfig):
    """Polyrhythm generator configuration.

    Attributes:
        note_duration: Base note duration in beats
        gate: Gate time as fraction
    """
    note_duration: float = 0.25
    gate: float = 0.8


@dataclass
class RhythmLayer:
    """Single layer in a polyrhythm.

    Attributes:
        pulses: Number of pulses
        pitch: MIDI note number
        velocity: Note velocity
        offset: Time offset in beats
    """
    pulses: int
    pitch: int = 60
    velocity: int = 100
    offset: float = 0.0


class PolyrhythmGenerator(Generator):
    """Polyrhythmic pattern generator.

    Creates layered rhythms with different pulse counts,
    supporting complex polymetric patterns.

    Example:
        >>> poly = PolyrhythmGenerator(cycle_beats=4)
        >>> poly.add_layer(RhythmLayer(pulses=3, pitch=36))  # 3 against
        >>> poly.add_layer(RhythmLayer(pulses=4, pitch=38))  # 4
        >>> events = poly.generate(cycles=4)
    """

    def __init__(
        self,
        cycle_beats: float = 4.0,
        config: Optional[PolyrhythmConfig] = None,
    ):
        """Initialize polyrhythm generator.

        Args:
            cycle_beats: Length of one cycle in beats
            config: Generator configuration
        """
        super().__init__(config or PolyrhythmConfig())
        self.cycle_beats = cycle_beats
        self.layers: List[RhythmLayer] = []

    @property  # type: ignore[override]
    def config(self) -> PolyrhythmConfig:
        return self._config

    @config.setter
    def config(self, value: PolyrhythmConfig):
        self._config = value

    def add_layer(self, layer: RhythmLayer) -> None:
        """Add a rhythm layer.

        Args:
            layer: Rhythm layer configuration
        """
        self.layers.append(layer)

    def clear_layers(self) -> None:
        """Clear all layers."""
        self.layers.clear()

    def generate(  # type: ignore[override]
        self,
        cycles: int = 1,
        start_time: float = 0.0,
    ) -> List[MIDIEvent]:
        """Generate polyrhythmic events.

        Args:
            cycles: Number of cycles
            start_time: Start time in seconds

        Returns:
            List of MIDIEvent objects
        """
        events = []
        beat_duration = 60.0 / self.config.tempo
        cycle_duration = self.cycle_beats * beat_duration
        note_duration = self.config.note_duration * beat_duration * self.config.gate

        for cycle in range(cycles):
            cycle_start = start_time + cycle * cycle_duration

            for layer in self.layers:
                pulse_interval = cycle_duration / layer.pulses

                for pulse in range(layer.pulses):
                    time = cycle_start + layer.offset * beat_duration + pulse * pulse_interval
                    step = cycle * max(lyr.pulses for lyr in self.layers) + pulse

                    time = self._apply_swing(time, step)
                    velocity = layer.velocity
                    time, velocity = self._apply_humanize(time, velocity)

                    events.extend(self._create_note_events(
                        time, layer.pitch, velocity, note_duration
                    ))

        return sorted(events, key=lambda e: e.time)

    def __repr__(self) -> str:
        pulses = [lyr.pulses for lyr in self.layers]
        return f"PolyrhythmGenerator({':'.join(map(str, pulses))})"


# ============================================================================
# Bit Shift Register Generator
# ============================================================================


@dataclass
class BitShiftRegisterConfig(GeneratorConfig):
    """Bit shift register generator configuration.

    Attributes:
        step_duration: Duration of each step in beats
        gate: Gate time as fraction (0.0-1.0)
        velocity_mode: How to determine velocity - 'fixed', 'random', 'pattern'
        velocity_min: Minimum velocity for random mode
        velocity_max: Maximum velocity for random mode
        velocity_pattern: List of velocities to cycle through in pattern mode
        duration_mode: How to determine note duration - 'fixed', 'random', 'pattern'
        duration_min: Minimum duration multiplier for random mode
        duration_max: Maximum duration multiplier for random mode
        duration_pattern: List of duration multipliers for pattern mode
    """
    step_duration: float = 0.25  # 16th notes
    gate: float = 0.8
    velocity_mode: str = 'fixed'
    velocity_min: int = 64
    velocity_max: int = 127
    velocity_pattern: Optional[List[int]] = None
    duration_mode: str = 'fixed'
    duration_min: float = 0.5
    duration_max: float = 1.0
    duration_pattern: Optional[List[float]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.velocity_mode not in ('fixed', 'random', 'pattern'):
            raise ValueError(f"velocity_mode must be 'fixed', 'random', or 'pattern', got {self.velocity_mode}")
        if self.duration_mode not in ('fixed', 'random', 'pattern'):
            raise ValueError(f"duration_mode must be 'fixed', 'random', or 'pattern', got {self.duration_mode}")
        if not 0 <= self.velocity_min <= 127:
            raise ValueError(f"velocity_min must be 0-127, got {self.velocity_min}")
        if not 0 <= self.velocity_max <= 127:
            raise ValueError(f"velocity_max must be 0-127, got {self.velocity_max}")


class BitShiftRegister:
    """Core bit shift register logic.

    A shift register that stores binary gate states. On each clock pulse,
    bits shift right, the input bit enters from the left, and the rightmost
    bit is output (shifted out).

    This is the fundamental building block used by BitShiftRegisterGenerator
    for creating rhythmic patterns with MIDI output.

    Example:
        >>> sr = BitShiftRegister(size=4)
        >>> sr.clock(1)  # Input 1, output 0 (initial state)
        0
        >>> sr.clock(0)  # Input 0, output 0
        0
        >>> sr.clock(1)  # Input 1, output 0
        0
        >>> sr.clock(1)  # Input 1, output 1 (first 1 finally exits)
        1
        >>> print(sr)  # Current state: [1, 1, 0, 1]
        1101
    """

    def __init__(self, size: int, initial_state: Optional[List[int]] = None):
        """Initialize bit shift register.

        Args:
            size: Number of bits in the register
            initial_state: Optional initial bit pattern (list of 0s and 1s)
        """
        if size < 1:
            raise ValueError(f"Size must be at least 1, got {size}")

        self.size = size

        if initial_state is not None:
            if len(initial_state) != size:
                raise ValueError(f"Initial state length {len(initial_state)} != size {size}")
            self.bits = [1 if b else 0 for b in initial_state]
        else:
            self.bits = [0] * size

    def clock(self, input_bit: int) -> int:
        """Clock the shift register.

        Shifts all bits right by one position:
        - New input_bit enters at index 0 (left)
        - Rightmost bit (index -1) is shifted out and returned

        Args:
            input_bit: Bit to shift in (0 or 1, truthy values become 1)

        Returns:
            The bit that was shifted out (gate output for this step)
        """
        input_bit = 1 if input_bit else 0
        shifted_out = self.bits[-1]
        self.bits = [input_bit] + self.bits[:-1]
        return shifted_out

    def peek(self, index: int = -1) -> int:
        """Peek at a bit without clocking.

        Args:
            index: Bit index (negative indices supported, -1 = output bit)

        Returns:
            Bit value at index
        """
        return self.bits[index]

    def reset(self, pattern: Optional[List[int]] = None) -> None:
        """Reset the register.

        Args:
            pattern: Optional new pattern (all zeros if None)
        """
        if pattern is not None:
            if len(pattern) != self.size:
                raise ValueError(f"Pattern length {len(pattern)} != size {self.size}")
            self.bits = [1 if b else 0 for b in pattern]
        else:
            self.bits = [0] * self.size

    def get_state(self) -> List[int]:
        """Get current register state.

        Returns:
            Copy of current bit pattern
        """
        return self.bits.copy()

    def set_state(self, state: List[int]) -> None:
        """Set register state directly.

        Args:
            state: New bit pattern
        """
        if len(state) != self.size:
            raise ValueError(f"State length {len(state)} != size {self.size}")
        self.bits = [1 if b else 0 for b in state]

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return "".join(str(b) for b in self.bits)

    def __str__(self) -> str:
        return "".join(str(b) for b in self.bits)


class BitShiftRegisterGenerator(Generator):
    """MIDI note generator using bit shift register logic.

    Creates rhythmic patterns by feeding a gate input sequence through a
    shift register. The output bit determines whether a note plays (1) or
    rests (0). Notes cycle through a configurable pitch sequence.

    The shift register introduces a delay equal to its size, creating
    interesting rhythmic variations and canonic effects when combined
    with other generators.

    Features:
    - Variable velocity (fixed, random, or pattern-based)
    - Variable duration (fixed, random, or pattern-based)
    - Configurable register size for different delay amounts
    - Support for custom gate input patterns or live input
    - Swing and humanization support

    Example:
        >>> # Create a 4-bit shift register with C major triad notes
        >>> sr_gen = BitShiftRegisterGenerator(
        ...     size=4,
        ...     pitches=[60, 64, 67],  # C, E, G
        ... )
        >>>
        >>> # Generate with a gate pattern
        >>> gate_pattern = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0]
        >>> events = sr_gen.generate(gate_inputs=gate_pattern)
        >>>
        >>> # With variable velocity and duration
        >>> config = BitShiftRegisterConfig(
        ...     tempo=120,
        ...     velocity_mode='random',
        ...     velocity_min=80,
        ...     velocity_max=120,
        ...     duration_mode='pattern',
        ...     duration_pattern=[1.0, 0.5, 0.75, 0.5],
        ... )
        >>> sr_gen = BitShiftRegisterGenerator(
        ...     size=4,
        ...     pitches=[36, 38, 42, 46],  # Drum sounds
        ...     config=config,
        ... )
    """

    def __init__(
        self,
        size: int = 4,
        pitches: Optional[List[Union[int, Note]]] = None,
        initial_state: Optional[List[int]] = None,
        config: Optional[BitShiftRegisterConfig] = None,
    ):
        """Initialize bit shift register generator.

        Args:
            size: Number of bits in the shift register
            pitches: List of MIDI pitches to cycle through (default: C4, D4, E4, G4)
            initial_state: Optional initial register state
            config: Generator configuration
        """
        super().__init__(config or BitShiftRegisterConfig())
        self._register = BitShiftRegister(size, initial_state)

        # Convert Notes to MIDI numbers
        if pitches is None:
            self._pitches = [60, 62, 64, 67]  # C4, D4, E4, G4
        else:
            self._pitches = [
                p.midi if isinstance(p, Note) else p
                for p in pitches
            ]

        if not self._pitches:
            raise ValueError("Must provide at least one pitch")

        self._step_counter = 0

    @property  # type: ignore[override]
    def config(self) -> BitShiftRegisterConfig:
        return self._config

    @config.setter
    def config(self, value: BitShiftRegisterConfig):
        self._config = value

    @property
    def register(self) -> BitShiftRegister:
        """Access the underlying shift register."""
        return self._register

    @property
    def pitches(self) -> List[int]:
        """Get current pitch sequence."""
        return self._pitches.copy()

    def set_pitches(self, pitches: List[Union[int, Note]]) -> None:
        """Set new pitch sequence.

        Args:
            pitches: List of MIDI pitches or Notes
        """
        self._pitches = [
            p.midi if isinstance(p, Note) else p
            for p in pitches
        ]
        if not self._pitches:
            raise ValueError("Must provide at least one pitch")

    def reset(self, pattern: Optional[List[int]] = None) -> None:
        """Reset the shift register and step counter.

        Args:
            pattern: Optional new register pattern
        """
        self._register.reset(pattern)
        self._step_counter = 0

    def _get_velocity(self, step: int) -> int:
        """Get velocity for a step based on configuration."""
        mode = self.config.velocity_mode

        if mode == 'fixed':
            return self.config.velocity

        elif mode == 'random':
            return self._rng.randint(
                self.config.velocity_min,
                self.config.velocity_max
            )

        elif mode == 'pattern':
            if self.config.velocity_pattern:
                idx = step % len(self.config.velocity_pattern)
                return self.config.velocity_pattern[idx]
            return self.config.velocity

        return self.config.velocity

    def _get_duration_multiplier(self, step: int) -> float:
        """Get duration multiplier for a step based on configuration."""
        mode = self.config.duration_mode

        if mode == 'fixed':
            return 1.0

        elif mode == 'random':
            return self._rng.uniform(
                self.config.duration_min,
                self.config.duration_max
            )

        elif mode == 'pattern':
            if self.config.duration_pattern:
                idx = step % len(self.config.duration_pattern)
                return self.config.duration_pattern[idx]
            return 1.0

        return 1.0

    def clock_step(
        self,
        input_bit: int,
        current_time: float,
    ) -> Tuple[Optional[List[MIDIEvent]], int]:
        """Process a single clock step.

        Useful for real-time/live processing where you feed bits one at a time.

        Args:
            input_bit: Gate input (0 or 1)
            current_time: Current time in seconds

        Returns:
            Tuple of (events or None if rest, output gate value)
        """
        output_gate = self._register.clock(input_bit)

        events = None
        if output_gate == 1:
            # Determine pitch from step counter
            pitch_idx = self._step_counter % len(self._pitches)
            pitch = self._pitches[pitch_idx]

            # Get velocity and duration
            velocity = self._get_velocity(self._step_counter)
            duration_mult = self._get_duration_multiplier(self._step_counter)

            beat_duration = 60.0 / self.config.tempo
            base_duration = self.config.step_duration * beat_duration * self.config.gate
            note_duration = base_duration * duration_mult

            # Apply swing and humanization
            time = self._apply_swing(current_time, self._step_counter)
            time, velocity = self._apply_humanize(time, velocity)

            events = self._create_note_events(time, pitch, velocity, note_duration)

        self._step_counter += 1
        return events, output_gate

    def generate(  # type: ignore[override]
        self,
        gate_inputs: Optional[List[int]] = None,
        num_steps: Optional[int] = None,
        start_time: float = 0.0,
        gate_probability: float = 0.5,
    ) -> List[MIDIEvent]:
        """Generate MIDI events from gate input sequence.

        Args:
            gate_inputs: List of gate values (0 or 1). If None, random gates
                        are generated based on gate_probability.
            num_steps: Number of steps to generate (required if gate_inputs is None)
            start_time: Start time in seconds
            gate_probability: Probability of gate=1 when generating random gates

        Returns:
            List of MIDIEvent objects
        """
        # Determine gate sequence
        if gate_inputs is not None:
            gates = gate_inputs
        elif num_steps is not None:
            gates = [
                1 if self._rng.random() < gate_probability else 0
                for _ in range(num_steps)
            ]
        else:
            raise ValueError("Must provide either gate_inputs or num_steps")

        events = []
        beat_duration = 60.0 / self.config.tempo
        step_duration = self.config.step_duration * beat_duration
        current_time = start_time

        for gate_in in gates:
            step_events, _ = self.clock_step(gate_in, current_time)
            if step_events:
                events.extend(step_events)
            current_time += step_duration

        return sorted(events, key=lambda e: e.time)

    def generate_with_trace(
        self,
        gate_inputs: List[int],
        start_time: float = 0.0,
    ) -> Tuple[List[MIDIEvent], List[Dict[str, Any]]]:
        """Generate events with detailed step-by-step trace.

        Useful for debugging, visualization, or educational purposes.

        Args:
            gate_inputs: List of gate values (0 or 1)
            start_time: Start time in seconds

        Returns:
            Tuple of (events, trace) where trace is a list of dicts with:
                - step: Step number
                - input_gate: Input gate value
                - register_state: Register state after clock
                - output_gate: Output gate value
                - pitch: Pitch played (or None for rest)
                - velocity: Velocity (or None for rest)
                - action: 'play' or 'rest'
        """
        events = []
        trace = []
        beat_duration = 60.0 / self.config.tempo
        step_duration = self.config.step_duration * beat_duration
        current_time = start_time

        for step, gate_in in enumerate(gate_inputs):
            step_events, output_gate = self.clock_step(gate_in, current_time)

            trace_entry = {
                'step': step,
                'input_gate': gate_in,
                'register_state': str(self._register),
                'output_gate': output_gate,
                'pitch': None,
                'velocity': None,
                'action': 'rest',
            }

            if step_events:
                events.extend(step_events)
                # Extract pitch and velocity from note on event
                note_on = next((e for e in step_events if e.is_note_on), None)
                if note_on:
                    trace_entry['pitch'] = note_on.data1
                    trace_entry['velocity'] = note_on.data2
                    trace_entry['action'] = 'play'

            trace.append(trace_entry)
            current_time += step_duration

        return sorted(events, key=lambda e: e.time), trace

    def __repr__(self) -> str:
        return f"BitShiftRegisterGenerator(size={self._register.size}, pitches={len(self._pitches)})"


# ============================================================================
# Utility Functions
# ============================================================================


def create_arp_from_progression(
    progression: List[Chord],
    pattern: ArpPattern = ArpPattern.UP,
    beats_per_chord: float = 4.0,
    config: Optional[ArpConfig] = None,
) -> List[MIDIEvent]:
    """Create arpeggiated sequence from chord progression.

    Args:
        progression: List of chords
        pattern: Arpeggio pattern
        beats_per_chord: Beats per chord
        config: Arpeggiator config

    Returns:
        List of MIDIEvent objects
    """
    config = config or ArpConfig()
    arp = Arpeggiator(pattern=pattern, config=config)

    events = []
    beat_duration = 60.0 / config.tempo
    current_time = 0.0

    for chord in progression:
        arp.set_chord(chord)
        chord_events = arp.generate(duration=beats_per_chord * beat_duration, start_time=current_time)
        events.extend(chord_events)
        current_time += beats_per_chord * beat_duration

    return sorted(events, key=lambda e: e.time)


def combine_generators(
    generators: List[Tuple[Generator, Dict[str, Any]]],
    start_time: float = 0.0,
) -> List[MIDIEvent]:
    """Combine output from multiple generators.

    Args:
        generators: List of (generator, kwargs) tuples
        start_time: Start time offset

    Returns:
        Combined list of MIDIEvent objects
    """
    all_events = []

    for gen, kwargs in generators:
        kwargs['start_time'] = kwargs.get('start_time', 0.0) + start_time
        events = gen.generate(**kwargs)
        all_events.extend(events)

    return sorted(all_events, key=lambda e: e.time)
