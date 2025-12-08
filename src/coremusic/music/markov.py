#!/usr/bin/env python3
"""Markov chain analysis and generation for MIDI files.

This module provides tools for analyzing MIDI files to extract Markov chains
representing note transition patterns, and for generating new MIDI variations
based on those chains.

Key Features:
- Configurable Markov chain order (1st order, 2nd order, etc.)
- Multiple modeling variants: pitch-only, pitch+duration, pitch+duration+velocity
- Separate rhythm (inter-onset interval) modeling
- Track-aware analysis and generation
- Granular node-edge scope editing
- Chain-scope adjustments (temperature, clamping, smoothing, gravity)
- JSON serialization for saving/loading trained chains

Example:
    >>> from coremusic.music.markov import MIDIMarkovAnalyzer, MIDIMarkovGenerator
    >>>
    >>> # Analyze a MIDI file
    >>> analyzer = MIDIMarkovAnalyzer(order=2)
    >>> chain = analyzer.analyze_file("song.mid")
    >>>
    >>> # Generate a variation
    >>> generator = MIDIMarkovGenerator(chain)
    >>> sequence = generator.generate(num_notes=64)
    >>> sequence.save("variation.mid")
    >>>
    >>> # Edit transitions
    >>> chain.set_transition_probability(60, 67, 0.8)  # Favor C4 -> G4
    >>> chain.remove_transition(60, 61)  # Remove C4 -> C#4
    >>>
    >>> # Apply chain-scope adjustments
    >>> chain.set_temperature(1.5)  # More random
    >>> chain.set_note_range(48, 84)  # Clamp to range
    >>> chain.set_gravity(60, 0.2)  # Bias toward C4
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from ..midi.utilities import MIDISequence, MIDITrack, MIDIEvent


# ============================================================================
# Enums and Types
# ============================================================================


class ModelingMode(Enum):
    """What properties to model in the Markov chain."""
    PITCH_ONLY = auto()           # Just pitch, constant duration/velocity
    PITCH_DURATION = auto()       # Pitch and duration, constant velocity
    PITCH_DURATION_VELOCITY = auto()  # Full note modeling


class RhythmMode(Enum):
    """How to handle rhythm (inter-onset intervals)."""
    CONSTANT = auto()         # Fixed rhythm
    MARKOV = auto()           # Separate Markov chain for IOI
    FROM_GENERATOR = auto()   # Use external generator (e.g., Euclidean)


# State type for Markov chains - tuple of values for higher-order chains
StateKey = Union[int, Tuple[int, ...]]


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class NoteData:
    """Represents a note with all its properties.

    Attributes:
        pitch: MIDI pitch (0-127)
        duration: Note duration in beats
        velocity: Note velocity (0-127)
        time: Onset time in beats (for context)
    """
    pitch: int
    duration: float
    velocity: int
    time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pitch': self.pitch,
            'duration': self.duration,
            'velocity': self.velocity,
            'time': self.time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NoteData':
        """Create from dictionary."""
        return cls(
            pitch=data['pitch'],
            duration=data['duration'],
            velocity=data['velocity'],
            time=data.get('time', 0.0),
        )


@dataclass
class TransitionEdge:
    """Represents a transition from one state to another.

    Attributes:
        from_state: Source state (pitch or tuple of pitches for higher order)
        to_state: Target state
        count: Number of times this transition occurred
        probability: Transition probability (computed from count)
        metadata: Optional additional data (duration, velocity distributions)
    """
    from_state: StateKey
    to_state: int
    count: int = 1
    probability: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'from_state': list(self.from_state) if isinstance(self.from_state, tuple) else self.from_state,
            'to_state': self.to_state,
            'count': self.count,
            'probability': self.probability,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransitionEdge':
        """Create from dictionary."""
        from_state = data['from_state']
        if isinstance(from_state, list):
            from_state = tuple(from_state)
        return cls(
            from_state=from_state,
            to_state=data['to_state'],
            count=data.get('count', 1),
            probability=data.get('probability', 0.0),
            metadata=data.get('metadata', {}),
        )


@dataclass
class ChainConfig:
    """Configuration for Markov chain behavior.

    Attributes:
        order: Chain order (1 = first-order, 2 = second-order, etc.)
        modeling_mode: What note properties to model
        rhythm_mode: How to handle rhythm
        temperature: Sampling temperature (1.0 = normal, >1 = more random, <1 = more deterministic)
        note_min: Minimum allowed note (for clamping)
        note_max: Maximum allowed note (for clamping)
        gravity_notes: Notes to bias toward {pitch: weight}
        gravity_strength: How strongly to apply gravity (0.0-1.0)
        smoothing_alpha: Laplace smoothing parameter (0 = no smoothing)
        default_duration: Default note duration in beats (for PITCH_ONLY mode)
        default_velocity: Default velocity (for modes without velocity)
        seed: Random seed for reproducibility
    """
    order: int = 1
    modeling_mode: ModelingMode = ModelingMode.PITCH_ONLY
    rhythm_mode: RhythmMode = RhythmMode.CONSTANT
    temperature: float = 1.0
    note_min: int = 0
    note_max: int = 127
    gravity_notes: Dict[int, float] = field(default_factory=dict)
    gravity_strength: float = 0.0
    smoothing_alpha: float = 0.0
    default_duration: float = 0.5
    default_velocity: int = 100
    seed: Optional[int] = None

    def __post_init__(self):
        if self.order < 1:
            raise ValueError(f"Order must be >= 1, got {self.order}")
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be > 0, got {self.temperature}")
        if not 0 <= self.note_min <= 127:
            raise ValueError(f"note_min must be 0-127, got {self.note_min}")
        if not 0 <= self.note_max <= 127:
            raise ValueError(f"note_max must be 0-127, got {self.note_max}")
        if self.note_min > self.note_max:
            raise ValueError(f"note_min ({self.note_min}) > note_max ({self.note_max})")
        if not 0 <= self.gravity_strength <= 1:
            raise ValueError(f"gravity_strength must be 0-1, got {self.gravity_strength}")
        if self.smoothing_alpha < 0:
            raise ValueError(f"smoothing_alpha must be >= 0, got {self.smoothing_alpha}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'order': self.order,
            'modeling_mode': self.modeling_mode.name,
            'rhythm_mode': self.rhythm_mode.name,
            'temperature': self.temperature,
            'note_min': self.note_min,
            'note_max': self.note_max,
            'gravity_notes': {str(k): v for k, v in self.gravity_notes.items()},
            'gravity_strength': self.gravity_strength,
            'smoothing_alpha': self.smoothing_alpha,
            'default_duration': self.default_duration,
            'default_velocity': self.default_velocity,
            'seed': self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChainConfig':
        """Create from dictionary."""
        return cls(
            order=data.get('order', 1),
            modeling_mode=ModelingMode[data.get('modeling_mode', 'PITCH_ONLY')],
            rhythm_mode=RhythmMode[data.get('rhythm_mode', 'CONSTANT')],
            temperature=data.get('temperature', 1.0),
            note_min=data.get('note_min', 0),
            note_max=data.get('note_max', 127),
            gravity_notes={int(k): v for k, v in data.get('gravity_notes', {}).items()},
            gravity_strength=data.get('gravity_strength', 0.0),
            smoothing_alpha=data.get('smoothing_alpha', 0.0),
            default_duration=data.get('default_duration', 0.5),
            default_velocity=data.get('default_velocity', 100),
            seed=data.get('seed'),
        )


# ============================================================================
# Markov Chain Core
# ============================================================================


class MarkovChain:
    """Core Markov chain for note transitions.

    Stores transition probabilities between states (notes or note sequences)
    and provides methods for sampling, editing, and adjusting the chain.

    The chain can be:
    - Trained from note sequences
    - Edited at node-edge level (individual transitions)
    - Adjusted at chain level (temperature, clamping, smoothing, gravity)
    - Serialized to/from JSON

    Example:
        >>> chain = MarkovChain(order=2)
        >>> chain.train([60, 62, 64, 62, 60, 64, 67, 64, 60])
        >>>
        >>> # Sample next note
        >>> next_note = chain.sample(history=[62, 64])
        >>>
        >>> # Edit transitions
        >>> chain.set_transition_probability((62, 64), 67, 0.9)
        >>> chain.remove_transition((60, 62), 64)
        >>>
        >>> # Chain-level adjustments
        >>> chain.set_temperature(1.5)
        >>> chain.apply_smoothing(alpha=0.1)
    """

    def __init__(self, config: Optional[ChainConfig] = None):
        """Initialize Markov chain.

        Args:
            config: Chain configuration
        """
        self.config = config or ChainConfig()
        self._rng = random.Random(self.config.seed)

        # Transitions: {from_state: {to_state: TransitionEdge}}
        self._transitions: Dict[StateKey, Dict[int, TransitionEdge]] = {}

        # Duration and velocity distributions (for PITCH_DURATION and PITCH_DURATION_VELOCITY modes)
        self._duration_chain: Optional['MarkovChain'] = None
        self._velocity_chain: Optional['MarkovChain'] = None

        # Rhythm chain (IOI)
        self._rhythm_chain: Optional['MarkovChain'] = None

        # Track metadata
        self._track_name: str = ""
        self._source_file: str = ""

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train(self, notes: Sequence[Union[int, NoteData]]) -> None:
        """Train the chain from a sequence of notes.

        Args:
            notes: Sequence of MIDI pitches or NoteData objects
        """
        if len(notes) <= self.config.order:
            return

        # Extract pitches
        pitches = [
            n.pitch if isinstance(n, NoteData) else n
            for n in notes
        ]

        # Build transition counts
        self._transitions.clear()

        for i in range(len(pitches) - self.config.order):
            # Build state key
            if self.config.order == 1:
                state: StateKey = pitches[i]
            else:
                state = tuple(pitches[i:i + self.config.order])

            next_pitch = pitches[i + self.config.order]

            # Initialize state if needed
            if state not in self._transitions:
                self._transitions[state] = {}

            # Add or update transition
            if next_pitch not in self._transitions[state]:
                self._transitions[state][next_pitch] = TransitionEdge(
                    from_state=state,
                    to_state=next_pitch,
                    count=0,
                )
            self._transitions[state][next_pitch].count += 1

        # Compute probabilities
        self._recompute_probabilities()

        # Train duration/velocity chains if using extended modes
        if self.config.modeling_mode in (ModelingMode.PITCH_DURATION, ModelingMode.PITCH_DURATION_VELOCITY):
            if all(isinstance(n, NoteData) for n in notes):
                self._train_duration_chain([n for n in notes if isinstance(n, NoteData)])

        if self.config.modeling_mode == ModelingMode.PITCH_DURATION_VELOCITY:
            if all(isinstance(n, NoteData) for n in notes):
                self._train_velocity_chain([n for n in notes if isinstance(n, NoteData)])

    def _train_duration_chain(self, notes: List[NoteData]) -> None:
        """Train duration sub-chain from NoteData sequence."""
        # Quantize durations to discrete values (in 16ths)
        # and create a separate first-order chain
        durations = [self._quantize_duration(n.duration) for n in notes]

        self._duration_chain = MarkovChain(ChainConfig(order=1))
        self._duration_chain._train_generic(durations)

    def _train_velocity_chain(self, notes: List[NoteData]) -> None:
        """Train velocity sub-chain from NoteData sequence."""
        # Quantize velocities to discrete values (groups of 8)
        velocities = [self._quantize_velocity(n.velocity) for n in notes]

        self._velocity_chain = MarkovChain(ChainConfig(order=1))
        self._velocity_chain._train_generic(velocities)

    def _train_generic(self, values: List[int]) -> None:
        """Train on generic integer sequence (internal use)."""
        if len(values) <= self.config.order:
            return

        self._transitions.clear()

        for i in range(len(values) - self.config.order):
            if self.config.order == 1:
                state: StateKey = values[i]
            else:
                state = tuple(values[i:i + self.config.order])

            next_val = values[i + self.config.order]

            if state not in self._transitions:
                self._transitions[state] = {}

            if next_val not in self._transitions[state]:
                self._transitions[state][next_val] = TransitionEdge(
                    from_state=state,
                    to_state=next_val,
                    count=0,
                )
            self._transitions[state][next_val].count += 1

        self._recompute_probabilities()

    def train_rhythm(self, ioi_sequence: List[float]) -> None:
        """Train rhythm chain from inter-onset intervals.

        Args:
            ioi_sequence: List of inter-onset intervals in beats
        """
        # Quantize IOIs to discrete values
        quantized = [self._quantize_rhythm(ioi) for ioi in ioi_sequence]

        self._rhythm_chain = MarkovChain(ChainConfig(order=1))
        self._rhythm_chain._train_generic(quantized)

    def _recompute_probabilities(self) -> None:
        """Recompute all transition probabilities from counts."""
        for state, edges in self._transitions.items():
            total = sum(e.count for e in edges.values())
            if total > 0:
                for edge in edges.values():
                    edge.probability = edge.count / total

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    def sample(
        self,
        history: Optional[List[int]] = None,
        apply_adjustments: bool = True,
    ) -> Optional[int]:
        """Sample next note given history.

        Args:
            history: Previous notes (length should match order)
            apply_adjustments: Whether to apply temperature, gravity, clamping

        Returns:
            Sampled next note, or None if no valid transition
        """
        if not self._transitions:
            return None

        # Build state key from history
        state = self._build_state_key(history)

        # Get transitions for this state
        if state not in self._transitions:
            # Fall back to random state
            state = self._rng.choice(list(self._transitions.keys()))

        edges = self._transitions.get(state, {})
        if not edges:
            return None

        # Get probabilities, applying adjustments if requested
        probs = self._get_adjusted_probabilities(edges, apply_adjustments)

        if not probs:
            return None

        # Sample
        notes = list(probs.keys())
        weights = list(probs.values())

        return self._rng.choices(notes, weights)[0]

    def sample_duration(self, current_pitch: int) -> float:
        """Sample duration for a note.

        Args:
            current_pitch: Current pitch (for context)

        Returns:
            Duration in beats
        """
        if self._duration_chain:
            quantized = self._duration_chain.sample(history=[current_pitch])
            if quantized is not None:
                return self._dequantize_duration(quantized)

        return self.config.default_duration

    def sample_velocity(self, current_pitch: int) -> int:
        """Sample velocity for a note.

        Args:
            current_pitch: Current pitch (for context)

        Returns:
            Velocity (0-127)
        """
        if self._velocity_chain:
            quantized = self._velocity_chain.sample(history=[current_pitch])
            if quantized is not None:
                return self._dequantize_velocity(quantized)

        return self.config.default_velocity

    def sample_rhythm(self, history: Optional[List[float]] = None) -> float:
        """Sample inter-onset interval.

        Args:
            history: Previous IOIs (for context)

        Returns:
            IOI in beats
        """
        if self._rhythm_chain:
            # Convert history to quantized values
            hist_quantized = None
            if history:
                hist_quantized = [self._quantize_rhythm(ioi) for ioi in history]

            quantized = self._rhythm_chain.sample(history=hist_quantized)
            if quantized is not None:
                return self._dequantize_rhythm(quantized)

        return self.config.default_duration

    def _build_state_key(self, history: Optional[List[int]]) -> StateKey:
        """Build state key from history."""
        if history is None or len(history) == 0:
            # Return random state
            return self._rng.choice(list(self._transitions.keys()))

        if self.config.order == 1:
            return history[-1]
        else:
            # Pad history if needed
            if len(history) < self.config.order:
                history = [history[0]] * (self.config.order - len(history)) + history
            return tuple(history[-self.config.order:])

    def _get_adjusted_probabilities(
        self,
        edges: Dict[int, TransitionEdge],
        apply_adjustments: bool,
    ) -> Dict[int, float]:
        """Get adjusted probabilities for sampling.

        Applies temperature, gravity, clamping, and smoothing.
        """
        probs: Dict[int, float] = {}

        for note, edge in edges.items():
            # Start with base probability
            prob = edge.probability

            if apply_adjustments:
                # Apply smoothing
                if self.config.smoothing_alpha > 0:
                    total_states = len(self._get_all_notes())
                    prob = (edge.count + self.config.smoothing_alpha) / (
                        sum(e.count for e in edges.values()) +
                        self.config.smoothing_alpha * total_states
                    )

                # Apply temperature
                if self.config.temperature != 1.0:
                    prob = prob ** (1.0 / self.config.temperature)

                # Apply gravity
                if self.config.gravity_strength > 0 and self.config.gravity_notes:
                    if note in self.config.gravity_notes:
                        gravity_weight = self.config.gravity_notes[note]
                        prob = prob * (1 + self.config.gravity_strength * gravity_weight)

                # Apply clamping
                if not (self.config.note_min <= note <= self.config.note_max):
                    prob = 0.0

            if prob > 0:
                probs[note] = prob

        # Renormalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        return probs

    # -------------------------------------------------------------------------
    # Node-Edge Scope Editing
    # -------------------------------------------------------------------------

    def get_transition_probability(self, from_state: StateKey, to_note: int) -> float:
        """Get probability of transition from state to note.

        Args:
            from_state: Source state
            to_note: Target note

        Returns:
            Transition probability (0 if not exists)
        """
        if from_state in self._transitions:
            if to_note in self._transitions[from_state]:
                return self._transitions[from_state][to_note].probability
        return 0.0

    def set_transition_probability(
        self,
        from_state: StateKey,
        to_note: int,
        probability: float,
    ) -> None:
        """Set probability of a specific transition.

        This will renormalize other transitions from the same state.

        Args:
            from_state: Source state
            to_note: Target note
            probability: New probability (0-1)
        """
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be 0-1, got {probability}")

        # Ensure state exists
        if from_state not in self._transitions:
            self._transitions[from_state] = {}

        # Create or update edge
        if to_note not in self._transitions[from_state]:
            self._transitions[from_state][to_note] = TransitionEdge(
                from_state=from_state,
                to_state=to_note,
                count=1,
            )

        # Set probability and renormalize others
        edges = self._transitions[from_state]
        edges[to_note].probability = probability

        # Scale other probabilities to maintain sum = 1
        remaining = 1.0 - probability
        other_total = sum(e.probability for n, e in edges.items() if n != to_note)

        if other_total > 0:
            scale = remaining / other_total
            for note, edge in edges.items():
                if note != to_note:
                    edge.probability *= scale
        elif remaining > 0 and len(edges) > 1:
            # Distribute remaining equally among others
            per_other = remaining / (len(edges) - 1)
            for note, edge in edges.items():
                if note != to_note:
                    edge.probability = per_other

    def add_transition(
        self,
        from_state: StateKey,
        to_note: int,
        probability: Optional[float] = None,
    ) -> None:
        """Add a new transition.

        Args:
            from_state: Source state
            to_note: Target note
            probability: Initial probability (auto-computed if None)
        """
        if from_state not in self._transitions:
            self._transitions[from_state] = {}

        if to_note in self._transitions[from_state]:
            return  # Already exists

        # Add with count of 1
        self._transitions[from_state][to_note] = TransitionEdge(
            from_state=from_state,
            to_state=to_note,
            count=1,
        )

        if probability is not None:
            self.set_transition_probability(from_state, to_note, probability)
        else:
            self._recompute_probabilities()

    def remove_transition(self, from_state: StateKey, to_note: int) -> bool:
        """Remove a transition.

        Args:
            from_state: Source state
            to_note: Target note

        Returns:
            True if removed, False if didn't exist
        """
        if from_state in self._transitions:
            if to_note in self._transitions[from_state]:
                del self._transitions[from_state][to_note]

                # Renormalize remaining
                edges = self._transitions[from_state]
                if edges:
                    total = sum(e.probability for e in edges.values())
                    if total > 0:
                        for edge in edges.values():
                            edge.probability /= total
                else:
                    # Remove empty state
                    del self._transitions[from_state]

                return True
        return False

    def scale_transition(
        self,
        from_state: StateKey,
        to_note: int,
        factor: float,
    ) -> None:
        """Scale a transition probability by a factor.

        Args:
            from_state: Source state
            to_note: Target note
            factor: Scale factor (e.g., 2.0 = double probability)
        """
        if from_state in self._transitions:
            if to_note in self._transitions[from_state]:
                current = self._transitions[from_state][to_note].probability
                new_prob = min(1.0, current * factor)
                self.set_transition_probability(from_state, to_note, new_prob)

    def get_transitions_from(self, state: StateKey) -> Dict[int, float]:
        """Get all transitions from a state.

        Args:
            state: Source state

        Returns:
            Dictionary of {to_note: probability}
        """
        if state not in self._transitions:
            return {}
        return {note: edge.probability for note, edge in self._transitions[state].items()}

    def get_transitions_to(self, note: int) -> Dict[StateKey, float]:
        """Get all transitions to a note.

        Args:
            note: Target note

        Returns:
            Dictionary of {from_state: probability}
        """
        result: Dict[StateKey, float] = {}
        for state, edges in self._transitions.items():
            if note in edges:
                result[state] = edges[note].probability
        return result

    # -------------------------------------------------------------------------
    # Chain-Scope Adjustments
    # -------------------------------------------------------------------------

    def set_temperature(self, temperature: float) -> None:
        """Set sampling temperature.

        Args:
            temperature: Temperature (1.0 = normal, >1 = more random, <1 = more deterministic)
        """
        if temperature <= 0:
            raise ValueError(f"Temperature must be > 0, got {temperature}")
        self.config.temperature = temperature

    def set_note_range(self, min_note: int, max_note: int) -> None:
        """Set allowed note range (clamping).

        Args:
            min_note: Minimum MIDI note
            max_note: Maximum MIDI note
        """
        if not 0 <= min_note <= 127:
            raise ValueError(f"min_note must be 0-127, got {min_note}")
        if not 0 <= max_note <= 127:
            raise ValueError(f"max_note must be 0-127, got {max_note}")
        if min_note > max_note:
            raise ValueError(f"min_note ({min_note}) > max_note ({max_note})")

        self.config.note_min = min_note
        self.config.note_max = max_note

    def set_gravity(self, note: int, weight: float) -> None:
        """Set gravity toward a specific note.

        Args:
            note: Target note
            weight: Gravity weight (higher = stronger pull)
        """
        self.config.gravity_notes[note] = weight

    def clear_gravity(self) -> None:
        """Clear all gravity settings."""
        self.config.gravity_notes.clear()

    def set_gravity_strength(self, strength: float) -> None:
        """Set overall gravity strength.

        Args:
            strength: Strength (0.0-1.0)
        """
        if not 0 <= strength <= 1:
            raise ValueError(f"Gravity strength must be 0-1, got {strength}")
        self.config.gravity_strength = strength

    def apply_smoothing(self, alpha: float) -> None:
        """Apply Laplace smoothing.

        Args:
            alpha: Smoothing parameter (0 = no smoothing)
        """
        if alpha < 0:
            raise ValueError(f"Smoothing alpha must be >= 0, got {alpha}")
        self.config.smoothing_alpha = alpha

    def sparsify(self, threshold: float) -> int:
        """Remove transitions below a probability threshold.

        Args:
            threshold: Minimum probability to keep

        Returns:
            Number of transitions removed
        """
        removed = 0
        states_to_check = list(self._transitions.keys())

        for state in states_to_check:
            edges = self._transitions[state]
            to_remove = [note for note, edge in edges.items() if edge.probability < threshold]

            for note in to_remove:
                del edges[note]
                removed += 1

            # Renormalize remaining
            if edges:
                total = sum(e.probability for e in edges.values())
                if total > 0:
                    for edge in edges.values():
                        edge.probability /= total
            else:
                del self._transitions[state]

        return removed

    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------

    def get_states(self) -> List[StateKey]:
        """Get all states in the chain."""
        return list(self._transitions.keys())

    def get_state_count(self) -> int:
        """Get number of states."""
        return len(self._transitions)

    def get_transition_count(self) -> int:
        """Get total number of transitions."""
        return sum(len(edges) for edges in self._transitions.values())

    def _get_all_notes(self) -> Set[int]:
        """Get all unique notes in the chain."""
        notes: Set[int] = set()
        for state, edges in self._transitions.items():
            if isinstance(state, int):
                notes.add(state)
            else:
                notes.update(state)
            notes.update(edges.keys())
        return notes

    def get_most_likely_sequence(
        self,
        start_state: StateKey,
        length: int,
    ) -> List[int]:
        """Get most likely sequence from a starting state.

        Args:
            start_state: Starting state
            length: Sequence length

        Returns:
            Most likely note sequence
        """
        sequence: List[int] = []
        if isinstance(start_state, int):
            sequence.append(start_state)
        else:
            sequence.extend(start_state)

        for _ in range(length - len(sequence)):
            state = self._build_state_key(sequence)
            if state not in self._transitions:
                break

            edges = self._transitions[state]
            if not edges:
                break

            # Get most likely
            best_note = max(edges.keys(), key=lambda n: edges[n].probability)
            sequence.append(best_note)

        return sequence

    def get_entropy(self, state: StateKey) -> float:
        """Calculate entropy for a state's transitions.

        Higher entropy = more uncertainty = more randomness.

        Args:
            state: State to calculate entropy for

        Returns:
            Entropy in bits
        """
        if state not in self._transitions:
            return 0.0

        edges = self._transitions[state]
        entropy = 0.0

        for edge in edges.values():
            p = edge.probability
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def get_average_entropy(self) -> float:
        """Calculate average entropy across all states."""
        if not self._transitions:
            return 0.0

        total = sum(self.get_entropy(state) for state in self._transitions)
        return total / len(self._transitions)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert chain to dictionary for serialization."""
        # Convert transitions
        transitions_list = []
        for state, edges in self._transitions.items():
            for note, edge in edges.items():
                transitions_list.append(edge.to_dict())

        result = {
            'version': '1.0',
            'config': self.config.to_dict(),
            'transitions': transitions_list,
            'track_name': self._track_name,
            'source_file': self._source_file,
        }

        # Include sub-chains if present
        if self._duration_chain:
            result['duration_chain'] = self._duration_chain.to_dict()
        if self._velocity_chain:
            result['velocity_chain'] = self._velocity_chain.to_dict()
        if self._rhythm_chain:
            result['rhythm_chain'] = self._rhythm_chain.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarkovChain':
        """Create chain from dictionary."""
        config = ChainConfig.from_dict(data.get('config', {}))
        chain = cls(config)

        # Load transitions
        for edge_data in data.get('transitions', []):
            edge = TransitionEdge.from_dict(edge_data)
            if edge.from_state not in chain._transitions:
                chain._transitions[edge.from_state] = {}
            chain._transitions[edge.from_state][edge.to_state] = edge

        chain._track_name = data.get('track_name', '')
        chain._source_file = data.get('source_file', '')

        # Load sub-chains
        if 'duration_chain' in data:
            chain._duration_chain = cls.from_dict(data['duration_chain'])
        if 'velocity_chain' in data:
            chain._velocity_chain = cls.from_dict(data['velocity_chain'])
        if 'rhythm_chain' in data:
            chain._rhythm_chain = cls.from_dict(data['rhythm_chain'])

        return chain

    def to_json(self, indent: int = 2) -> str:
        """Convert chain to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> 'MarkovChain':
        """Create chain from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, filepath: Union[str, Path]) -> None:
        """Save chain to JSON file.

        Args:
            filepath: Output file path
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'MarkovChain':
        """Load chain from JSON file.

        Args:
            filepath: Input file path

        Returns:
            Loaded MarkovChain
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Chain file not found: {filepath}")
        return cls.from_json(path.read_text())

    # -------------------------------------------------------------------------
    # Quantization Helpers
    # -------------------------------------------------------------------------

    def _quantize_duration(self, duration: float) -> int:
        """Quantize duration to discrete value (16th notes)."""
        # Quantize to 16th notes, max 4 bars = 64 16ths
        sixteenths = max(1, min(64, round(duration * 4)))
        return sixteenths

    def _dequantize_duration(self, quantized: int) -> float:
        """Convert quantized duration back to beats."""
        return quantized / 4.0

    def _quantize_velocity(self, velocity: int) -> int:
        """Quantize velocity to groups of 8."""
        return (velocity // 8) * 8

    def _dequantize_velocity(self, quantized: int) -> int:
        """Convert quantized velocity back, adding small variation."""
        return min(127, quantized + self._rng.randint(0, 7))

    def _quantize_rhythm(self, ioi: float) -> int:
        """Quantize inter-onset interval to discrete value."""
        # Quantize to 32nd notes, max 4 beats = 32 32nds
        thirtyseconds = max(1, min(32, round(ioi * 8)))
        return thirtyseconds

    def _dequantize_rhythm(self, quantized: int) -> float:
        """Convert quantized IOI back to beats."""
        return quantized / 8.0

    def __repr__(self) -> str:
        return (
            f"MarkovChain(order={self.config.order}, "
            f"states={self.get_state_count()}, "
            f"transitions={self.get_transition_count()})"
        )


# ============================================================================
# MIDI Analyzer
# ============================================================================


class MIDIMarkovAnalyzer:
    """Analyzes MIDI files to extract Markov chains.

    Supports single-track analysis with configurable modeling modes.

    Example:
        >>> analyzer = MIDIMarkovAnalyzer(order=2)
        >>> chain = analyzer.analyze_file("song.mid", track_index=0)
        >>>
        >>> # Or analyze all tracks
        >>> chains = analyzer.analyze_all_tracks("song.mid")
    """

    def __init__(
        self,
        order: int = 1,
        modeling_mode: ModelingMode = ModelingMode.PITCH_ONLY,
        rhythm_mode: RhythmMode = RhythmMode.CONSTANT,
        config: Optional[ChainConfig] = None,
    ):
        """Initialize analyzer.

        Args:
            order: Markov chain order
            modeling_mode: What note properties to model
            rhythm_mode: How to handle rhythm
            config: Full chain config (overrides other params)
        """
        if config:
            self.config = config
        else:
            self.config = ChainConfig(
                order=order,
                modeling_mode=modeling_mode,
                rhythm_mode=rhythm_mode,
            )

    def analyze_file(
        self,
        filepath: Union[str, Path],
        track_index: int = 0,
    ) -> MarkovChain:
        """Analyze a single track from a MIDI file.

        Args:
            filepath: MIDI file path
            track_index: Index of track to analyze (0-based)

        Returns:
            Trained MarkovChain
        """
        sequence = MIDISequence.load(str(filepath))
        return self.analyze_track(sequence, track_index, str(filepath))

    def analyze_track(
        self,
        sequence: MIDISequence,
        track_index: int = 0,
        source_file: str = "",
    ) -> MarkovChain:
        """Analyze a track from a MIDISequence.

        Args:
            sequence: MIDI sequence
            track_index: Track index
            source_file: Source filename (for metadata)

        Returns:
            Trained MarkovChain
        """
        if track_index >= len(sequence.tracks):
            raise ValueError(f"Track index {track_index} out of range (have {len(sequence.tracks)} tracks)")

        track = sequence.tracks[track_index]
        return self._analyze_track_impl(track, sequence.tempo, source_file)

    def analyze_all_tracks(
        self,
        filepath: Union[str, Path],
    ) -> List[MarkovChain]:
        """Analyze all tracks in a MIDI file.

        Args:
            filepath: MIDI file path

        Returns:
            List of MarkovChains, one per track
        """
        sequence = MIDISequence.load(str(filepath))
        chains = []

        for i, track in enumerate(sequence.tracks):
            chain = self._analyze_track_impl(track, sequence.tempo, str(filepath))
            chains.append(chain)

        return chains

    def _analyze_track_impl(
        self,
        track: MIDITrack,
        tempo: float,
        source_file: str,
    ) -> MarkovChain:
        """Internal implementation for analyzing a track."""
        # Extract notes
        notes = self._extract_notes(track, tempo)

        if not notes:
            # Return empty chain
            chain = MarkovChain(self.config)
            chain._track_name = track.name
            chain._source_file = source_file
            return chain

        # Create and train chain
        chain = MarkovChain(ChainConfig(
            order=self.config.order,
            modeling_mode=self.config.modeling_mode,
            rhythm_mode=self.config.rhythm_mode,
        ))

        chain._track_name = track.name
        chain._source_file = source_file

        # Train pitch chain
        chain.train(notes)

        # Train rhythm chain if needed
        if self.config.rhythm_mode == RhythmMode.MARKOV and len(notes) > 1:
            iois = self._extract_iois(notes)
            if iois:
                chain.train_rhythm(iois)

        return chain

    def _extract_notes(self, track: MIDITrack, tempo: float) -> List[NoteData]:
        """Extract notes from track as NoteData objects."""
        notes: List[NoteData] = []

        # Match note-ons with note-offs
        note_ons: Dict[int, MIDIEvent] = {}

        beat_duration = 60.0 / tempo

        for event in sorted(track.events, key=lambda e: e.time):
            if event.is_note_on:
                note_ons[event.data1] = event
            elif event.is_note_off:
                if event.data1 in note_ons:
                    on_event = note_ons.pop(event.data1)

                    # Convert times to beats
                    time_beats = on_event.time / beat_duration
                    duration_beats = (event.time - on_event.time) / beat_duration

                    notes.append(NoteData(
                        pitch=event.data1,
                        duration=max(0.001, duration_beats),  # Minimum duration
                        velocity=on_event.data2,
                        time=time_beats,
                    ))

        # Sort by time
        notes.sort(key=lambda n: n.time)

        return notes

    def _extract_iois(self, notes: List[NoteData]) -> List[float]:
        """Extract inter-onset intervals from notes."""
        if len(notes) < 2:
            return []

        iois = []
        for i in range(1, len(notes)):
            ioi = notes[i].time - notes[i - 1].time
            if ioi > 0:
                iois.append(ioi)

        return iois


# ============================================================================
# MIDI Generator
# ============================================================================


class MIDIMarkovGenerator:
    """Generates MIDI sequences from trained Markov chains.

    Example:
        >>> chain = MIDIMarkovAnalyzer().analyze_file("song.mid")
        >>> generator = MIDIMarkovGenerator(chain)
        >>> sequence = generator.generate(num_notes=64, tempo=120.0)
        >>> sequence.save("variation.mid")
    """

    def __init__(
        self,
        chain: MarkovChain,
        rhythm_generator: Optional[Any] = None,
    ):
        """Initialize generator.

        Args:
            chain: Trained MarkovChain
            rhythm_generator: Optional external rhythm generator
                             (must have generate() method returning MIDIEvents)
        """
        self.chain = chain
        self.rhythm_generator = rhythm_generator
        self._rng = random.Random(chain.config.seed)

    def generate(
        self,
        num_notes: int = 32,
        tempo: float = 120.0,
        start_pitch: Optional[int] = None,
        channel: int = 0,
    ) -> MIDISequence:
        """Generate a new MIDI sequence.

        Args:
            num_notes: Number of notes to generate
            tempo: Tempo in BPM
            start_pitch: Starting pitch (random if None)
            channel: MIDI channel

        Returns:
            Generated MIDISequence
        """
        sequence = MIDISequence(tempo=tempo)
        track = sequence.add_track(f"Generated from {self.chain._track_name or 'chain'}")
        track.channel = channel

        if self.chain.get_state_count() == 0:
            return sequence

        # Initialize history
        history: List[int] = []

        if start_pitch is not None:
            history.append(start_pitch)
        else:
            # Pick random starting state
            states = self.chain.get_states()
            if states:
                start_state = self._rng.choice(states)
                if isinstance(start_state, int):
                    history.append(start_state)
                else:
                    history.extend(start_state)

        current_time = 0.0
        beat_duration = 60.0 / tempo
        ioi_history: List[float] = []

        for i in range(num_notes):
            # Sample next pitch
            pitch = self.chain.sample(history)
            if pitch is None:
                break

            # Sample duration
            duration = self._get_duration(pitch)

            # Sample velocity
            velocity = self._get_velocity(pitch)

            # Sample rhythm (IOI for next note)
            ioi = self._get_rhythm(ioi_history)

            # Add note to track
            track.add_note(
                time=current_time * beat_duration,
                note=pitch,
                velocity=velocity,
                duration=duration * beat_duration,
                channel=channel,
            )

            # Update state
            history.append(pitch)
            if len(history) > self.chain.config.order:
                history = history[-self.chain.config.order:]

            ioi_history.append(ioi)
            if len(ioi_history) > 4:
                ioi_history = ioi_history[-4:]

            current_time += ioi

        return sequence

    def generate_to_track(
        self,
        track: MIDITrack,
        num_notes: int = 32,
        start_time: float = 0.0,
        tempo: float = 120.0,
        start_pitch: Optional[int] = None,
        channel: Optional[int] = None,
    ) -> None:
        """Generate notes directly into an existing track.

        Args:
            track: Target MIDITrack
            num_notes: Number of notes to generate
            start_time: Start time in seconds
            tempo: Tempo in BPM
            start_pitch: Starting pitch (random if None)
            channel: MIDI channel (uses track default if None)
        """
        if self.chain.get_state_count() == 0:
            return

        ch = channel if channel is not None else track.channel

        # Initialize history
        history: List[int] = []

        if start_pitch is not None:
            history.append(start_pitch)
        else:
            states = self.chain.get_states()
            if states:
                start_state = self._rng.choice(states)
                if isinstance(start_state, int):
                    history.append(start_state)
                else:
                    history.extend(start_state)

        current_time = start_time
        beat_duration = 60.0 / tempo
        ioi_history: List[float] = []

        for i in range(num_notes):
            pitch = self.chain.sample(history)
            if pitch is None:
                break

            duration = self._get_duration(pitch)
            velocity = self._get_velocity(pitch)
            ioi = self._get_rhythm(ioi_history)

            track.add_note(
                time=current_time,
                note=pitch,
                velocity=velocity,
                duration=duration * beat_duration,
                channel=ch,
            )

            history.append(pitch)
            if len(history) > self.chain.config.order:
                history = history[-self.chain.config.order:]

            ioi_history.append(ioi)
            if len(ioi_history) > 4:
                ioi_history = ioi_history[-4:]

            current_time += ioi * beat_duration

    def _get_duration(self, pitch: int) -> float:
        """Get duration for a note."""
        mode = self.chain.config.modeling_mode

        if mode in (ModelingMode.PITCH_DURATION, ModelingMode.PITCH_DURATION_VELOCITY):
            return self.chain.sample_duration(pitch)
        else:
            return self.chain.config.default_duration

    def _get_velocity(self, pitch: int) -> int:
        """Get velocity for a note."""
        mode = self.chain.config.modeling_mode

        if mode == ModelingMode.PITCH_DURATION_VELOCITY:
            return self.chain.sample_velocity(pitch)
        else:
            return self.chain.config.default_velocity

    def _get_rhythm(self, ioi_history: List[float]) -> float:
        """Get inter-onset interval."""
        mode = self.chain.config.rhythm_mode

        if mode == RhythmMode.MARKOV:
            return self.chain.sample_rhythm(ioi_history)
        elif mode == RhythmMode.FROM_GENERATOR and self.rhythm_generator:
            # This would need integration with external generators
            # For now, fall back to default
            return self.chain.config.default_duration
        else:
            return self.chain.config.default_duration


# ============================================================================
# Utility Functions
# ============================================================================


def analyze_and_generate(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    num_notes: int = 64,
    order: int = 1,
    temperature: float = 1.0,
    track_index: int = 0,
) -> MIDISequence:
    """Convenience function to analyze a MIDI file and generate a variation.

    Args:
        input_file: Source MIDI file
        output_file: Output MIDI file
        num_notes: Number of notes to generate
        order: Markov chain order
        temperature: Sampling temperature
        track_index: Track to analyze

    Returns:
        Generated MIDISequence
    """
    # Analyze
    analyzer = MIDIMarkovAnalyzer(order=order)
    chain = analyzer.analyze_file(input_file, track_index)

    # Adjust
    chain.set_temperature(temperature)

    # Generate
    source_seq = MIDISequence.load(str(input_file))
    generator = MIDIMarkovGenerator(chain)
    output_seq = generator.generate(num_notes=num_notes, tempo=source_seq.tempo)

    # Save
    output_seq.save(str(output_file))

    return output_seq


def merge_chains(
    chains: List[MarkovChain],
    weights: Optional[List[float]] = None,
) -> MarkovChain:
    """Merge multiple chains into one.

    Useful for combining chains from different songs or tracks.

    Args:
        chains: List of chains to merge
        weights: Optional weights for each chain (uniform if None)

    Returns:
        Merged MarkovChain
    """
    if not chains:
        raise ValueError("Must provide at least one chain")

    if weights is None:
        weights = [1.0] * len(chains)

    if len(weights) != len(chains):
        raise ValueError(f"Weights length {len(weights)} != chains length {len(chains)}")

    # Use config from first chain
    merged = MarkovChain(chains[0].config)

    # Collect all transitions with weighted counts
    weighted_transitions: Dict[StateKey, Dict[int, float]] = {}

    for chain, weight in zip(chains, weights):
        for state, edges in chain._transitions.items():
            if state not in weighted_transitions:
                weighted_transitions[state] = {}

            for note, edge in edges.items():
                if note not in weighted_transitions[state]:
                    weighted_transitions[state][note] = 0.0
                weighted_transitions[state][note] += edge.count * weight

    # Convert to transition edges
    for state, note_counts in weighted_transitions.items():
        total = sum(note_counts.values())
        merged._transitions[state] = {}

        for note, count in note_counts.items():
            merged._transitions[state][note] = TransitionEdge(
                from_state=state,
                to_state=note,
                count=int(count),
                probability=count / total if total > 0 else 0.0,
            )

    return merged


def chain_statistics(chain: MarkovChain) -> Dict[str, Any]:
    """Get statistics about a chain.

    Args:
        chain: MarkovChain to analyze

    Returns:
        Dictionary of statistics
    """
    notes = chain._get_all_notes()

    return {
        'order': chain.config.order,
        'modeling_mode': chain.config.modeling_mode.name,
        'state_count': chain.get_state_count(),
        'transition_count': chain.get_transition_count(),
        'unique_notes': len(notes),
        'note_range': (min(notes), max(notes)) if notes else (0, 0),
        'average_entropy': chain.get_average_entropy(),
        'track_name': chain._track_name,
        'source_file': chain._source_file,
    }
