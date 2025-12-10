#!/usr/bin/env python3
"""Bayesian network analysis and generation for MIDI files.

This module provides tools for analyzing MIDI files to build Bayesian networks
representing probabilistic dependencies between note properties, and for generating
new MIDI variations based on those networks.

Key Features:
- Configurable network structure (fixed, learned, or manual)
- Multiple modeling variants: pitch-only to full (pitch+duration+velocity+rhythm)
- Temporal context support (single-step or configurable window)
- Conditional probability tables with smoothing
- Structure learning from data
- JSON serialization for saving/loading trained networks

Example:
    >>> # Analyze a MIDI file
    >>> analyzer = MIDIBayesAnalyzer()
    >>> network = analyzer.analyze_file("song.mid")
    >>>
    >>> # Generate a variation
    >>> generator = MIDIBayesGenerator(network)
    >>> sequence = generator.generate(num_notes=64)
    >>> sequence.save("variation.mid")
    >>>
    >>> # Manual structure configuration
    >>> network.add_edge("prev_pitch", "pitch")
    >>> network.add_edge("pitch", "duration")
    >>> network.add_edge("pitch", "velocity")
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from coremusic.midi.utilities import MIDIEvent, MIDISequence, MIDITrack

# ============================================================================
# Enums and Types
# ============================================================================


class NetworkMode(Enum):
    """What properties to model in the Bayesian network."""
    PITCH_ONLY = auto()           # Just pitch dependencies
    PITCH_DURATION = auto()       # Pitch and duration
    PITCH_DURATION_VELOCITY = auto()  # Pitch, duration, velocity
    FULL = auto()                 # All properties including rhythm (IOI)


class StructureMode(Enum):
    """How to determine network structure."""
    FIXED = auto()        # Predefined structure
    LEARNED = auto()      # Learn structure from data
    MANUAL = auto()       # User-specified structure


# Variable names used in the network
VAR_PITCH = "pitch"
VAR_DURATION = "duration"
VAR_VELOCITY = "velocity"
VAR_IOI = "ioi"  # Inter-onset interval
VAR_PREV_PITCH = "prev_pitch"
VAR_PREV_DURATION = "prev_duration"
VAR_PREV_VELOCITY = "prev_velocity"
VAR_PREV_IOI = "prev_ioi"

# For higher-order models
def var_pitch_lag(lag: int) -> str:
    """Get variable name for pitch at lag n."""
    return f"pitch_lag{lag}" if lag > 0 else VAR_PITCH


def var_duration_lag(lag: int) -> str:
    """Get variable name for duration at lag n."""
    return f"duration_lag{lag}" if lag > 0 else VAR_DURATION


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class NoteObservation:
    """A single note observation with all properties.

    Attributes:
        pitch: MIDI pitch (0-127)
        duration: Note duration in beats
        velocity: Note velocity (0-127)
        ioi: Inter-onset interval from previous note (beats)
        time: Absolute onset time in beats
    """
    pitch: int
    duration: float
    velocity: int
    ioi: float = 0.0
    time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pitch': self.pitch,
            'duration': self.duration,
            'velocity': self.velocity,
            'ioi': self.ioi,
            'time': self.time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NoteObservation':
        """Create from dictionary."""
        return cls(
            pitch=data['pitch'],
            duration=data['duration'],
            velocity=data['velocity'],
            ioi=data.get('ioi', 0.0),
            time=data.get('time', 0.0),
        )


@dataclass
class NetworkConfig:
    """Configuration for Bayesian network.

    Attributes:
        mode: What properties to model
        structure_mode: How to determine structure
        temporal_order: Number of previous time steps to consider (1 = previous note only)
        smoothing_alpha: Laplace smoothing parameter
        pitch_bins: Number of bins for pitch discretization (0 = no binning)
        duration_bins: Number of bins for duration discretization
        velocity_bins: Number of bins for velocity discretization
        ioi_bins: Number of bins for IOI discretization
        default_duration: Default duration in beats
        default_velocity: Default velocity
        default_ioi: Default IOI in beats
        seed: Random seed for reproducibility
    """
    mode: NetworkMode = NetworkMode.PITCH_DURATION_VELOCITY
    structure_mode: StructureMode = StructureMode.FIXED
    temporal_order: int = 1
    smoothing_alpha: float = 1.0
    pitch_bins: int = 0  # 0 = use raw MIDI values
    duration_bins: int = 16  # Quantize to 16th notes
    velocity_bins: int = 8  # 8 velocity levels
    ioi_bins: int = 16
    default_duration: float = 0.5
    default_velocity: int = 100
    default_ioi: float = 0.5
    seed: Optional[int] = None

    def __post_init__(self):
        if self.temporal_order < 1:
            raise ValueError(f"temporal_order must be >= 1, got {self.temporal_order}")
        if self.smoothing_alpha < 0:
            raise ValueError(f"smoothing_alpha must be >= 0, got {self.smoothing_alpha}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mode': self.mode.name,
            'structure_mode': self.structure_mode.name,
            'temporal_order': self.temporal_order,
            'smoothing_alpha': self.smoothing_alpha,
            'pitch_bins': self.pitch_bins,
            'duration_bins': self.duration_bins,
            'velocity_bins': self.velocity_bins,
            'ioi_bins': self.ioi_bins,
            'default_duration': self.default_duration,
            'default_velocity': self.default_velocity,
            'default_ioi': self.default_ioi,
            'seed': self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkConfig':
        """Create from dictionary."""
        return cls(
            mode=NetworkMode[data.get('mode', 'PITCH_DURATION_VELOCITY')],
            structure_mode=StructureMode[data.get('structure_mode', 'FIXED')],
            temporal_order=data.get('temporal_order', 1),
            smoothing_alpha=data.get('smoothing_alpha', 1.0),
            pitch_bins=data.get('pitch_bins', 0),
            duration_bins=data.get('duration_bins', 16),
            velocity_bins=data.get('velocity_bins', 8),
            ioi_bins=data.get('ioi_bins', 16),
            default_duration=data.get('default_duration', 0.5),
            default_velocity=data.get('default_velocity', 100),
            default_ioi=data.get('default_ioi', 0.5),
            seed=data.get('seed'),
        )


# ============================================================================
# Conditional Probability Table
# ============================================================================


class CPT:
    """Conditional Probability Table for a variable.

    Stores P(variable | parents) as a nested dictionary structure.
    Supports Laplace smoothing and sampling.
    """

    def __init__(
        self,
        variable: str,
        parents: Tuple[str, ...] = (),
        smoothing_alpha: float = 1.0,
    ):
        """Initialize CPT.

        Args:
            variable: Name of this variable
            parents: Tuple of parent variable names
            smoothing_alpha: Laplace smoothing parameter
        """
        self.variable = variable
        self.parents = parents
        self.smoothing_alpha = smoothing_alpha

        # Counts: {parent_values: {value: count}}
        # parent_values is a tuple of parent values
        self._counts: Dict[Tuple[int, ...], Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # All observed values for this variable
        self._values: Set[int] = set()

        # Cached probabilities (invalidated on updates)
        self._probs_cache: Optional[Dict[Tuple[int, ...], Dict[int, float]]] = None

    def observe(self, value: int, parent_values: Tuple[int, ...] = ()) -> None:
        """Record an observation.

        Args:
            value: Observed value of this variable
            parent_values: Values of parent variables
        """
        self._counts[parent_values][value] += 1
        self._values.add(value)
        self._probs_cache = None

    def get_probability(self, value: int, parent_values: Tuple[int, ...] = ()) -> float:
        """Get P(value | parent_values).

        Args:
            value: Value to get probability for
            parent_values: Parent variable values

        Returns:
            Probability with Laplace smoothing
        """
        counts = self._counts.get(parent_values, {})
        total = sum(counts.values())
        count = counts.get(value, 0)

        # Laplace smoothing
        num_values = len(self._values) if self._values else 1
        prob = (count + self.smoothing_alpha) / (total + self.smoothing_alpha * num_values)

        return prob

    def get_distribution(self, parent_values: Tuple[int, ...] = ()) -> Dict[int, float]:
        """Get full probability distribution given parents.

        Args:
            parent_values: Parent variable values

        Returns:
            Dictionary {value: probability}
        """
        if not self._values:
            return {}

        probs = {}
        for value in self._values:
            probs[value] = self.get_probability(value, parent_values)

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        return probs

    def sample(
        self,
        parent_values: Tuple[int, ...] = (),
        rng: Optional[random.Random] = None,
    ) -> Optional[int]:
        """Sample a value from the distribution.

        Args:
            parent_values: Parent variable values
            rng: Random number generator

        Returns:
            Sampled value, or None if no values observed
        """
        if not self._values:
            return None

        rng = rng or random.Random()
        probs = self.get_distribution(parent_values)

        if not probs:
            return None

        values = list(probs.keys())
        weights = list(probs.values())

        return rng.choices(values, weights)[0]

    def get_entropy(self, parent_values: Tuple[int, ...] = ()) -> float:
        """Calculate entropy of the distribution given parents.

        Args:
            parent_values: Parent variable values

        Returns:
            Entropy in bits
        """
        probs = self.get_distribution(parent_values)
        entropy = 0.0

        for p in probs.values():
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Convert counts to JSON-serializable format
        counts_list = []
        for parent_vals, value_counts in self._counts.items():
            counts_list.append({
                'parent_values': list(parent_vals),
                'counts': dict(value_counts),
            })

        return {
            'variable': self.variable,
            'parents': list(self.parents),
            'smoothing_alpha': self.smoothing_alpha,
            'counts': counts_list,
            'values': list(self._values),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CPT':
        """Create from dictionary."""
        cpt = cls(
            variable=data['variable'],
            parents=tuple(data.get('parents', [])),
            smoothing_alpha=data.get('smoothing_alpha', 1.0),
        )

        # Restore counts
        for entry in data.get('counts', []):
            parent_vals = tuple(entry['parent_values'])
            for val_str, count in entry['counts'].items():
                val = int(val_str)
                cpt._counts[parent_vals][val] = count

        # Restore values
        cpt._values = set(data.get('values', []))

        return cpt

    def __repr__(self) -> str:
        return f"CPT({self.variable} | {', '.join(self.parents) or 'none'})"


# ============================================================================
# Bayesian Network
# ============================================================================


class BayesianNetwork:
    """Bayesian network for modeling note dependencies.

    The network consists of:
    - Variables: pitch, duration, velocity, ioi (and lagged versions)
    - Edges: directed dependencies between variables
    - CPTs: conditional probability tables for each variable

    Example:
        >>> network = BayesianNetwork()
        >>> network.add_variable("pitch")
        >>> network.add_variable("prev_pitch")
        >>> network.add_variable("duration")
        >>> network.add_edge("prev_pitch", "pitch")
        >>> network.add_edge("pitch", "duration")
        >>>
        >>> # Train from observations
        >>> for obs in observations:
        ...     network.observe({"pitch": obs.pitch, "prev_pitch": prev.pitch, ...})
        >>>
        >>> # Sample
        >>> sample = network.sample(evidence={"prev_pitch": 60})
    """

    def __init__(self, config: Optional[NetworkConfig] = None):
        """Initialize network.

        Args:
            config: Network configuration
        """
        self.config = config or NetworkConfig()
        self._rng = random.Random(self.config.seed)

        # Graph structure
        self._variables: Set[str] = set()
        self._edges: Set[Tuple[str, str]] = set()  # (parent, child)
        self._parents: Dict[str, Set[str]] = defaultdict(set)
        self._children: Dict[str, Set[str]] = defaultdict(set)

        # CPTs for each variable
        self._cpts: Dict[str, CPT] = {}

        # Metadata
        self._track_name: str = ""
        self._source_file: str = ""
        self._num_observations: int = 0

    # -------------------------------------------------------------------------
    # Structure Management
    # -------------------------------------------------------------------------

    def add_variable(self, name: str) -> None:
        """Add a variable to the network.

        Args:
            name: Variable name
        """
        if name not in self._variables:
            self._variables.add(name)
            self._cpts[name] = CPT(name, smoothing_alpha=self.config.smoothing_alpha)

    def remove_variable(self, name: str) -> None:
        """Remove a variable and all its edges.

        Args:
            name: Variable name
        """
        if name in self._variables:
            # Remove edges
            edges_to_remove = [
                (p, c) for (p, c) in self._edges
                if p == name or c == name
            ]
            for edge in edges_to_remove:
                self.remove_edge(*edge)

            self._variables.discard(name)
            self._cpts.pop(name, None)
            self._parents.pop(name, None)
            self._children.pop(name, None)

    def add_edge(self, parent: str, child: str) -> None:
        """Add a directed edge from parent to child.

        Args:
            parent: Parent variable name
            child: Child variable name
        """
        # Ensure variables exist
        self.add_variable(parent)
        self.add_variable(child)

        if (parent, child) not in self._edges:
            # Check for cycles
            if self._would_create_cycle(parent, child):
                raise ValueError(f"Adding edge {parent} -> {child} would create a cycle")

            self._edges.add((parent, child))
            self._parents[child].add(parent)
            self._children[parent].add(child)

            # Update CPT for child
            self._update_cpt_parents(child)

    def remove_edge(self, parent: str, child: str) -> None:
        """Remove an edge.

        Args:
            parent: Parent variable name
            child: Child variable name
        """
        if (parent, child) in self._edges:
            self._edges.discard((parent, child))
            self._parents[child].discard(parent)
            self._children[parent].discard(child)
            self._update_cpt_parents(child)

    def _would_create_cycle(self, parent: str, child: str) -> bool:
        """Check if adding edge would create a cycle."""
        # BFS from child to see if we can reach parent
        visited = set()
        queue = [child]

        while queue:
            node = queue.pop(0)
            if node == parent:
                return True

            if node not in visited:
                visited.add(node)
                queue.extend(self._children.get(node, []))

        return False

    def _update_cpt_parents(self, variable: str) -> None:
        """Update CPT after parent structure changes."""
        parents = tuple(sorted(self._parents.get(variable, set())))

        # Create new CPT with updated parents
        self._cpts[variable] = CPT(
            variable,
            parents,
            self.config.smoothing_alpha,
        )

        # Note: This loses existing observations. For structure learning,
        # retrain after structure changes.

    def get_variables(self) -> List[str]:
        """Get all variables."""
        return list(self._variables)

    def get_edges(self) -> List[Tuple[str, str]]:
        """Get all edges as (parent, child) tuples."""
        return list(self._edges)

    def get_parents(self, variable: str) -> List[str]:
        """Get parents of a variable."""
        return list(self._parents.get(variable, set()))

    def get_children(self, variable: str) -> List[str]:
        """Get children of a variable."""
        return list(self._children.get(variable, set()))

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def observe(self, observation: Dict[str, int]) -> None:
        """Record an observation of all variables.

        Args:
            observation: Dictionary {variable_name: discretized_value}
        """
        for variable in self._variables:
            if variable not in observation:
                continue

            value = observation[variable]
            parents = tuple(sorted(self._parents.get(variable, set())))

            # Get parent values
            parent_values = tuple(
                observation.get(p, 0) for p in parents
            )

            self._cpts[variable].observe(value, parent_values)

        self._num_observations += 1

    def train(self, observations: List[NoteObservation]) -> None:
        """Train network from a sequence of observations.

        Automatically sets up temporal variables and observations.

        Args:
            observations: List of note observations
        """
        if len(observations) < self.config.temporal_order + 1:
            return

        # Setup structure if using fixed mode
        if self.config.structure_mode == StructureMode.FIXED:
            self._setup_fixed_structure()
        elif self.config.structure_mode == StructureMode.LEARNED:
            self._learn_structure(observations)

        # Process observations with temporal context
        for i in range(self.config.temporal_order, len(observations)):
            obs_dict = self._build_observation_dict(observations, i)
            self.observe(obs_dict)

    def _setup_fixed_structure(self) -> None:
        """Set up fixed network structure based on mode."""
        # Clear existing structure
        self._variables.clear()
        self._edges.clear()
        self._parents.clear()
        self._children.clear()
        self._cpts.clear()

        # Add current variables based on mode
        self.add_variable(VAR_PITCH)

        if self.config.mode in (NetworkMode.PITCH_DURATION, NetworkMode.PITCH_DURATION_VELOCITY, NetworkMode.FULL):
            self.add_variable(VAR_DURATION)

        if self.config.mode in (NetworkMode.PITCH_DURATION_VELOCITY, NetworkMode.FULL):
            self.add_variable(VAR_VELOCITY)

        if self.config.mode == NetworkMode.FULL:
            self.add_variable(VAR_IOI)

        # Add temporal variables and edges
        for lag in range(1, self.config.temporal_order + 1):
            prev_pitch = var_pitch_lag(lag)
            self.add_variable(prev_pitch)
            self.add_edge(prev_pitch, VAR_PITCH)

            if self.config.mode in (NetworkMode.PITCH_DURATION, NetworkMode.PITCH_DURATION_VELOCITY, NetworkMode.FULL):
                prev_dur = var_duration_lag(lag)
                self.add_variable(prev_dur)
                # Previous pitch influences current duration
                self.add_edge(prev_pitch, VAR_DURATION)

        # Add same-timestep dependencies
        if self.config.mode in (NetworkMode.PITCH_DURATION, NetworkMode.PITCH_DURATION_VELOCITY, NetworkMode.FULL):
            self.add_edge(VAR_PITCH, VAR_DURATION)

        if self.config.mode in (NetworkMode.PITCH_DURATION_VELOCITY, NetworkMode.FULL):
            self.add_edge(VAR_PITCH, VAR_VELOCITY)
            self.add_edge(VAR_DURATION, VAR_VELOCITY)

        if self.config.mode == NetworkMode.FULL:
            self.add_edge(VAR_PITCH, VAR_IOI)
            self.add_edge(VAR_DURATION, VAR_IOI)

    def _learn_structure(self, observations: List[NoteObservation]) -> None:
        """Learn network structure from data.

        Uses a simple greedy approach based on mutual information.
        """
        # Start with independent structure
        self._setup_fixed_structure()

        # For now, use fixed structure as baseline
        # More sophisticated structure learning could be added here
        # (e.g., K2 algorithm, PC algorithm, etc.)

    def _build_observation_dict(
        self,
        observations: List[NoteObservation],
        index: int,
    ) -> Dict[str, int]:
        """Build observation dictionary with temporal context.

        Args:
            observations: All observations
            index: Current observation index

        Returns:
            Dictionary of discretized variable values
        """
        current = observations[index]
        obs_dict: Dict[str, int] = {}

        # Current values
        obs_dict[VAR_PITCH] = self._discretize_pitch(current.pitch)

        if VAR_DURATION in self._variables:
            obs_dict[VAR_DURATION] = self._discretize_duration(current.duration)

        if VAR_VELOCITY in self._variables:
            obs_dict[VAR_VELOCITY] = self._discretize_velocity(current.velocity)

        if VAR_IOI in self._variables:
            obs_dict[VAR_IOI] = self._discretize_ioi(current.ioi)

        # Lagged values
        for lag in range(1, self.config.temporal_order + 1):
            prev_idx = index - lag
            if prev_idx >= 0:
                prev = observations[prev_idx]

                prev_pitch_var = var_pitch_lag(lag)
                if prev_pitch_var in self._variables:
                    obs_dict[prev_pitch_var] = self._discretize_pitch(prev.pitch)

                prev_dur_var = var_duration_lag(lag)
                if prev_dur_var in self._variables:
                    obs_dict[prev_dur_var] = self._discretize_duration(prev.duration)

        return obs_dict

    # -------------------------------------------------------------------------
    # Inference and Sampling
    # -------------------------------------------------------------------------

    def sample(
        self,
        evidence: Optional[Dict[str, int]] = None,
    ) -> Dict[str, int]:
        """Sample from the network given evidence.

        Uses ancestral sampling (topological order).

        Args:
            evidence: Fixed values for some variables

        Returns:
            Sampled values for all variables
        """
        evidence = evidence or {}
        sample: Dict[str, int] = dict(evidence)

        # Topological sort
        order = self._topological_sort()

        for variable in order:
            if variable in sample:
                continue  # Already have evidence

            parents = tuple(sorted(self._parents.get(variable, set())))
            parent_values = tuple(sample.get(p, 0) for p in parents)

            cpt = self._cpts.get(variable)
            if cpt:
                value = cpt.sample(parent_values, self._rng)
                if value is not None:
                    sample[variable] = value

        return sample

    def get_probability(
        self,
        variable: str,
        value: int,
        evidence: Dict[str, int],
    ) -> float:
        """Get P(variable=value | evidence).

        Args:
            variable: Variable to query
            value: Value to get probability for
            evidence: Evidence dictionary

        Returns:
            Probability
        """
        parents = tuple(sorted(self._parents.get(variable, set())))
        parent_values = tuple(evidence.get(p, 0) for p in parents)

        cpt = self._cpts.get(variable)
        if cpt:
            return cpt.get_probability(value, parent_values)
        return 0.0

    def get_distribution(
        self,
        variable: str,
        evidence: Dict[str, int],
    ) -> Dict[int, float]:
        """Get P(variable | evidence).

        Args:
            variable: Variable to query
            evidence: Evidence dictionary

        Returns:
            Probability distribution
        """
        parents = tuple(sorted(self._parents.get(variable, set())))
        parent_values = tuple(evidence.get(p, 0) for p in parents)

        cpt = self._cpts.get(variable)
        if cpt:
            return cpt.get_distribution(parent_values)
        return {}

    def _topological_sort(self) -> List[str]:
        """Get variables in topological order."""
        # Kahn's algorithm
        in_degree = {v: len(self._parents.get(v, set())) for v in self._variables}
        queue = [v for v in self._variables if in_degree[v] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for child in self._children.get(node, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return result

    # -------------------------------------------------------------------------
    # Discretization
    # -------------------------------------------------------------------------

    def _discretize_pitch(self, pitch: int) -> int:
        """Discretize pitch value."""
        if self.config.pitch_bins <= 0:
            return pitch  # Use raw MIDI value

        # Bin into pitch_bins groups
        bin_size = 128 // self.config.pitch_bins
        return min(self.config.pitch_bins - 1, pitch // bin_size)

    def _undiscretize_pitch(self, binned: int) -> int:
        """Convert discretized pitch back to MIDI value."""
        if self.config.pitch_bins <= 0:
            return binned

        bin_size = 128 // self.config.pitch_bins
        # Return center of bin
        return binned * bin_size + bin_size // 2

    def _discretize_duration(self, duration: float) -> int:
        """Discretize duration to bin index."""
        # Quantize to bins (in 16th notes assuming quarter = 1 beat)
        sixteenths = max(1, min(self.config.duration_bins, round(duration * 4)))
        return sixteenths - 1  # 0-indexed

    def _undiscretize_duration(self, binned: int) -> float:
        """Convert discretized duration back to beats."""
        return (binned + 1) / 4.0

    def _discretize_velocity(self, velocity: int) -> int:
        """Discretize velocity to bin index."""
        bin_size = 128 // self.config.velocity_bins
        return min(self.config.velocity_bins - 1, velocity // bin_size)

    def _undiscretize_velocity(self, binned: int) -> int:
        """Convert discretized velocity back to MIDI value."""
        bin_size = 128 // self.config.velocity_bins
        return min(127, binned * bin_size + bin_size // 2)

    def _discretize_ioi(self, ioi: float) -> int:
        """Discretize inter-onset interval to bin index."""
        # Similar to duration
        units = max(1, min(self.config.ioi_bins, round(ioi * 4)))
        return units - 1

    def _undiscretize_ioi(self, binned: int) -> float:
        """Convert discretized IOI back to beats."""
        return (binned + 1) / 4.0

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_num_observations(self) -> int:
        """Get total number of observations."""
        return self._num_observations

    def get_entropy(self, variable: str, evidence: Optional[Dict[str, int]] = None) -> float:
        """Get entropy of a variable given evidence.

        Args:
            variable: Variable name
            evidence: Optional evidence

        Returns:
            Entropy in bits
        """
        evidence = evidence or {}
        parents = tuple(sorted(self._parents.get(variable, set())))
        parent_values = tuple(evidence.get(p, 0) for p in parents)

        cpt = self._cpts.get(variable)
        if cpt:
            return cpt.get_entropy(parent_values)
        return 0.0

    def get_average_entropy(self) -> float:
        """Get average entropy across all variables."""
        if not self._variables:
            return 0.0

        total = sum(self.get_entropy(var) for var in self._variables)
        return total / len(self._variables)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert network to dictionary."""
        return {
            'version': '1.0',
            'config': self.config.to_dict(),
            'variables': list(self._variables),
            'edges': [list(e) for e in self._edges],
            'cpts': {var: cpt.to_dict() for var, cpt in self._cpts.items()},
            'track_name': self._track_name,
            'source_file': self._source_file,
            'num_observations': self._num_observations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BayesianNetwork':
        """Create network from dictionary."""
        config = NetworkConfig.from_dict(data.get('config', {}))
        network = cls(config)

        # Restore variables
        for var in data.get('variables', []):
            network.add_variable(var)

        # Restore edges
        for parent, child in data.get('edges', []):
            network.add_edge(parent, child)

        # Restore CPTs
        for var, cpt_data in data.get('cpts', {}).items():
            network._cpts[var] = CPT.from_dict(cpt_data)

        network._track_name = data.get('track_name', '')
        network._source_file = data.get('source_file', '')
        network._num_observations = data.get('num_observations', 0)

        return network

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> 'BayesianNetwork':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, filepath: Union[str, Path]) -> None:
        """Save network to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'BayesianNetwork':
        """Load network from JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Network file not found: {filepath}")
        return cls.from_json(path.read_text())

    def __repr__(self) -> str:
        return (
            f"BayesianNetwork(variables={len(self._variables)}, "
            f"edges={len(self._edges)}, "
            f"observations={self._num_observations})"
        )


# ============================================================================
# MIDI Analyzer
# ============================================================================


class MIDIBayesAnalyzer:
    """Analyzes MIDI files to build Bayesian networks.

    Example:
        >>> analyzer = MIDIBayesAnalyzer()
        >>> network = analyzer.analyze_file("song.mid", track_index=0)
        >>>
        >>> # With custom config
        >>> config = NetworkConfig(mode=NetworkMode.FULL, temporal_order=2)
        >>> analyzer = MIDIBayesAnalyzer(config=config)
        >>> network = analyzer.analyze_file("song.mid")
    """

    def __init__(
        self,
        mode: NetworkMode = NetworkMode.PITCH_DURATION_VELOCITY,
        temporal_order: int = 1,
        config: Optional[NetworkConfig] = None,
    ):
        """Initialize analyzer.

        Args:
            mode: What properties to model
            temporal_order: Number of previous notes to consider
            config: Full configuration (overrides other params)
        """
        if config:
            self.config = config
        else:
            self.config = NetworkConfig(
                mode=mode,
                temporal_order=temporal_order,
            )

    def analyze_file(
        self,
        filepath: Union[str, Path],
        track_index: int = 0,
    ) -> BayesianNetwork:
        """Analyze a single track from a MIDI file.

        Args:
            filepath: MIDI file path
            track_index: Track index (0-based)

        Returns:
            Trained BayesianNetwork
        """
        sequence = MIDISequence.load(str(filepath))
        return self.analyze_track(sequence, track_index, str(filepath))

    def analyze_track(
        self,
        sequence: MIDISequence,
        track_index: int = 0,
        source_file: str = "",
    ) -> BayesianNetwork:
        """Analyze a track from a MIDISequence.

        Args:
            sequence: MIDI sequence
            track_index: Track index
            source_file: Source filename (for metadata)

        Returns:
            Trained BayesianNetwork
        """
        if track_index >= len(sequence.tracks):
            raise ValueError(f"Track index {track_index} out of range")

        track = sequence.tracks[track_index]
        observations = self._extract_observations(track, sequence.tempo)

        network = BayesianNetwork(self.config)
        network._track_name = track.name
        network._source_file = source_file

        if observations:
            network.train(observations)

        return network

    def analyze_all_tracks(
        self,
        filepath: Union[str, Path],
    ) -> List[BayesianNetwork]:
        """Analyze all tracks in a MIDI file.

        Args:
            filepath: MIDI file path

        Returns:
            List of BayesianNetworks, one per track
        """
        sequence = MIDISequence.load(str(filepath))
        networks = []

        for i in range(len(sequence.tracks)):
            network = self.analyze_track(sequence, i, str(filepath))
            networks.append(network)

        return networks

    def _extract_observations(
        self,
        track: MIDITrack,
        tempo: float,
    ) -> List[NoteObservation]:
        """Extract note observations from track."""
        observations: List[NoteObservation] = []

        # Match note-ons with note-offs
        note_ons: Dict[int, MIDIEvent] = {}
        beat_duration = 60.0 / tempo

        for event in sorted(track.events, key=lambda e: e.time):
            if event.is_note_on:
                note_ons[event.data1] = event
            elif event.is_note_off:
                if event.data1 in note_ons:
                    on_event = note_ons.pop(event.data1)

                    time_beats = on_event.time / beat_duration
                    duration_beats = (event.time - on_event.time) / beat_duration

                    observations.append(NoteObservation(
                        pitch=event.data1,
                        duration=max(0.001, duration_beats),
                        velocity=on_event.data2,
                        time=time_beats,
                    ))

        # Sort by time and compute IOIs
        observations.sort(key=lambda n: n.time)

        for i in range(1, len(observations)):
            observations[i].ioi = observations[i].time - observations[i - 1].time

        return observations


# ============================================================================
# MIDI Generator
# ============================================================================


class MIDIBayesGenerator:
    """Generates MIDI sequences from trained Bayesian networks.

    Example:
        >>> network = MIDIBayesAnalyzer().analyze_file("song.mid")
        >>> generator = MIDIBayesGenerator(network)
        >>> sequence = generator.generate(num_notes=64, tempo=120.0)
        >>> sequence.save("variation.mid")
    """

    def __init__(self, network: BayesianNetwork):
        """Initialize generator.

        Args:
            network: Trained BayesianNetwork
        """
        self.network = network
        self._rng = random.Random(network.config.seed)

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
        track = sequence.add_track(f"Generated from {self.network._track_name or 'network'}")
        track.channel = channel

        if self.network.get_num_observations() == 0:
            return sequence

        # Generate notes
        history: List[NoteObservation] = []
        current_time = 0.0
        beat_duration = 60.0 / tempo

        for i in range(num_notes):
            # Build evidence from history
            evidence = self._build_evidence(history, start_pitch if i == 0 else None)

            # Sample from network
            sample = self.network.sample(evidence)

            # Convert to note
            pitch = self.network._undiscretize_pitch(sample.get(VAR_PITCH, 60))
            duration = self.network._undiscretize_duration(
                sample.get(VAR_DURATION, self.network.config.duration_bins // 2)
            )
            velocity = self.network._undiscretize_velocity(
                sample.get(VAR_VELOCITY, self.network.config.velocity_bins // 2)
            )
            ioi = self.network._undiscretize_ioi(
                sample.get(VAR_IOI, self.network.config.ioi_bins // 2)
            )

            # Use defaults if not modeled
            if VAR_DURATION not in self.network._variables:
                duration = self.network.config.default_duration
            if VAR_VELOCITY not in self.network._variables:
                velocity = self.network.config.default_velocity
            if VAR_IOI not in self.network._variables:
                ioi = self.network.config.default_ioi

            # Add note
            track.add_note(
                time=current_time * beat_duration,
                note=pitch,
                velocity=velocity,
                duration=duration * beat_duration,
                channel=channel,
            )

            # Update history
            obs = NoteObservation(
                pitch=pitch,
                duration=duration,
                velocity=velocity,
                ioi=ioi,
                time=current_time,
            )
            history.append(obs)

            # Trim history to temporal order
            if len(history) > self.network.config.temporal_order:
                history = history[-self.network.config.temporal_order:]

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
        if self.network.get_num_observations() == 0:
            return

        ch = channel if channel is not None else track.channel

        history: List[NoteObservation] = []
        current_time = start_time
        beat_duration = 60.0 / tempo

        for i in range(num_notes):
            evidence = self._build_evidence(history, start_pitch if i == 0 else None)
            sample = self.network.sample(evidence)

            pitch = self.network._undiscretize_pitch(sample.get(VAR_PITCH, 60))
            duration = self.network._undiscretize_duration(
                sample.get(VAR_DURATION, self.network.config.duration_bins // 2)
            )
            velocity = self.network._undiscretize_velocity(
                sample.get(VAR_VELOCITY, self.network.config.velocity_bins // 2)
            )
            ioi = self.network._undiscretize_ioi(
                sample.get(VAR_IOI, self.network.config.ioi_bins // 2)
            )

            if VAR_DURATION not in self.network._variables:
                duration = self.network.config.default_duration
            if VAR_VELOCITY not in self.network._variables:
                velocity = self.network.config.default_velocity
            if VAR_IOI not in self.network._variables:
                ioi = self.network.config.default_ioi

            track.add_note(
                time=current_time,
                note=pitch,
                velocity=velocity,
                duration=duration * beat_duration,
                channel=ch,
            )

            obs = NoteObservation(
                pitch=pitch,
                duration=duration,
                velocity=velocity,
                ioi=ioi,
                time=current_time / beat_duration,
            )
            history.append(obs)

            if len(history) > self.network.config.temporal_order:
                history = history[-self.network.config.temporal_order:]

            current_time += ioi * beat_duration

    def _build_evidence(
        self,
        history: List[NoteObservation],
        start_pitch: Optional[int],
    ) -> Dict[str, int]:
        """Build evidence dictionary from history."""
        evidence: Dict[str, int] = {}

        # Add lagged variables from history
        for lag in range(1, self.network.config.temporal_order + 1):
            hist_idx = len(history) - lag
            if hist_idx >= 0:
                prev = history[hist_idx]

                prev_pitch_var = var_pitch_lag(lag)
                if prev_pitch_var in self.network._variables:
                    evidence[prev_pitch_var] = self.network._discretize_pitch(prev.pitch)

                prev_dur_var = var_duration_lag(lag)
                if prev_dur_var in self.network._variables:
                    evidence[prev_dur_var] = self.network._discretize_duration(prev.duration)

        # Add start pitch if specified and no history
        if start_pitch is not None and len(history) == 0:
            evidence[VAR_PITCH] = self.network._discretize_pitch(start_pitch)

        return evidence


# ============================================================================
# Utility Functions
# ============================================================================


def analyze_and_generate(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    num_notes: int = 64,
    mode: NetworkMode = NetworkMode.PITCH_DURATION_VELOCITY,
    temporal_order: int = 1,
    track_index: int = 0,
) -> MIDISequence:
    """Convenience function to analyze a MIDI file and generate a variation.

    Args:
        input_file: Source MIDI file
        output_file: Output MIDI file
        num_notes: Number of notes to generate
        mode: Network modeling mode
        temporal_order: Temporal order
        track_index: Track to analyze

    Returns:
        Generated MIDISequence
    """
    config = NetworkConfig(mode=mode, temporal_order=temporal_order)
    analyzer = MIDIBayesAnalyzer(config=config)
    network = analyzer.analyze_file(input_file, track_index)

    source_seq = MIDISequence.load(str(input_file))
    generator = MIDIBayesGenerator(network)
    output_seq = generator.generate(num_notes=num_notes, tempo=source_seq.tempo)

    output_seq.save(str(output_file))
    return output_seq


def merge_networks(
    networks: List[BayesianNetwork],
    weights: Optional[List[float]] = None,
) -> BayesianNetwork:
    """Merge multiple networks by combining their CPTs.

    Args:
        networks: List of networks to merge
        weights: Optional weights for each network

    Returns:
        Merged BayesianNetwork
    """
    if not networks:
        raise ValueError("Must provide at least one network")

    if weights is None:
        weights = [1.0] * len(networks)

    if len(weights) != len(networks):
        raise ValueError(f"Weights length {len(weights)} != networks length {len(networks)}")

    # Use first network as template
    merged = BayesianNetwork(networks[0].config)

    # Copy structure from first network
    for var in networks[0].get_variables():
        merged.add_variable(var)
    for parent, child in networks[0].get_edges():
        merged.add_edge(parent, child)

    # Merge CPTs by weighted averaging of counts
    for var in merged._variables:
        merged_cpt = merged._cpts[var]

        for network, weight in zip(networks, weights):
            if var not in network._cpts:
                continue

            source_cpt = network._cpts[var]

            # Add weighted counts
            for parent_vals, value_counts in source_cpt._counts.items():
                for value, count in value_counts.items():
                    merged_cpt._counts[parent_vals][value] += int(count * weight)
                    merged_cpt._values.add(value)

    return merged


def network_statistics(network: BayesianNetwork) -> Dict[str, Any]:
    """Get statistics about a network.

    Args:
        network: BayesianNetwork to analyze

    Returns:
        Dictionary of statistics
    """
    return {
        'mode': network.config.mode.name,
        'structure_mode': network.config.structure_mode.name,
        'temporal_order': network.config.temporal_order,
        'num_variables': len(network._variables),
        'num_edges': len(network._edges),
        'num_observations': network.get_num_observations(),
        'average_entropy': network.get_average_entropy(),
        'variables': network.get_variables(),
        'edges': network.get_edges(),
        'track_name': network._track_name,
        'source_file': network._source_file,
    }
