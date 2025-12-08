#!/usr/bin/env python3
"""Music generation using trained neural network models.

This module provides:
- Sampling strategies: Greedy, Temperature, TopK, Nucleus (Top-p)
- MusicGenerator: Generate music sequences from trained models

Example:
    >>> from coremusic.music.neural import MusicGenerator, NoteEncoder
    >>> from coremusic.music.neural.generation import TemperatureSampling
    >>> from coremusic.kann import NeuralNetwork
    >>>
    >>> model = NeuralNetwork.load('model.kan')
    >>> encoder = NoteEncoder()
    >>> generator = MusicGenerator(model, encoder, seq_length=32)
    >>>
    >>> # Generate with temperature sampling
    >>> tokens = generator.generate(length=64, temperature=0.8)
    >>> events = encoder.decode(tokens)
"""

import array
import math
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Union

from coremusic.kann import NeuralNetwork
from coremusic.midi.utilities import MIDISequence

from .data import BaseEncoder

# ============================================================================
# Sampling Strategies
# ============================================================================


class SamplingStrategy(ABC):
    """Base class for sampling strategies.

    Sampling strategies determine how to select the next token from
    the model's output probability distribution.
    """

    @abstractmethod
    def sample(self, probabilities: Union[array.array, List[float]]) -> int:
        """Sample next token from probability distribution.

        Args:
            probabilities: Probability distribution over vocabulary

        Returns:
            Sampled token index
        """
        pass


class GreedySampling(SamplingStrategy):
    """Always select the highest probability token.

    This produces deterministic output - the same seed always generates
    the same sequence. Good for testing but often produces repetitive music.

    Example:
        >>> sampler = GreedySampling()
        >>> token = sampler.sample([0.1, 0.7, 0.2])  # Always returns 1
    """

    def sample(self, probabilities: Union[array.array, List[float]]) -> int:
        """Return index of maximum probability."""
        max_idx = 0
        max_prob = probabilities[0]
        for i in range(1, len(probabilities)):
            if probabilities[i] > max_prob:
                max_prob = probabilities[i]
                max_idx = i
        return max_idx


class TemperatureSampling(SamplingStrategy):
    """Sample with temperature scaling.

    Temperature controls randomness:
    - temperature < 1.0: More deterministic (sharper distribution)
    - temperature = 1.0: Original distribution
    - temperature > 1.0: More random (flatter distribution)

    Args:
        temperature: Temperature parameter (default: 1.0)

    Example:
        >>> sampler = TemperatureSampling(temperature=0.8)
        >>> token = sampler.sample(probabilities)
    """

    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = temperature

    def sample(self, probabilities: Union[array.array, List[float]]) -> int:
        """Sample with temperature scaling."""
        n = len(probabilities)

        # Apply temperature scaling in log space
        scaled = []
        for p in probabilities:
            # Add small epsilon to avoid log(0)
            log_p = math.log(max(p, 1e-10))
            scaled.append(math.exp(log_p / self.temperature))

        # Normalize
        total = sum(scaled)
        if total > 0:
            scaled = [s / total for s in scaled]
        else:
            # Uniform if all zeros
            scaled = [1.0 / n] * n

        # Sample from distribution
        r = random.random()
        cumsum = 0.0
        for i in range(n):
            cumsum += scaled[i]
            if r < cumsum:
                return i
        return n - 1


class TopKSampling(SamplingStrategy):
    """Sample from top-k most likely tokens.

    Only considers the k highest probability tokens, redistributing
    probability mass among them. Helps avoid unlikely/nonsensical outputs.

    Args:
        k: Number of top tokens to consider
        temperature: Temperature for sampling among top-k (default: 1.0)

    Example:
        >>> sampler = TopKSampling(k=10, temperature=0.8)
        >>> token = sampler.sample(probabilities)
    """

    def __init__(self, k: int = 10, temperature: float = 1.0):
        if k <= 0:
            raise ValueError("k must be positive")
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.k = k
        self.temperature = temperature

    def sample(self, probabilities: Union[array.array, List[float]]) -> int:
        """Sample from top-k tokens."""
        n = len(probabilities)
        k = min(self.k, n)

        # Get indices sorted by probability (descending)
        indexed = [(i, probabilities[i]) for i in range(n)]
        indexed.sort(key=lambda x: x[1], reverse=True)

        # Take top-k
        top_k = indexed[:k]

        # Apply temperature scaling
        scaled = []
        for idx, p in top_k:
            log_p = math.log(max(p, 1e-10))
            scaled.append((idx, math.exp(log_p / self.temperature)))

        # Normalize
        total = sum(s[1] for s in scaled)
        if total > 0:
            scaled = [(idx, s / total) for idx, s in scaled]
        else:
            scaled = [(idx, 1.0 / k) for idx, _ in scaled]

        # Sample
        r = random.random()
        cumsum = 0.0
        for idx, prob in scaled:
            cumsum += prob
            if r < cumsum:
                return idx
        return scaled[-1][0]


class NucleusSampling(SamplingStrategy):
    """Sample from nucleus (top-p) of probability mass.

    Dynamically selects the smallest set of tokens whose cumulative
    probability exceeds p. More adaptive than fixed top-k.

    Args:
        p: Cumulative probability threshold (0.0-1.0)
        temperature: Temperature for sampling (default: 1.0)

    Example:
        >>> sampler = NucleusSampling(p=0.9, temperature=0.8)
        >>> token = sampler.sample(probabilities)
    """

    def __init__(self, p: float = 0.9, temperature: float = 1.0):
        if not 0.0 < p <= 1.0:
            raise ValueError("p must be in (0, 1]")
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.p = p
        self.temperature = temperature

    def sample(self, probabilities: Union[array.array, List[float]]) -> int:
        """Sample from nucleus."""
        n = len(probabilities)

        # Get indices sorted by probability (descending)
        indexed = [(i, probabilities[i]) for i in range(n)]
        indexed.sort(key=lambda x: x[1], reverse=True)

        # Find nucleus (smallest set with cumulative prob >= p)
        nucleus = []
        cumsum = 0.0
        for idx, prob in indexed:
            nucleus.append((idx, prob))
            cumsum += prob
            if cumsum >= self.p:
                break

        # Apply temperature scaling
        scaled = []
        for idx, p in nucleus:
            log_p = math.log(max(p, 1e-10))
            scaled.append((idx, math.exp(log_p / self.temperature)))

        # Normalize
        total = sum(s[1] for s in scaled)
        if total > 0:
            scaled = [(idx, s / total) for idx, s in scaled]
        else:
            scaled = [(idx, 1.0 / len(scaled)) for idx, _ in scaled]

        # Sample
        r = random.random()
        cumsum = 0.0
        for idx, prob in scaled:
            cumsum += prob
            if r < cumsum:
                return idx
        return scaled[-1][0]


# ============================================================================
# Music Generator
# ============================================================================


class MusicGenerator:
    """Generate music using a trained neural network model.

    This class handles the generation loop: feeding sequences to the model,
    sampling from outputs, and converting tokens back to MIDI.

    Example:
        >>> model = NeuralNetwork.load('model.kan')
        >>> encoder = NoteEncoder()
        >>> generator = MusicGenerator(model, encoder, seq_length=32)
        >>>
        >>> # Generate tokens
        >>> tokens = generator.generate(length=64, temperature=0.8)
        >>>
        >>> # Generate MIDI sequence
        >>> sequence = generator.generate_midi(duration_beats=32, tempo=120)
        >>> sequence.save('output.mid')
    """

    def __init__(
        self,
        model: NeuralNetwork,
        encoder: BaseEncoder,
        seq_length: int,
        sampling: Optional[SamplingStrategy] = None,
    ):
        """Initialize generator.

        Args:
            model: Trained neural network
            encoder: Encoder used for tokenization
            seq_length: Sequence length the model was trained on
            sampling: Sampling strategy (default: TemperatureSampling(1.0))
        """
        self.model = model
        self.encoder = encoder
        self.seq_length = seq_length
        self.sampling = sampling or TemperatureSampling(1.0)

    def set_sampling(self, sampling: SamplingStrategy) -> "MusicGenerator":
        """Set sampling strategy.

        Args:
            sampling: New sampling strategy

        Returns:
            Self for chaining
        """
        self.sampling = sampling
        return self

    def generate(
        self,
        seed: Optional[List[int]] = None,
        length: int = 100,
        temperature: Optional[float] = None,
    ) -> List[int]:
        """Generate a sequence of tokens.

        Args:
            seed: Starting sequence (random if None, must be seq_length tokens)
            length: Number of new tokens to generate
            temperature: Override sampling temperature (if using TemperatureSampling)

        Returns:
            List of generated token indices (seed + new tokens)
        """
        vocab_size = self.encoder.vocab_size

        # Initialize seed
        if seed is None:
            # Random seed
            seed = [random.randint(0, vocab_size - 1) for _ in range(self.seq_length)]
        elif len(seed) < self.seq_length:
            # Pad with random tokens
            pad_length = self.seq_length - len(seed)
            seed = [random.randint(0, vocab_size - 1) for _ in range(pad_length)] + list(seed)
        elif len(seed) > self.seq_length:
            # Take last seq_length tokens
            seed = list(seed[-self.seq_length:])
        else:
            seed = list(seed)

        # Temporarily override temperature if specified
        original_sampling = self.sampling
        if temperature is not None and isinstance(self.sampling, TemperatureSampling):
            self.sampling = TemperatureSampling(temperature)

        # Generate tokens
        generated = seed.copy()

        for _ in range(length):
            # Get last seq_length tokens
            context = generated[-self.seq_length:]

            # One-hot encode for model input
            input_data = array.array('f', [0.0] * (self.seq_length * vocab_size))
            for i, token in enumerate(context):
                if 0 <= token < vocab_size:
                    input_data[i * vocab_size + token] = 1.0

            # Get model prediction
            output = self.model.apply(input_data)

            # Sample next token
            next_token = self.sampling.sample(output)
            generated.append(next_token)

        # Restore original sampling
        self.sampling = original_sampling

        return generated

    def generate_midi(
        self,
        seed: Optional[List[int]] = None,
        seed_midi: Optional[str] = None,
        duration_beats: int = 32,
        tempo: float = 120.0,
        track_name: str = "Generated",
    ) -> MIDISequence:
        """Generate a MIDI sequence.

        Args:
            seed: Starting token sequence
            seed_midi: Path to MIDI file to use as seed (overrides seed)
            duration_beats: Approximate length in beats
            tempo: Output tempo (BPM)
            track_name: Name for the generated track

        Returns:
            MIDISequence ready for playback or saving
        """
        # Load seed from MIDI if provided
        if seed_midi is not None:
            seed = self.encoder.encode_file(seed_midi)
            if len(seed) > self.seq_length:
                seed = seed[-self.seq_length:]

        # Estimate tokens needed (rough: 4 tokens per beat for note-only encoding)
        tokens_per_beat = 4
        length = duration_beats * tokens_per_beat

        # Generate tokens
        tokens = self.generate(seed=seed, length=length)

        # Remove seed from output if we want only new content
        new_tokens = tokens[self.seq_length:]

        # Decode to MIDI events
        events = self.encoder.decode(new_tokens, tempo=tempo)

        # Create MIDI sequence
        sequence = MIDISequence(tempo=tempo)
        track = sequence.add_track(track_name)

        # Add events to track
        for event in events:
            track.events.append(event)

        return sequence

    def continue_sequence(
        self,
        midi_path: str,
        continuation_length: int = 32,
        tempo: Optional[float] = None,
    ) -> MIDISequence:
        """Continue an existing MIDI file.

        Args:
            midi_path: Path to input MIDI file
            continuation_length: Number of beats to add
            tempo: Output tempo (uses input tempo if None)

        Returns:
            New MIDISequence with original + continuation
        """
        # Load original
        original = MIDISequence.load(midi_path)
        if tempo is None:
            tempo = original.tempo

        # Encode original
        all_events = []
        for track in original.tracks:
            all_events.extend(track.events)
        all_events.sort(key=lambda e: e.time)
        original_tokens = self.encoder.encode(all_events)

        # Get seed (last seq_length tokens)
        if len(original_tokens) >= self.seq_length:
            seed = original_tokens[-self.seq_length:]
        else:
            seed = original_tokens

        # Generate continuation
        tokens_per_beat = 4
        length = continuation_length * tokens_per_beat
        generated = self.generate(seed=seed, length=length)
        new_tokens = generated[self.seq_length:]

        # Decode new tokens
        new_events = self.encoder.decode(new_tokens, tempo=tempo)

        # Get end time of original
        original_end_time = max(e.time for e in all_events) if all_events else 0.0

        # Offset new events
        for event in new_events:
            event.time += original_end_time

        # Create combined sequence
        combined = MIDISequence(tempo=tempo)
        track = combined.add_track("Combined")

        # Add original events
        for event in all_events:
            track.events.append(event)

        # Add new events
        for event in new_events:
            track.events.append(event)

        # Sort by time
        track.events.sort(key=lambda e: e.time)

        return combined

    def generate_variations(
        self,
        seed: List[int],
        num_variations: int = 4,
        length: int = 64,
        temperature_range: tuple = (0.5, 1.2),
    ) -> List[List[int]]:
        """Generate multiple variations from the same seed.

        Args:
            seed: Starting sequence
            num_variations: Number of variations to generate
            length: Length of each variation
            temperature_range: (min, max) temperature range

        Returns:
            List of generated token sequences
        """
        variations = []
        temp_min, temp_max = temperature_range
        temp_step = (temp_max - temp_min) / max(1, num_variations - 1)

        for i in range(num_variations):
            temp = temp_min + i * temp_step
            tokens = self.generate(seed=seed, length=length, temperature=temp)
            variations.append(tokens)

        return variations
