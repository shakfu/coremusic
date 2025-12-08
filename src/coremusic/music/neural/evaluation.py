#!/usr/bin/env python3
"""Evaluation metrics and model comparison tools for neural music generation.

This module provides:
- MusicMetrics: Metrics for evaluating generated music quality
- ModelComparison: Compare different models on the same dataset

Example:
    >>> from coremusic.music.neural import MusicMetrics, ModelComparison
    >>>
    >>> # Evaluate a single sequence
    >>> generated = [60, 62, 64, 65, 67]
    >>> reference = [60, 62, 64, 67, 69]
    >>> similarity = MusicMetrics.pitch_histogram_similarity(generated, reference)
    >>>
    >>> # Compare models
    >>> comparison = ModelComparison(test_dataset)
    >>> comparison.evaluate_model('MLP', mlp_model)
    >>> comparison.evaluate_model('LSTM', lstm_model)
    >>> results = comparison.compare()
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from coremusic.kann import NeuralNetwork
from coremusic.midi.utilities import MIDIEvent

from .data import BaseEncoder, MIDIDataset

# ============================================================================
# Music Metrics
# ============================================================================


class MusicMetrics:
    """Metrics for evaluating generated music quality.

    All methods are static and can be used without instantiation.

    Example:
        >>> similarity = MusicMetrics.pitch_histogram_similarity(generated, reference)
        >>> density = MusicMetrics.note_density(events, duration=10.0)
        >>> rep_score = MusicMetrics.repetition_score(sequence, n=4)
    """

    @staticmethod
    def pitch_histogram(sequence: List[int], normalize: bool = True) -> List[float]:
        """Compute pitch histogram from a note sequence.

        Args:
            sequence: List of MIDI note numbers (0-127)
            normalize: If True, normalize to sum to 1.0

        Returns:
            List of 128 values representing pitch distribution
        """
        histogram = [0.0] * 128
        for note in sequence:
            if 0 <= note < 128:
                histogram[note] += 1.0

        if normalize:
            total = sum(histogram)
            if total > 0:
                histogram = [h / total for h in histogram]

        return histogram

    @staticmethod
    def pitch_histogram_similarity(
        generated: List[int], reference: List[int]
    ) -> float:
        """Compare pitch distributions using cosine similarity.

        Args:
            generated: Generated note sequence
            reference: Reference note sequence

        Returns:
            Similarity score (0.0 to 1.0, higher is more similar)
        """
        hist_gen = MusicMetrics.pitch_histogram(generated)
        hist_ref = MusicMetrics.pitch_histogram(reference)

        # Cosine similarity
        dot_product = sum(g * r for g, r in zip(hist_gen, hist_ref))
        norm_gen = math.sqrt(sum(g * g for g in hist_gen))
        norm_ref = math.sqrt(sum(r * r for r in hist_ref))

        if norm_gen == 0 or norm_ref == 0:
            return 0.0

        return dot_product / (norm_gen * norm_ref)

    @staticmethod
    def interval_histogram(sequence: List[int], normalize: bool = True) -> List[float]:
        """Compute interval histogram from a note sequence.

        Args:
            sequence: List of MIDI note numbers
            normalize: If True, normalize to sum to 1.0

        Returns:
            List of 49 values for intervals -24 to +24 semitones
        """
        # Intervals from -24 to +24 (49 bins)
        histogram = [0.0] * 49

        for i in range(1, len(sequence)):
            interval = sequence[i] - sequence[i - 1]
            # Clamp to range -24 to +24
            interval = max(-24, min(24, interval))
            histogram[interval + 24] += 1.0

        if normalize:
            total = sum(histogram)
            if total > 0:
                histogram = [h / total for h in histogram]

        return histogram

    @staticmethod
    def interval_histogram_similarity(
        generated: List[int], reference: List[int]
    ) -> float:
        """Compare interval distributions using cosine similarity.

        Args:
            generated: Generated note sequence
            reference: Reference note sequence

        Returns:
            Similarity score (0.0 to 1.0)
        """
        hist_gen = MusicMetrics.interval_histogram(generated)
        hist_ref = MusicMetrics.interval_histogram(reference)

        dot_product = sum(g * r for g, r in zip(hist_gen, hist_ref))
        norm_gen = math.sqrt(sum(g * g for g in hist_gen))
        norm_ref = math.sqrt(sum(r * r for r in hist_ref))

        if norm_gen == 0 or norm_ref == 0:
            return 0.0

        return dot_product / (norm_gen * norm_ref)

    @staticmethod
    def note_density(events: List[MIDIEvent], duration: Optional[float] = None) -> float:
        """Calculate notes per second.

        Args:
            events: List of MIDI events
            duration: Total duration in seconds (auto-calculated if None)

        Returns:
            Notes per second
        """
        note_ons = [e for e in events if e.is_note_on]
        if not note_ons:
            return 0.0

        if duration is None:
            if not events:
                return 0.0
            duration = max(e.time for e in events)
            if duration <= 0:
                duration = 1.0

        return len(note_ons) / duration

    @staticmethod
    def pitch_range(events: List[MIDIEvent]) -> Tuple[int, int]:
        """Get minimum and maximum pitch.

        Args:
            events: List of MIDI events

        Returns:
            Tuple of (min_pitch, max_pitch)
        """
        note_ons = [e for e in events if e.is_note_on]
        if not note_ons:
            return (0, 0)

        pitches = [e.data1 for e in note_ons]
        return (min(pitches), max(pitches))

    @staticmethod
    def pitch_range_from_sequence(sequence: List[int]) -> Tuple[int, int]:
        """Get minimum and maximum pitch from a token sequence.

        Args:
            sequence: List of note numbers

        Returns:
            Tuple of (min_pitch, max_pitch)
        """
        valid = [n for n in sequence if 0 <= n < 128]
        if not valid:
            return (0, 0)
        return (min(valid), max(valid))

    @staticmethod
    def repetition_score(sequence: List[int], n: int = 4) -> float:
        """Measure n-gram repetition in a sequence.

        Args:
            sequence: Token sequence
            n: N-gram size

        Returns:
            Score from 0.0 (no repetition) to 1.0 (all repeated)
        """
        if len(sequence) < n:
            return 0.0

        ngrams = []
        for i in range(len(sequence) - n + 1):
            ngram = tuple(sequence[i : i + n])
            ngrams.append(ngram)

        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)

        if total_ngrams == 0:
            return 0.0

        # Repetition = 1 - (unique / total)
        return 1.0 - (unique_ngrams / total_ngrams)

    @staticmethod
    def unique_notes(sequence: List[int]) -> int:
        """Count unique notes in sequence.

        Args:
            sequence: Note sequence

        Returns:
            Number of unique notes
        """
        return len(set(n for n in sequence if 0 <= n < 128))

    @staticmethod
    def average_interval(sequence: List[int]) -> float:
        """Calculate average absolute interval between consecutive notes.

        Args:
            sequence: Note sequence

        Returns:
            Average absolute interval in semitones
        """
        if len(sequence) < 2:
            return 0.0

        intervals = [abs(sequence[i] - sequence[i - 1]) for i in range(1, len(sequence))]
        return sum(intervals) / len(intervals)

    @staticmethod
    def self_similarity_matrix(sequence: List[int], window: int = 8) -> List[List[float]]:
        """Compute self-similarity matrix for structure analysis.

        Args:
            sequence: Token sequence
            window: Window size for comparison

        Returns:
            2D similarity matrix
        """
        n = len(sequence) - window + 1
        if n <= 0:
            return [[1.0]]

        matrix = []
        for i in range(n):
            row = []
            window_i = sequence[i : i + window]
            for j in range(n):
                window_j = sequence[j : j + window]
                # Simple match count similarity
                matches = sum(1 for a, b in zip(window_i, window_j) if a == b)
                similarity = matches / window
                row.append(similarity)
            matrix.append(row)

        return matrix

    @staticmethod
    def evaluate_sequence(
        sequence: List[int], reference: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """Compute all metrics for a sequence.

        Args:
            sequence: Generated sequence
            reference: Reference sequence for comparison (optional)

        Returns:
            Dictionary of metric names to values
        """
        metrics = {
            "length": float(len(sequence)),
            "unique_notes": float(MusicMetrics.unique_notes(sequence)),
            "repetition_4gram": MusicMetrics.repetition_score(sequence, n=4),
            "repetition_8gram": MusicMetrics.repetition_score(sequence, n=8),
            "average_interval": MusicMetrics.average_interval(sequence),
        }

        pitch_min, pitch_max = MusicMetrics.pitch_range_from_sequence(sequence)
        metrics["pitch_range"] = float(pitch_max - pitch_min)
        metrics["pitch_min"] = float(pitch_min)
        metrics["pitch_max"] = float(pitch_max)

        if reference is not None:
            metrics["pitch_similarity"] = MusicMetrics.pitch_histogram_similarity(
                sequence, reference
            )
            metrics["interval_similarity"] = MusicMetrics.interval_histogram_similarity(
                sequence, reference
            )

        return metrics


# ============================================================================
# Model Comparison
# ============================================================================


@dataclass
class ModelResult:
    """Results from evaluating a single model."""

    name: str
    loss: float
    metrics: Dict[str, float] = field(default_factory=dict)
    generated_samples: List[List[int]] = field(default_factory=list)


class ModelComparison:
    """Compare different models on the same dataset.

    Example:
        >>> comparison = ModelComparison(test_dataset, encoder)
        >>> comparison.evaluate_model('MLP', mlp_model, x_test, y_test)
        >>> comparison.evaluate_model('LSTM', lstm_model, x_test, y_test)
        >>> results = comparison.compare()
        >>> print(results)
    """

    def __init__(
        self,
        dataset: MIDIDataset,
        encoder: Optional[BaseEncoder] = None,
        reference_sequence: Optional[List[int]] = None,
    ):
        """Initialize comparison.

        Args:
            dataset: Test dataset for evaluation
            encoder: Encoder for tokenization (uses dataset.encoder if None)
            reference_sequence: Reference sequence for similarity metrics
        """
        self.dataset = dataset
        self.encoder = encoder or dataset.encoder
        self.reference_sequence = reference_sequence
        self.results: Dict[str, ModelResult] = {}

        # If no reference provided, use first sequence from dataset
        if self.reference_sequence is None and dataset.sequences:
            self.reference_sequence = dataset.sequences[0]

    def evaluate_model(
        self,
        name: str,
        model: NeuralNetwork,
        x_test=None,
        y_test=None,
        num_samples: int = 5,
        generation_length: int = 64,
    ) -> ModelResult:
        """Evaluate a model and store results.

        Args:
            name: Model name for identification
            model: Trained neural network
            x_test: Test input data (uses dataset if None)
            y_test: Test target data (uses dataset if None)
            num_samples: Number of sequences to generate for evaluation
            generation_length: Length of generated sequences

        Returns:
            ModelResult with metrics
        """
        # Prepare test data if not provided
        if x_test is None or y_test is None:
            x_test, y_test = self.dataset.prepare_training_data(use_numpy=True)

        # Compute loss
        loss = model.cost(x_test, y_test)

        # Generate samples for quality evaluation
        from .generation import MusicGenerator, TemperatureSampling

        generator = MusicGenerator(
            model, self.encoder, self.dataset.seq_length,
            sampling=TemperatureSampling(0.8)
        )

        generated_samples = []
        all_metrics: Dict[str, List[float]] = {}

        for _ in range(num_samples):
            seed = self.dataset.get_sample_sequence(self.dataset.seq_length)
            tokens = generator.generate(seed=seed, length=generation_length)
            generated_samples.append(tokens)

            # Compute metrics for this sample
            sample_metrics = MusicMetrics.evaluate_sequence(
                tokens, self.reference_sequence
            )
            for key, value in sample_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

        # Average metrics across samples
        avg_metrics = {key: sum(values) / len(values) for key, values in all_metrics.items()}
        avg_metrics["loss"] = loss

        result = ModelResult(
            name=name,
            loss=loss,
            metrics=avg_metrics,
            generated_samples=generated_samples,
        )

        self.results[name] = result
        return result

    def compare(self) -> Dict[str, Dict[str, float]]:
        """Return comparison table as nested dict.

        Returns:
            Dict mapping model names to their metrics
        """
        return {name: result.metrics for name, result in self.results.items()}

    def compare_table(self) -> str:
        """Return comparison as formatted table string.

        Returns:
            Formatted table string
        """
        if not self.results:
            return "No models evaluated yet."

        # Get all metric names
        all_metrics: set[str] = set()
        for result in self.results.values():
            all_metrics.update(result.metrics.keys())
        metric_names = sorted(all_metrics)

        # Build table
        lines = []

        # Header
        header = ["Model"] + metric_names
        lines.append(" | ".join(f"{h:>15}" for h in header))
        lines.append("-" * len(lines[0]))

        # Rows
        for name, result in sorted(self.results.items()):
            row = [name]
            for metric in metric_names:
                value = result.metrics.get(metric, 0.0)
                row.append(f"{value:>15.4f}")
            lines.append(" | ".join(row))

        return "\n".join(lines)

    def best_model(self, metric: str = "loss", lower_is_better: bool = True) -> str:
        """Get the name of the best model for a given metric.

        Args:
            metric: Metric name to compare
            lower_is_better: If True, lower values are better

        Returns:
            Name of the best model
        """
        if not self.results:
            return ""

        best_name = None
        best_value = None

        for name, result in self.results.items():
            value = result.metrics.get(metric, float("inf") if lower_is_better else float("-inf"))
            if best_value is None:
                best_value = value
                best_name = name
            elif lower_is_better and value < best_value:
                best_value = value
                best_name = name
            elif not lower_is_better and value > best_value:
                best_value = value
                best_name = name

        return best_name or ""

    def get_samples(self, model_name: str) -> List[List[int]]:
        """Get generated samples for a model.

        Args:
            model_name: Name of the model

        Returns:
            List of generated token sequences
        """
        if model_name not in self.results:
            return []
        return self.results[model_name].generated_samples

    def summary(self) -> str:
        """Get a summary of the comparison.

        Returns:
            Summary string
        """
        if not self.results:
            return "No models evaluated yet."

        lines = [
            f"Model Comparison Summary ({len(self.results)} models)",
            "=" * 50,
        ]

        best_loss = self.best_model("loss", lower_is_better=True)
        best_pitch_sim = self.best_model("pitch_similarity", lower_is_better=False)
        best_interval_sim = self.best_model("interval_similarity", lower_is_better=False)

        lines.append(f"Best loss: {best_loss}")
        if best_pitch_sim:
            lines.append(f"Best pitch similarity: {best_pitch_sim}")
        if best_interval_sim:
            lines.append(f"Best interval similarity: {best_interval_sim}")

        lines.append("")
        lines.append(self.compare_table())

        return "\n".join(lines)
