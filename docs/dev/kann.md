# KANN Neural Network Integration Plan

This document outlines the plan for integrating neural network-based music learning and generation into coremusic using the KANN library wrapper.

## Overview

The goal is to provide a high-level API for training neural networks on MIDI data and generating new music. This builds on:

- `coremusic.kann` - Low-level KANN wrapper (NeuralNetwork, GraphBuilder, DataSet)
- `coremusic.objects` - MusicSequence, MusicTrack for MIDI I/O
- `coremusic.music.theory` - Music theory utilities (scales, chords, intervals)

## Architecture

```
coremusic.music.neural/
    __init__.py           # Public API exports
    models.py             # Pre-built model architectures
    data.py               # MIDI data preprocessing and encoding
    training.py           # Training utilities and callbacks
    generation.py         # Music generation strategies
    evaluation.py         # Model evaluation metrics
```

## 1. Data Representation (`data.py`)

### 1.1 MIDI Event Encoding Schemes

Different encoding schemes trade off between simplicity and expressiveness:

#### Option A: Note-Only Encoding (Simplest)
```python
class NoteEncoder:
    """Encode MIDI as sequence of note numbers (0-127)."""
    vocab_size = 128

    def encode(self, midi_events) -> List[int]:
        """Extract note-on events, return list of MIDI note numbers."""

    def decode(self, indices) -> List[MIDIEvent]:
        """Convert note indices back to MIDI events."""
```

**Pros**: Simple, small vocabulary
**Cons**: Loses timing, velocity, duration information

#### Option B: Piano Roll Encoding
```python
class PianoRollEncoder:
    """Encode MIDI as piano roll (time steps x 128 notes)."""

    def encode(self, midi_events, resolution=16) -> Array2D:
        """Convert to binary piano roll at given resolution (steps per beat)."""

    def decode(self, piano_roll, tempo=120) -> List[MIDIEvent]:
        """Convert piano roll back to MIDI events."""
```

**Pros**: Captures polyphony naturally
**Cons**: Large input size, sparse data

#### Option C: Event-Based Encoding (Most Expressive)
```python
class EventEncoder:
    """Encode MIDI as sequence of tokenized events."""

    # Token types:
    # - NOTE_ON_0 to NOTE_ON_127 (128 tokens)
    # - NOTE_OFF_0 to NOTE_OFF_127 (128 tokens)
    # - TIME_SHIFT_1 to TIME_SHIFT_100 (100 tokens for timing)
    # - VELOCITY_1 to VELOCITY_32 (32 velocity buckets)

    vocab_size = 128 + 128 + 100 + 32  # = 388

    def encode(self, midi_events) -> List[int]:
        """Convert MIDI events to token sequence."""

    def decode(self, tokens) -> List[MIDIEvent]:
        """Convert tokens back to MIDI events."""
```

**Pros**: Preserves timing, velocity, note-off
**Cons**: Larger vocabulary, longer sequences

#### Option D: Relative Pitch Encoding
```python
class RelativePitchEncoder:
    """Encode as intervals from previous note."""

    # Tokens: -24 to +24 semitones (49 interval tokens)
    # Plus special tokens: REST, HOLD, etc.
    vocab_size = 49 + 10  # intervals + special

    def encode(self, midi_events) -> List[int]:
        """Convert to relative pitch intervals."""
```

**Pros**: Transposition invariant, smaller vocab
**Cons**: Accumulates errors during generation

### 1.2 Data Pipeline

```python
class MIDIDataset:
    """Load and preprocess MIDI files for training."""

    def __init__(self, encoder: BaseEncoder, seq_length: int):
        self.encoder = encoder
        self.seq_length = seq_length
        self.sequences = []

    def load_file(self, path: str):
        """Load a single MIDI file."""

    def load_directory(self, path: str, pattern="*.mid"):
        """Load all MIDI files from directory."""

    def prepare_training_data(self) -> Tuple[Array2D, Array2D]:
        """
        Prepare X, Y arrays for training.
        X: input sequences (one-hot or embedding)
        Y: target (next token prediction)
        """

    def augment(self, transpose_range=(-6, 6), tempo_range=(0.8, 1.2)):
        """Data augmentation: transposition, tempo variation."""
```

## 2. Model Architectures (`models.py`)

### 2.1 Feedforward MLP (Baseline)

```python
def create_mlp_model(
    input_size: int,      # seq_length * vocab_size (flattened one-hot)
    hidden_sizes: List[int] = [512, 256],
    output_size: int = 128,
    dropout: float = 0.3
) -> NeuralNetwork:
    """
    Simple MLP for next-note prediction.

    Use case: Quick experiments, small datasets
    Limitation: No temporal awareness beyond fixed window
    """
```

### 2.2 Simple RNN

```python
def create_rnn_model(
    input_size: int,      # vocab_size (per timestep)
    hidden_size: int = 256,
    output_size: int = 128,
    rnn_flags: int = 0
) -> NeuralNetwork:
    """
    Vanilla RNN for sequence modeling.

    Use case: Learning short-term patterns
    Limitation: Vanishing gradients, limited memory
    """
```

### 2.3 LSTM Network

```python
def create_lstm_model(
    input_size: int,
    hidden_size: int = 512,
    num_layers: int = 2,
    output_size: int = 128,
    dropout: float = 0.3
) -> NeuralNetwork:
    """
    LSTM for long-term sequence modeling.

    Use case: Learning musical structure, phrases
    Strength: Better gradient flow, longer memory
    """
```

### 2.4 GRU Network

```python
def create_gru_model(
    input_size: int,
    hidden_size: int = 512,
    output_size: int = 128
) -> NeuralNetwork:
    """
    GRU - simplified LSTM alternative.

    Use case: Similar to LSTM, fewer parameters
    Strength: Faster training, comparable performance
    """
```

### 2.5 Stacked RNN with Attention (Advanced)

```python
def create_attention_model(
    input_size: int,
    hidden_size: int = 512,
    attention_size: int = 128,
    output_size: int = 128
) -> NeuralNetwork:
    """
    RNN with attention mechanism.

    Use case: Learning long-range dependencies
    Note: Requires custom graph building with GraphBuilder
    """
```

### 2.6 Model Factory

```python
class ModelFactory:
    """Factory for creating pre-configured models."""

    @staticmethod
    def create(
        model_type: str,  # 'mlp', 'rnn', 'lstm', 'gru'
        encoder: BaseEncoder,
        seq_length: int,
        **kwargs
    ) -> NeuralNetwork:
        """Create model appropriate for encoder and task."""
```

## 3. Training (`training.py`)

### 3.1 Training Configuration

```python
@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 100
    min_epochs: int = 10
    early_stopping_patience: int = 10
    validation_split: float = 0.1
    gradient_clip: float = 5.0

    # Learning rate schedule
    lr_decay: float = 0.95
    lr_decay_epochs: int = 10
```

### 3.2 Training Loop

```python
class Trainer:
    """High-level training interface."""

    def __init__(self, model: NeuralNetwork, config: TrainingConfig):
        self.model = model
        self.config = config
        self.history = {'loss': [], 'val_loss': []}

    def train(self, x_train, y_train, x_val=None, y_val=None):
        """
        Train the model with progress reporting.

        Returns training history.
        """

    def evaluate(self, x_test, y_test) -> Dict[str, float]:
        """Evaluate model on test set."""

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
```

### 3.3 Callbacks

```python
class Callback:
    """Base callback class."""
    def on_epoch_start(self, epoch): pass
    def on_epoch_end(self, epoch, logs): pass
    def on_train_end(self, logs): pass

class EarlyStopping(Callback):
    """Stop training when validation loss stops improving."""

class ModelCheckpoint(Callback):
    """Save best model during training."""

class LearningRateScheduler(Callback):
    """Adjust learning rate during training."""

class ProgressLogger(Callback):
    """Log training progress."""
```

## 4. Generation (`generation.py`)

### 4.1 Sampling Strategies

```python
class SamplingStrategy:
    """Base class for sampling strategies."""

    def sample(self, probabilities: array.array) -> int:
        """Sample next token from probability distribution."""
        raise NotImplementedError

class GreedySampling(SamplingStrategy):
    """Always pick highest probability token."""

class TemperatureSampling(SamplingStrategy):
    """Sample with temperature scaling."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

class TopKSampling(SamplingStrategy):
    """Sample from top-k most likely tokens."""

    def __init__(self, k: int = 10, temperature: float = 1.0):
        self.k = k
        self.temperature = temperature

class NucleusSampling(SamplingStrategy):
    """Sample from smallest set with cumulative prob >= p."""

    def __init__(self, p: float = 0.9, temperature: float = 1.0):
        self.p = p
        self.temperature = temperature
```

### 4.2 Generator

```python
class MusicGenerator:
    """Generate music using trained model."""

    def __init__(
        self,
        model: NeuralNetwork,
        encoder: BaseEncoder,
        sampling: SamplingStrategy = TemperatureSampling(1.0)
    ):
        self.model = model
        self.encoder = encoder
        self.sampling = sampling

    def generate(
        self,
        seed: Optional[List[int]] = None,
        length: int = 100,
        temperature: float = 1.0
    ) -> List[int]:
        """
        Generate a sequence of tokens.

        Args:
            seed: Starting sequence (uses random if None)
            length: Number of tokens to generate
            temperature: Sampling temperature

        Returns:
            List of generated token indices
        """

    def generate_midi(
        self,
        seed_midi: Optional[str] = None,
        duration_beats: int = 32,
        tempo: int = 120
    ) -> MusicSequence:
        """
        Generate MIDI output.

        Args:
            seed_midi: Path to seed MIDI file
            duration_beats: Length of output in beats
            tempo: Output tempo

        Returns:
            MusicSequence object ready for playback/export
        """

    def continue_sequence(
        self,
        midi_path: str,
        continuation_length: int = 16
    ) -> MusicSequence:
        """Continue an existing MIDI file."""
```

### 4.3 Constrained Generation

```python
class ConstrainedGenerator(MusicGenerator):
    """Generate music with musical constraints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraints = []

    def add_scale_constraint(self, scale: str, root: int):
        """Constrain notes to a specific scale."""

    def add_chord_constraint(self, chord_progression: List[str]):
        """Constrain to follow chord progression."""

    def add_rhythm_constraint(self, pattern: List[float]):
        """Constrain to rhythmic pattern."""

    def generate(self, *args, **kwargs):
        """Generate with constraints applied."""
```

## 5. Evaluation (`evaluation.py`)

### 5.1 Metrics

```python
class MusicMetrics:
    """Evaluate generated music quality."""

    @staticmethod
    def pitch_histogram_similarity(
        generated: List[int],
        reference: List[int]
    ) -> float:
        """Compare pitch distributions."""

    @staticmethod
    def interval_histogram_similarity(
        generated: List[int],
        reference: List[int]
    ) -> float:
        """Compare interval distributions."""

    @staticmethod
    def note_density(events: List[MIDIEvent], duration: float) -> float:
        """Calculate notes per second."""

    @staticmethod
    def pitch_range(events: List[MIDIEvent]) -> Tuple[int, int]:
        """Get min and max pitch."""

    @staticmethod
    def repetition_score(sequence: List[int], n: int = 4) -> float:
        """Measure n-gram repetition (0=no repetition, 1=all repeated)."""

    @staticmethod
    def self_similarity_matrix(sequence: List[int]) -> Array2D:
        """Compute self-similarity matrix for structure analysis."""
```

### 5.2 Comparison Tools

```python
class ModelComparison:
    """Compare different models on same dataset."""

    def __init__(self, test_data: MIDIDataset):
        self.test_data = test_data
        self.results = {}

    def evaluate_model(self, name: str, model: NeuralNetwork):
        """Evaluate a model and store results."""

    def compare(self) -> pd.DataFrame:
        """Return comparison table."""

    def plot_comparison(self, metric: str):
        """Plot comparison chart."""
```

## 6. High-Level API

### 6.1 Simple Interface

```python
# In coremusic.music.neural/__init__.py

def train_music_model(
    midi_files: List[str],
    model_type: str = 'lstm',
    output_path: str = 'model.kan',
    **kwargs
) -> NeuralNetwork:
    """
    One-line training interface.

    Example:
        model = train_music_model(
            ['bach/*.mid', 'beethoven/*.mid'],
            model_type='lstm',
            hidden_size=512,
            epochs=100
        )
    """

def generate_music(
    model_path: str,
    output_path: str,
    duration: int = 32,
    temperature: float = 1.0,
    seed_midi: Optional[str] = None
) -> str:
    """
    One-line generation interface.

    Example:
        generate_music(
            'model.kan',
            'output.mid',
            duration=64,
            temperature=0.8
        )
    """

def continue_music(
    model_path: str,
    input_midi: str,
    output_path: str,
    bars: int = 8
) -> str:
    """Continue an existing MIDI file."""
```

### 6.2 CLI Integration

```bash
# Training
coremusic train --model lstm --input ./midi_files/ --output model.kan --epochs 100

# Generation
coremusic generate --model model.kan --output generated.mid --length 64 --temperature 0.8

# Continuation
coremusic continue --model model.kan --input seed.mid --output continued.mid --bars 8

# Evaluation
coremusic evaluate --model model.kan --test ./test_midi/
```

## 7. Implementation Phases

### Phase 1: Core Data Pipeline [COMPLETE]
- [x] Implement `NoteEncoder` (simplest encoding)
- [x] Implement `EventEncoder` (most expressive encoding)
- [x] Implement `PianoRollEncoder` (polyphonic encoding)
- [x] Implement `RelativePitchEncoder` (transposition-invariant encoding)
- [x] Implement `MIDIDataset` for loading and preprocessing
- [x] Add data augmentation (transposition)
- [x] Unit tests for encoding/decoding roundtrip

### Phase 2: Basic Models [COMPLETE]
- [x] Implement `create_mlp_model`
- [x] Implement `create_lstm_model`
- [x] Implement `create_gru_model`
- [x] Implement `create_rnn_model`
- [x] Implement `ModelFactory` class
- [x] Add model save/load utilities (via `NeuralNetwork.save()`/`load()`)

### Phase 3: Training Infrastructure [COMPLETE]
- [x] Implement `TrainingConfig` dataclass
- [x] Implement `Trainer` class with train/evaluate methods
- [x] Implement `Callback` base class and system
- [x] Implement `EarlyStopping` callback
- [x] Implement `ModelCheckpoint` callback
- [x] Implement `ProgressLogger` callback
- [x] Implement `LearningRateScheduler` callback
- [x] Integration tests with Trainer and callbacks

### Phase 4: Generation [COMPLETE]
- [x] Implement `SamplingStrategy` base class
- [x] Implement `GreedySampling`
- [x] Implement `TemperatureSampling`
- [x] Implement `TopKSampling`
- [x] Implement `NucleusSampling` (top-p)
- [x] Implement `MusicGenerator` class
- [x] Add `generate()` for token sequences
- [x] Add `generate_midi()` for MIDI output
- [x] Add `continue_sequence()` for MIDI continuation
- [x] Add `generate_variations()` for multiple outputs
- [x] Full pipeline integration tests

### Phase 5: Advanced Features [COMPLETE]
- [x] Implement `EventEncoder` (full expressiveness)
- [x] Add evaluation metrics (`MusicMetrics` class)
- [x] Add model comparison tools (`ModelComparison` class)
- [ ] Add constrained generation (future enhancement)

### Phase 6: Polish [COMPLETE]
- [x] CLI integration (`coremusic neural` commands)
- [x] High-level API functions (`train_music_model`, `generate_music`, `continue_music`)
- [x] Comprehensive test suite (117 tests for neural module)
- [ ] Documentation and examples (future enhancement)
- [ ] Performance optimization (future enhancement)
- [ ] User guide with tutorials (future enhancement)

## 8. Example Usage

### Training on Pachelbel's Canon

```python
import coremusic as cm
from coremusic.music.neural import (
    MIDIDataset, NoteEncoder, create_lstm_model,
    Trainer, TrainingConfig, MusicGenerator, TemperatureSampling
)

# 1. Load and encode data
encoder = NoteEncoder()
dataset = MIDIDataset(encoder, seq_length=32)
dataset.load_file('tests/canon.mid')
dataset.augment(transpose_range=(-5, 7))  # Data augmentation

x_train, y_train = dataset.prepare_training_data()
print(f"Training samples: {x_train.rows}")

# 2. Create model
model = create_lstm_model(
    input_size=encoder.vocab_size,
    hidden_size=256,
    output_size=encoder.vocab_size
)

# 3. Train
config = TrainingConfig(
    learning_rate=0.001,
    batch_size=32,
    max_epochs=100,
    early_stopping_patience=10
)

trainer = Trainer(model, config)
history = trainer.train(x_train, y_train)

# 4. Save model
model.save('canon_lstm.kan')

# 5. Generate new music
generator = MusicGenerator(
    model=model,
    encoder=encoder,
    sampling=TemperatureSampling(temperature=0.8)
)

# Generate 64 beats of new music
sequence = generator.generate_midi(duration_beats=64, tempo=80)
sequence.save('generated_canon.mid')

# Or continue from seed
continuation = generator.continue_sequence(
    'tests/canon.mid',
    continuation_length=32
)
continuation.save('canon_continued.mid')
```

### Comparing Model Architectures

```python
from coremusic.music.neural import (
    create_mlp_model, create_rnn_model, create_lstm_model, create_gru_model,
    ModelComparison
)

# Load test data
test_dataset = MIDIDataset(encoder, seq_length=32)
test_dataset.load_file('tests/canon.mid')

# Create models
models = {
    'MLP': create_mlp_model(input_size=32*128, hidden_sizes=[256], output_size=128),
    'RNN': create_rnn_model(input_size=128, hidden_size=256, output_size=128),
    'LSTM': create_lstm_model(input_size=128, hidden_size=256, output_size=128),
    'GRU': create_gru_model(input_size=128, hidden_size=256, output_size=128),
}

# Train and compare
comparison = ModelComparison(test_dataset)
for name, model in models.items():
    # Train each model...
    comparison.evaluate_model(name, model)

# Print results
print(comparison.compare())
```

## 9. References

- KANN Library: https://github.com/attractivechaos/kann
- Music Transformer (Huang et al.): https://arxiv.org/abs/1809.04281
- MusicVAE (Roberts et al.): https://arxiv.org/abs/1803.05428
- MuseNet (OpenAI): https://openai.com/blog/musenet/
