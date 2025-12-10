# CHANGELOG

All notable project-wide changes will be documented in this file. Note that each subproject has its own CHANGELOG.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Commons Changelog](https://common-changelog.org). This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Types of Changes

- Added: for new features.
- Changed: for changes in existing functionality.
- Deprecated: for soon-to-be removed features.
- Removed: for now removed features.
- Fixed: for any bug fixes.
- Security: in case of vulnerabilities.

---

## [Unreleased]

### Removed

- **Neural Module** - Removed `coremusic.music.neural` subpackage
  - Out of scope for the core package
  - Included: `api.py`, `data.py`, `evaluation.py`, `generation.py`, `models.py`, `training.py`
  - Associated tests removed: `test_music_neural.py`, `test_neural_training_classical.py`
  - Demo scripts removed: `tests/demos/neural/`
  - Keeping the package lean and fit-for-purpose

- **Generative Algorithms** - Moved from package to examples
  - `coremusic.music.generative` module removed from package
  - `coremusic.music.markov` module removed from package
  - `coremusic.music.bayes` module removed from package
  - Code relocated to `tests/examples/` directory for reference
  - Associated tests removed: `test_music_generative.py`, `test_music_markov.py`, `test_music_bayes.py`

- **DAW Integration** - Removed `coremusic.daw` module
  - Moved to `tests/examples/` directory
  - Associated tests removed: `test_daw.py`

- **CLI Commands** - Removed generative CLI commands
  - `coremusic generate` command removed
  - `coremusic neural` command removed

## [0.1.10]

### Changed

- **Improved Code Quality** - Comprehensive lint and type checking fixes
  - Fixed all ruff/flake8 lint errors without using noqa suppressions
  - Converted star imports (`from x import *`) to explicit imports in `__init__.py` files
  - Fixed E741 ambiguous variable names (`l` -> `lyr`)
  - Fixed E721 type comparisons (use `is` instead of `==` for type checks)
  - Fixed F841 unused variable assignments
  - All modules now pass `make lint` and `make typecheck`

### Removed

- **Test Suite Cleanup** - Removed 4 low-quality/redundant test files (20 tests)
  - `test_fourcharcode.py` - Tested Python builtins, not coremusic functionality
  - `test_coverage_improvements.py` - Overly broad exception tests with no specific assertions
  - `test_audiotoolbox_audio_queue.py` - Trivial assertions (only checked `is not None`)
  - `test_audiounit.py` - No behavior verification, just type checks
  - Test count: 1699 -> 1679 (all remaining tests pass)

### Added

- **Command Line Interface** - Comprehensive CLI for audio and MIDI operations (`coremusic.cli`)
  - `coremusic audio` - Audio file operations (info, duration, metadata)
  - `coremusic devices` - Audio device management (list, default, info)
  - `coremusic plugins` - AudioUnit plugin discovery (list, find, info, params)
  - `coremusic analyze` - Audio analysis (peak, rms, silence, tempo, spectrum, key, mfcc)
  - `coremusic convert` - Format conversion (file, batch)
  - `coremusic midi` - MIDI device discovery (devices, inputs, outputs, send, file)
  - `coremusic generate` - Generative algorithms (arpeggio, euclidean, melody)
    - Transform support: `--transform`/`-t` to apply transforms (humanize, reverse, arpeggiate, quantize, velocity_scale)
    - Tempo option: `--bpm`/`-b` (renamed from `--tempo`/`-t`)
  - `coremusic sequence` - MIDI sequence operations (info, play, tracks)
  - JSON output support (`--json`) for scripting integration

- **Bit Shift Register Generator** - Gate-based sequencing with variable velocity and duration (`coremusic.music.generative`)
  - `BitShiftRegister` - Core 8-bit shift register with configurable feedback taps
    - Left/right shift operations with XOR feedback
    - Rotate operations preserving bit state
    - Clock-based operation with gate output
    - Seed configuration for reproducible sequences
  - `BitShiftRegisterConfig` - Configuration for the generator
    - Variable velocity modes: fixed, random, pattern-based
    - Variable duration modes: fixed, random, pattern-based
    - Configurable note, channel, and base parameters
  - `BitShiftRegisterGenerator` - Full MIDI event generator
    - Step-based generation with clock advancement
    - MIDI file output integration
    - Humanization and swing support
  - 47 tests covering all functionality

- **Bayesian Network MIDI Analysis** - Probabilistic modeling of note dependencies (`coremusic.music.bayes`)
  - `BayesianNetwork` - Core Bayesian network implementation
    - Configurable network structure (fixed, learned, or manual)
    - Directed acyclic graph with cycle detection
    - Conditional probability tables (CPT) with Laplace smoothing
    - Topological sampling (ancestral sampling)
  - `NetworkConfig` - Comprehensive configuration dataclass
    - Network modes: pitch-only, pitch+duration, pitch+duration+velocity, full (with IOI)
    - Structure modes: fixed, learned, manual
    - Configurable temporal order (1st, 2nd, higher)
    - Discretization bins for each variable
  - **Network Structure**
    - `add_variable()`, `add_edge()` - Build custom network structures
    - `remove_variable()`, `remove_edge()` - Modify structures
    - Automatic cycle detection
  - **Conditional Probability Tables**
    - `CPT` class for storing P(variable | parents)
    - Laplace smoothing for unseen observations
    - Sampling and entropy calculation
  - `MIDIBayesAnalyzer` - MIDI file analysis
    - `analyze_file()` - Create network from MIDI file
    - `analyze_track()` - Analyze specific track
    - `analyze_all_tracks()` - Create networks for all tracks
  - `MIDIBayesGenerator` - MIDI generation from networks
    - `generate()` - Generate new MIDI sequence
    - `generate_to_track()` - Add generated notes to existing track
    - Start pitch control for deterministic beginnings
  - **Utility Functions**
    - `analyze_and_generate()` - One-step analysis and generation
    - `merge_networks()` - Combine multiple networks
    - `network_statistics()` - Get network metrics
  - 66 comprehensive tests covering all functionality

- **Markov Chain MIDI Analysis** - Advanced MIDI file analysis and generation using Markov chains (`coremusic.music.markov`)
  - `MarkovChain` - Core Markov chain implementation
    - Configurable order (1st, 2nd, higher-order chains)
    - Training from note sequences
    - Probability matrix with transition sampling
    - JSON serialization for saving/loading models
  - `ChainConfig` - Comprehensive configuration dataclass
    - Modeling modes: pitch-only, pitch+duration, pitch+duration+velocity
    - Rhythm modes: constant, Markov-based, external generator
    - Temperature control for sampling randomness
    - Note range clamping (min/max MIDI notes)
    - Gravity weights for biasing toward specific notes
    - Probability smoothing (Laplace smoothing)
  - **Node-Edge Editing** - Granular transition manipulation
    - `set_transition_probability()` - Set specific transition weights
    - `remove_transition()` - Remove specific transitions
    - `get_transition_probability()` - Query transition weights
    - `get_transitions_from()` - Get all transitions from a state
  - **Chain-Scope Adjustments** - Global chain modifications
    - `set_temperature()` - Control sampling randomness
    - `set_note_range()` - Clamp output to MIDI range
    - `set_gravity()` - Bias toward specific notes
    - `sparsify()` - Remove low-probability transitions
  - `MIDIMarkovAnalyzer` - MIDI file analysis
    - `analyze_file()` - Create chain from MIDI file
    - `analyze_track()` - Analyze specific track
    - `analyze_all_tracks()` - Create chains for all tracks
  - `MIDIMarkovGenerator` - MIDI generation from chains
    - `generate()` - Generate new MIDI sequence
    - `generate_to_track()` - Add generated notes to existing track
    - Start pitch control for deterministic beginnings
  - **Utility Functions**
    - `analyze_and_generate()` - One-step analysis and generation
    - `merge_chains()` - Combine multiple chains
    - `chain_statistics()` - Get chain metrics (states, transitions, entropy)
  - 64 comprehensive tests covering all functionality

- **Music Theory and Generative Module** - Complete music theory foundations and MIDI-enabled generative algorithms (`coremusic.music`)
  - **Music Theory Primitives** (`src/coremusic/music/theory.py`)
    - `Note` class with MIDI number conversion, transposition, frequency calculation (A4=440Hz)
    - `Interval` class with standard intervals (unison through compound intervals)
    - `Scale` class with 25+ scale types:
      - Diatonic: major, natural/harmonic/melodic minor
      - Modes: dorian, phrygian, lydian, mixolydian, locrian
      - Pentatonic: major, minor, blues major/minor
      - Jazz: bebop major/dominant/minor, whole tone, diminished
      - World: harmonic major, double harmonic, hungarian minor, neapolitan
      - Exotic: hirajoshi, in-sen, iwato, pelog
    - `Chord` class with 35+ chord types:
      - Triads: major, minor, diminished, augmented, sus2, sus4
      - 7ths: dominant7, major7, minor7, diminished7, half-diminished7, minorMajor7
      - Extended: 9th, 11th, 13th variants
      - Altered: 7b5, 7#5, 7b9, 7#9, 7#11
      - Added tone: add9, add11, 6, minor6
    - `ChordProgression` class with Roman numeral parsing (I, ii, IV, V7, etc.)
    - Enharmonic note handling (flats normalized to sharps internally)
  - **Generative Algorithms** (`src/coremusic/music/generative.py`)
    - `Arpeggiator` with 10 pattern modes:
      - UP, DOWN, UP_DOWN, DOWN_UP, RANDOM, RANDOM_WALK
      - OUTSIDE_IN, INSIDE_OUT, CHORD, AS_PLAYED
    - `EuclideanGenerator` using Bjorklund's algorithm for mathematical rhythm patterns
      - Classic patterns: tresillo (3,8), cinquillo (5,8), rumba (7,12)
    - `MarkovGenerator` for probabilistic note sequences
      - Training from note sequences
      - Scale constraint for harmonic coherence
    - `ProbabilisticGenerator` for weighted random note selection
      - Custom note weights for biased selection
      - Rest probability for rhythmic variety
    - `SequenceGenerator` (step sequencer)
      - Per-step note, velocity, gate length
      - Step probability for variation
    - `MelodyGenerator` for rule-based melodic phrases
      - Scale-constrained motion with configurable step size
      - Rest insertion and phrase structure
    - `PolyrhythmGenerator` for layered polyrhythmic patterns
      - Multiple independent rhythmic layers
      - Cross-rhythm pattern generation
    - Common features across generators:
      - Swing timing (0.0-1.0)
      - Humanization for timing and velocity
      - Reproducible results via seed parameter
  - **MIDI Integration**
    - All generators output `MIDIEvent` objects compatible with `coremusic.midi.utilities`
    - Direct integration with `MIDITrack` and `MIDISequence` for file export
    - Utility functions: `create_arp_from_progression()`, `combine_generators()`
  - **Test Coverage**
    - 80 tests in `tests/test_music_theory.py` (Note, Interval, Scale, Chord, ChordProgression)
    - 79 tests in `tests/test_music_generative.py` (all generators + MIDI file generation)
    - 19 MIDI file generation tests creating actual `.mid` files in `build/midi_files/`
  - **Generated MIDI Files** (`build/midi_files/`)
    - Arpeggiator demos: up, up-down, random patterns
    - Euclidean rhythms: tresillo (3,8), cinquillo (5,8), 7/12
    - Markov melodies: trained and pentatonic-constrained
    - Probabilistic: Dorian scale, weighted selection
    - Sequences: drum patterns, melodic sequences
    - Melodies: major scale, blues scale
    - Polyrhythms: 3:4, 5:4:3
    - Combined: arp+drums, chord progression, full composition

  **Example Usage:**

  ```python
  import coremusic as cm
  from coremusic.music.theory import Note, Scale, ScaleType, Chord, ChordType
  from coremusic.music.generative import Arpeggiator, ArpPattern, EuclideanGenerator

  # Music Theory
  c4 = Note.from_name("C4")
  print(f"MIDI: {c4.midi}, Frequency: {c4.frequency:.2f} Hz")

  c_major = Scale(Note.from_name("C4"), ScaleType.MAJOR)
  print(f"C Major: {[str(n) for n in c_major.notes]}")

  cm_chord = Chord(Note.from_name("C4"), ChordType.MAJOR_7)
  print(f"Cmaj7: {[str(n) for n in cm_chord.notes]}")

  # Arpeggiator
  chord = Chord(Note.from_name("C4"), ChordType.MAJOR)
  arp = Arpeggiator(chord.notes, pattern=ArpPattern.UP_DOWN, note_duration=0.25)
  events = arp.generate(num_notes=8)

  # Euclidean Rhythms
  euclidean = EuclideanGenerator(
      pulses=5, steps=8,  # Cinquillo pattern
      note=Note.from_name("C4"),
      step_duration=0.125
  )
  events = euclidean.generate(num_cycles=2)

  # Export to MIDI file
  from coremusic.midi.utilities import MIDISequence, MIDITrack
  sequence = MIDISequence()
  track = sequence.create_track("Arpeggio")
  for event in events:
      if event.status == MIDIStatus.NOTE_ON:
          track.add_note(event.time, event.data1, event.data2, 0.2)
  sequence.save("arpeggio.mid")
  ```

- **matplotlib as dev dependency** - Enables audio visualization tests (`tests/test_audio_visualization.py`)

- **MIDI Transformation Pipeline** - Composable pipeline for analyzing and transforming MIDI files (`coremusic.midi.transform`)
  - **Base Classes**
    - `MIDITransformer` - Abstract base class for all transformers
    - `Pipeline` - Chain of transformers applied in sequence with fluent API
  - **Pitch Transformers**
    - `Transpose` - Shift notes by semitones (with clamping to 0-127)
    - `Invert` - Mirror melody around a pivot note (retrograde inversion)
    - `Harmonize` - Add parallel intervals (thirds, fifths, triads, etc.)
  - **Time Transformers**
    - `Quantize` - Snap timing to grid with configurable strength and swing
    - `TimeStretch` - Speed up or slow down (tempo change)
    - `TimeShift` - Move events forward/backward in time
    - `Reverse` - Retrograde (reverse note order preserving durations)
  - **Velocity Transformers**
    - `VelocityScale` - Scale by factor or compress to min/max range
    - `VelocityCurve` - Apply curves (linear, log, exp, soft, hard, custom)
    - `Humanize` - Add human-like timing and velocity variation
  - **Filter Transformers**
    - `NoteFilter` - Filter by pitch range, velocity, channel (with invert option)
    - `ScaleFilter` - Filter notes to only allow those in a given musical scale (scale mask)
    - `EventTypeFilter` - Keep or remove specific MIDI event types
  - **Track Transformers**
    - `ChannelRemap` - Remap MIDI channels
    - `TrackMerge` - Merge all tracks into a single track
    - `Arpeggiate` - Convert chords to arpeggios (up, down, up_down, random patterns)
  - **Convenience Functions**
    - `transpose()`, `quantize()`, `humanize()`, `reverse()`, `scale_velocity()`, `filter_to_scale()`
  - **Test Coverage**
    - 79 tests in `tests/test_midi_transform.py`
    - Integration tests generating MIDI files in `build/midi_files/transform_tests/`
  - **Generated MIDI Files** (`build/midi_files/transform_tests/`)
    - Pre/post pairs for each transformation: `<name>_pre.mid` and `<name>_post.mid`
    - Transformations: transposed, quantized, humanized, harmonized, arpeggiated, reversed, inverted, velocity curved, time stretched, pipeline (combined)

- **Audio Slicing File Generation Tests** - Integration tests demonstrating audio transformations (`tests/test_audio_slicing.py`)
  - **Generated Audio Files** (`build/audio_files/slicing_tests/`)
    - Pre/post pairs for each transformation: `<name>_pre.wav` and `<name>_post.wav`
    - Transformations: shuffled, reversed, pattern, repeated, filtered, sorted by duration, normalized
  - Uses `scipy.io.wavfile` for WAV file output

  **Example Usage:**

  ```python
  from coremusic.midi.utilities import MIDISequence
  from coremusic.midi.transform import Pipeline, Transpose, Quantize, Humanize, VelocityScale

  # Load MIDI file
  seq = MIDISequence.load("input.mid")

  # Create transformation pipeline
  pipeline = Pipeline([
      Transpose(semitones=5),              # Up a perfect fourth
      Quantize(grid=0.125, strength=0.8),  # Quantize to 16th notes
      VelocityScale(min_vel=40, max_vel=100),  # Compress velocity range
      Humanize(timing=0.02, velocity=10),  # Add human feel
  ])

  # Apply transformations and save
  transformed = pipeline.apply(seq)
  transformed.save("output.mid")
  ```

### Changed

- **Demo Files Reorganization** - Restructured `tests/demos/` for better organization and usability
  - **Split monolithic demos into focused single-purpose files** organized by category:
    - `analysis/` - Audio analysis (file_info, peak_rms, silence_detection)
    - `audiounit/` - AudioUnit plugins (list_plugins, plugin_info, parameter_control, factory_presets, discover_plugins, stream_format)
    - `conversion/` - Format conversion (stereo_to_mono, format_presets)
    - `devices/` - Audio device management (list_devices, default_devices, find_device)
    - `effects/` - Audio effects (create_chain, find_by_name, fourcc_reference)
    - `link/` - Ableton Link (session, beat_tracking)
    - `midi/` - MIDI (create_sequence, multi_track, routing)
    - `numpy/` - NumPy integration (read_audio, channel_analysis, format_dtypes)
    - `slicing/` - Audio slicing (onset_slicing, grid_slicing, recombine)
    - `streaming/` - Real-time streaming (input_stream, output_stream, latency_comparison)
    - `visualization/` - Audio visualization (waveform, spectrogram, spectrum)
  - **Reduced print() noise** - Demos now only print results, not verbose logging
  - **Removed sys.path.insert** - All demos now run cleanly with `uv run python`
  - **Updated README.md** - Usage examples now use `uv run python` commands
  - **Improved daw.py audio quality** - Synthesized sounds now sound musical:
    - Punchy electronic drums with pitch-swept kick, layered snare, crisp hi-hats
    - A minor bass line with warm sub-harmonic tones
    - Lush ambient pad with Am-F-C-G chord progression and detuned oscillators
    - Expressive vocal lead melody with vibrato in A minor
    - Clair de Lune-inspired piano melody in Db major
    - Jazz chord progression (Dm9-G7-Cmaj7-Am7) for MIDI instruments
    - E minor arpeggio pattern with bell-like tones for effects demo

- **Constants Export** - All constant enum classes from `coremusic.constants` are now exported directly from the main `coremusic` package for convenience:
  ```python
  # Now you can do:
  import coremusic as cm
  cm.AudioFileProperty.DATA_FORMAT
  cm.AudioFormatID.LINEAR_PCM

  # Instead of:
  from coremusic.constants import AudioFileProperty
  ```

### Deprecated

- **Legacy Constant Getter Functions** - The `get_*` functions in `coremusic.capi` (e.g., `get_audio_format_linear_pcm()`) are now deprecated in favor of the enum classes in `coremusic.constants`. The getter functions remain for backward compatibility but new code should use the enum classes:
  ```python
  # Deprecated:
  capi.get_audio_file_property_data_format()

  # Preferred:
  from coremusic import AudioFileProperty
  AudioFileProperty.DATA_FORMAT
  ```

### Fixed

- **MIDI File Save/Load Bug** - Fixed two critical bugs in MIDI file I/O (`coremusic.midi.utilities`):
  - **Track count mismatch**: Format 1 MIDI files now correctly report track count (including tempo track)
  - **Meta event parsing**: Track names and other meta events (0xFF) are now correctly parsed during load
  - Track names set via `track.name` are now properly preserved through save/load cycles

- **Docstring Typos** - Fixed 38 instances of malformed docstrings in `capi.pyx` where `"Returns:\n      f OSStatus result code"` has been corrected to `"Returns:\n        OSStatus result code (0 on success)"`.

### Added

- **CODE_REVIEW.md** - Comprehensive code review report covering architecture, API design, test coverage, and recommendations.

### Documentation

- **Enhanced Module Docstring** - The main `coremusic/__init__.py` now includes comprehensive documentation:
  - Basic usage examples with `AudioFile` context manager
  - Async/await support with `AsyncAudioFile` and `AsyncAudioQueue` examples
  - NumPy integration guide with `NUMPY_AVAILABLE` flag and usage patterns
  - Module organization overview
- **Async Classes Exported** - `AsyncAudioFile`, `AsyncAudioQueue`, `open_audio_file_async`, and `create_output_queue_async` are now included in `__all__` for better discoverability.
- **Async Audio Tutorial** (`docs/tutorials/async_audio.rst`) - Complete guide covering:
  - Async file operations with `AsyncAudioFile`
  - Streaming audio chunks asynchronously
  - Concurrent file processing with `asyncio.gather()`
  - Producer-consumer patterns for streaming
  - Integration with web frameworks (FastAPI example)
- **API Quickstart Guide** (`docs/api/quickstart.rst`) - Rapid introduction covering:
  - Import patterns for OO and functional APIs
  - Audio file, AudioUnit, and MIDI operations
  - Constants usage with enum classes
  - Async operations and error handling
  - NumPy integration and quick reference table
- **Common Patterns Cookbook** (`docs/cookbook/common_patterns.rst`) - Essential patterns including:
  - Resource management (context managers, multiple resources)
  - Error handling (graceful recovery, retry patterns)
  - Format handling (detection, validation, conversion pipelines)
  - Streaming patterns (generators, progress tracking)
  - Caching patterns (LRU cache, file hash cache)
  - Batch processing (parallel and sequential)

## [0.1.9]

### Added

- **Performance Optimizations Suite** - Complete infrastructure for high-performance audio processing (January 2025)
  - **Memory-Mapped File Access** (`src/coremusic/audio/mmap_file.py`)
    - `MMapAudioFile` class for fast random access to large audio files without loading into RAM
    - Support for WAV and AIFF format parsing with zero-copy access
    - NumPy integration with zero-copy when possible via `read_as_numpy()`
    - Array-like indexing support (`file[100:200]`) for intuitive frame access
    - Properties: `format`, `frame_count`, `duration`, `sample_rate`, `channels`
    - Context manager support for automatic resource cleanup
    - Lazy format parsing - only reads metadata when needed
    - 19 comprehensive tests in `tests/test_mmap_file.py` (100% passing)
  - **Buffer Pooling System** (`src/coremusic/audio/buffer_pool.py`)
    - `BufferPool` class for thread-safe buffer reuse to reduce allocation overhead
    - `PooledBuffer` context manager for automatic buffer acquisition and release
    - Statistics tracking (cache hits, misses, hit rate, outstanding buffers)
    - Global pool management with `get_global_pool()` and `reset_global_pool()`
    - Configurable max buffers per size with LRU eviction
    - `BufferPoolStats` class for detailed performance monitoring
    - Fixed critical deadlock bugs in stats property and summary method
    - 23 comprehensive tests in `tests/test_buffer_pool.py` (100% passing)
  - **Cython Performance Optimizations** (consolidated into `src/coremusic/capi.pyx`)
    - High-performance audio operations with typed memoryviews (`float32_t[:, ::1]`)
    - GIL release with `nogil` for parallel processing capabilities
    - Zero-overhead inline utility functions (`clip_float32`, `db_to_linear`, `linear_to_db`)
    - Compiler directives for maximum performance (`boundscheck=False`, `wraparound=False`, `cdivision=True`)
    - **Normalization Functions**:
      - `normalize_audio()` / `normalize_audio_float32()` - Peak normalization with target level
    - **Gain Functions**:
      - `apply_gain()` / `apply_gain_float32()` - dB-based gain adjustment
    - **Signal Analysis**:
      - `calculate_rms()` / `calculate_rms_float32()` - RMS level calculation
      - `calculate_peak()` / `calculate_peak_float32()` - Peak amplitude detection
    - **Format Conversions**:
      - `convert_float32_to_int16()` - Float to 16-bit integer with clipping
      - `convert_int16_to_float32()` - Integer to float normalization
      - `stereo_to_mono_float32()` - Stereo to mono downmixing (average)
      - `mono_to_stereo_float32()` - Mono to stereo upmixing (duplicate)
    - **Audio Mixing**:
      - `mix_audio_float32()` - Mix two audio signals with configurable ratio
    - **Fade Effects**:
      - `apply_fade_in_float32()` - Linear fade-in with configurable duration
      - `apply_fade_out_float32()` - Linear fade-out with configurable duration
    - 22 comprehensive tests in `tests/test_cython_ops.py` (100% passing)
    - Performance test verifies < 100ms for 10 seconds of 44.1kHz stereo audio
  - **Benchmarking Suite** (`benchmarks/bench_performance.py`)
    - Comprehensive benchmark infrastructure for performance measurement
    - Benchmarks for AudioFile, MMapAudioFile, BufferPool, and Cython operations
    - Statistics collection (mean, median, standard deviation)
    - Warmup runs to stabilize measurements
    - Multiple iterations with outlier detection
    - Configurable file paths and iteration counts
  - **Integration with Audio Package**
    - All Cython optimizations exported from `coremusic.audio` module
    - `CYTHON_OPS_AVAILABLE` flag for runtime feature detection
    - Backward compatible - existing code continues to work
    - Zero-copy operations when possible for maximum performance
  - **Total Test Count**: 1234 tests passing (1170 existing + 64 new performance tests)
  - **Zero Test Regressions**: All existing functionality preserved

  **Performance Benefits:**
  - Memory-mapped files: Fast random access without loading entire file into memory
  - Buffer pooling: Reduced allocation overhead through buffer reuse
  - Cython optimizations: 10-100x speedup for common audio operations vs pure Python
  - GIL release: Enables parallel processing and concurrent operations

  **Example Usage:**

  ```python
  import coremusic as cm
  import numpy as np

  # Memory-mapped file access (fast random access)
  with cm.MMapAudioFile("large_file.wav") as mmap_file:
      # Fast random frame access without loading entire file
      chunk = mmap_file[1000:2000]  # Read frames 1000-2000

      # Zero-copy NumPy access when possible
      audio_np = mmap_file.read_as_numpy(start_frame=0, num_frames=44100)

      print(f"Duration: {mmap_file.duration:.2f}s")
      print(f"Format: {mmap_file.format}")

  # Buffer pooling (efficient memory management)
  from coremusic.audio import BufferPool, get_global_pool

  # Use global pool
  with get_global_pool().acquire(size=4096) as buffer:
      # Use buffer for audio processing
      # Automatically returned to pool when done
      pass

  # Or create custom pool
  pool = BufferPool(max_buffers_per_size=10)
  with pool.acquire(size=8192) as buffer:
      process_audio(buffer)

  # Check pool statistics
  stats = pool.stats
  print(f"Hit rate: {stats['hit_rate']:.1%}")
  print(f"Outstanding: {stats['outstanding']}")

  # Cython-optimized operations (10-100x faster)
  audio = np.random.randn(44100, 2).astype(np.float32)

  # Normalize audio (very fast)
  normalized = cm.normalize_audio(audio, target_peak=0.9)

  # Apply gain in dB
  gained = cm.apply_gain(audio, gain_db=6.0)

  # Calculate signal metrics
  rms = cm.calculate_rms(audio)
  peak = cm.calculate_peak(audio)

  # Mix two signals
  output = np.zeros_like(audio)
  cm.mix_audio_float32(output, audio, other_audio, mix_ratio=0.5)

  # Apply fades
  cm.apply_fade_in_float32(audio, fade_frames=2205)  # 50ms at 44.1kHz
  cm.apply_fade_out_float32(audio, fade_frames=2205)

  # Format conversions
  int16_data = np.zeros((44100, 2), dtype=np.int16)
  cm.convert_float32_to_int16(audio, int16_data)

  # Channel conversions
  mono = np.zeros(44100, dtype=np.float32)
  cm.stereo_to_mono_float32(audio, mono)
  ```

  **Use Cases:**
  - High-performance audio applications requiring fast I/O
  - Real-time audio processing with low latency requirements
  - Large audio file manipulation without memory constraints
  - Batch processing workflows with buffer reuse
  - Audio analysis and DSP requiring maximum performance
  - Professional audio software with strict performance requirements

- **MIDI and AudioUnit Plugin Support for DAW Module** - Complete MIDI sequencing and plugin integration (October 2025)
  - **MIDINote and MIDIClip Classes** (`src/coremusic/daw.py`)
    - `MIDINote` dataclass for individual MIDI notes with pitch, velocity, timing, duration, and channel
    - `MIDIClip` class for MIDI note containers with sorting and time-range queries
    - `add_note()` method for adding MIDI notes with automatic sorting
    - `get_notes_in_range()` for querying notes within time ranges
  - **Enhanced Clip Class** with MIDI support
    - Added `clip_type` parameter ('audio' or 'midi')
    - `is_midi` property for type checking
    - Support for `MIDIClip` as source data
    - Unified API for both audio and MIDI clips
  - **AudioUnitPlugin Class** - Complete wrapper for AudioUnit instruments and effects
    - Automatic AudioUnit initialization with sample rate configuration
    - `send_midi()` method for sending MIDI events to instruments
    - `process_audio()` method for audio effects processing
    - Support for both 4-character codes and full plugin names
    - Proper resource management with `dispose()` and `__del__` cleanup
    - Works with both instrument (`aumu`) and effect (`aufx`) plugins
  - **Enhanced Track Class** plugin support
    - Updated `add_plugin()` creates `AudioUnitPlugin` instances
    - New `set_instrument()` method for MIDI track instruments
    - Plugin chain management (instruments first, then effects)
    - Support for both audio processing and MIDI-driven instruments
  - **Comprehensive Demo Enhancements** (`tests/demos/demo_daw.py`)
    - **MIDI Rendering Functions**:
      - `render_midi_to_audio()` - Convert MIDI notes to audio with synthesized instruments
      - Support for piano, synth, and bass instrument types
      - MIDI note-to-frequency conversion (440Hz = MIDI note 69)
      - Velocity-sensitive rendering with harmonic generation
    - **Audio Effects Functions**:
      - `apply_delay_effect()` - Delay/echo with configurable feedback and mix
      - `apply_reverb_effect()` - Comb filter-based reverb (Freeverb-inspired)
      - Effect chaining support for complex processing
    - **New Demo Functions**:
      - `demo_midi_clip()` - C major scale with piano rendering (creates `midi_piano_melody.wav`)
      - `demo_midi_instruments()` - Chord progression (Am-F-C-G) with 3 instruments (piano/synth/bass)
      - `demo_audio_effects()` - Delay, reverb, and combined effects demonstration
    - **13 Audio Files Generated**:
      - 4 MIDI demonstration files (piano melody + 3 chord variations)
      - 4 effects demonstration files (original + delay + reverb + combo)
      - 5 DAW workflow files (full mix + 4 track stems)
  - **Test Coverage**: All demos run successfully with audio output verification
  - **Total Test Count**: 1074 tests passing (DAW module fully functional)

  **Example Usage:**

  ```python
  import coremusic as cm
  from coremusic.daw import MIDIClip, Clip, Timeline

  # Create MIDI clip with notes
  midi_clip = MIDIClip()
  midi_clip.add_note(note=60, velocity=100, start_time=0.0, duration=0.5)  # C4
  midi_clip.add_note(note=64, velocity=90, start_time=0.5, duration=0.5)   # E4
  midi_clip.add_note(note=67, velocity=95, start_time=1.0, duration=0.5)   # G4

  # Add to MIDI track with instrument
  timeline = Timeline(sample_rate=48000, tempo=120.0)
  piano_track = timeline.add_track("Piano", "midi")
  piano_track.set_instrument("dls ")  # DLSMusicDevice (Apple GM synth)

  # Add MIDI clip to track
  clip = Clip(midi_clip, clip_type="midi")
  clip.duration = 2.0
  piano_track.add_clip(clip, start_time=0.0)

  # Add audio effects to track
  piano_track.add_plugin("AUDelay", plugin_type="effect")
  piano_track.add_plugin("AUReverb", plugin_type="effect")

  # Audio track with effects
  guitar_track = timeline.add_track("Guitar", "audio")
  guitar_track.add_plugin("AUHighpass", plugin_type="effect")
  guitar_track.add_plugin("AUDelay", plugin_type="effect")
  ```

  **Use Cases:**
  - MIDI sequencing and composition
  - Virtual instrument playback (software synths)
  - Audio effects processing chains
  - Complete DAW-style production workflows
  - Live MIDI performance with effects
  - Music production and arrangement

## [0.1.8]

### Added

- **DAW (Digital Audio Workstation) Essentials Module** - Complete DAW building blocks for multi-track applications (January 2025)
  - **New Module**: `coremusic.daw` provides high-level DAW abstractions
  - **Timeline Class** - Multi-track timeline with transport control
    - Sample rate and tempo configuration
    - Multi-track audio and MIDI support
    - Transport control: play, pause, stop, record
    - Playhead position management
    - Timeline duration calculation
    - Ableton Link synchronization support
    - Session state tracking (playing, recording)
  - **Track Class** - Individual audio or MIDI track
    - Audio and MIDI track types
    - Clip management (add, remove, query by time)
    - Volume, pan, mute, solo controls
    - Recording arm state
    - AudioUnit plugin chain integration
    - Parameter automation lanes
    - Automatic clip organization
  - **Clip Class** - Audio/MIDI clip representation
    - Audio file or MIDI sequence source
    - Trim functionality with offset and duration
    - Fade in/out support
    - Gain control (linear multiplier)
    - Method chaining for fluent API
    - Automatic duration detection from audio files
    - Timeline positioning (start time, end time)
  - **AutomationLane Class** - Parameter automation
    - Time-based automation points
    - Three interpolation modes:
      - Linear interpolation (smooth transitions)
      - Step interpolation (instant changes)
      - Cubic interpolation (smooth curves)
    - Automatic point sorting by time
    - Value interpolation at any time point
    - Point management (add, remove, clear)
  - **TimelineMarker Class** - Markers and cue points
    - Position-based markers (seconds)
    - Named markers with optional colors
    - Automatic sorting by position
    - Range-based marker queries
  - **TimeRange Class** - Time range representation
    - Start/end time with duration calculation
    - Containment checking
    - Loop region support
  - **Integration Features**:
    - AudioUnit plugin loading and configuration
    - Ableton Link tempo synchronization
    - Automatic clip duration from AudioFile
    - Transport control with state management
  - **Comprehensive Test Coverage**: 52 tests in `tests/test_daw.py` (100% passing)
  - **Interactive Demo**: `tests/demos/demo_daw.py` with 10 examples
  - **Total Test Count**: 1074 tests passing, 46 skipped (up from 1022 passed)

  **Example Usage:**

  ```python
  import coremusic as cm

  # Create DAW session
  timeline = cm.Timeline(sample_rate=48000, tempo=128.0)

  # Add tracks
  drums = timeline.add_track("Drums", "audio")
  vocals = timeline.add_track("Vocals", "audio")

  # Add clips with trimming and fades
  drums.add_clip(cm.Clip("drums.wav"), start_time=0.0)
  vocals.add_clip(
      cm.Clip("vocals.wav").trim(2.0, 26.0).set_fades(0.5, 1.0),
      start_time=8.0
  )

  # Add automation
  volume_auto = vocals.automate("volume")
  volume_auto.add_point(8.0, 0.0)   # Fade in
  volume_auto.add_point(10.0, 1.0)  # Full volume

  # Add markers and loop region
  timeline.add_marker(0.0, "Intro")
  timeline.add_marker(16.0, "Chorus")
  timeline.set_loop_region(16.0, 32.0)

  # Transport control
  timeline.play()
  timeline.pause()
  timeline.stop()

  # Recording
  vocals.record_enable(True)
  timeline.record()
  ```

  **Use Cases:**
  - Multi-track audio/MIDI recording applications
  - DAW-like timeline interfaces
  - Music production software
  - Live performance tools with transport control
  - Automated mixing and mastering workflows
  - Educational DAW implementations
  - Audio post-production tools

- **Audio Analysis and Feature Extraction** - Comprehensive audio analysis framework for music information retrieval (October 2025)
  - **New Module**: `coremusic.audio.analysis` provides advanced audio analysis capabilities
  - **AudioAnalyzer Class** for comprehensive audio feature extraction
    - **Beat Detection**: Onset-based beat detection with tempo estimation
      - Spectral flux onset detection
      - Autocorrelation-based tempo estimation
      - Downbeat detection for bar tracking
      - Confidence scoring for detection quality
    - **Pitch Detection**: Autocorrelation-based pitch tracking
      - Fundamental frequency detection
      - MIDI note number conversion
      - Cents offset calculation for tuning analysis
      - Confidence scoring per frame
    - **Spectral Analysis**: Frequency domain feature extraction
      - Spectral centroid (brightness measure)
      - Spectral rolloff (frequency content boundary)
      - Peak detection in frequency spectrum
      - FFT-based spectrum analysis at any time point
    - **MFCC Extraction**: Mel-Frequency Cepstral Coefficients
      - Configurable coefficient count (default 13)
      - Mel filterbank implementation
      - DCT transformation for cepstral features
      - Frame-by-frame MFCC matrices
    - **Key Detection**: Musical key and mode estimation
      - Chromagram computation (12 pitch classes)
      - Krumhansl-Schmuckler key profiles
      - Major/minor mode detection
      - Time-averaged chroma analysis
    - **Audio Fingerprinting**: Unique audio identification
      - Spectral peak extraction
      - Peak constellation mapping
      - Hash-based fingerprint generation
      - Content-based audio matching
  - **LivePitchDetector Class** for real-time pitch tracking
    - Streaming pitch detection for live audio
    - Autocorrelation-based algorithm
    - Configurable buffer size and sample rate
    - Returns PitchInfo with frequency, MIDI note, and confidence
  - **Data Classes** for structured results
    - **BeatInfo**: tempo, beats, downbeats, confidence
    - **PitchInfo**: frequency, midi_note, cents_offset, confidence
  - **SciPy Integration**: Leverages scipy.signal for DSP operations
  - **Optional Dependencies**: Requires NumPy and SciPy with graceful fallback
  - **Comprehensive Test Coverage**: 42 tests in `tests/test_audio_analysis.py` (100% passing)
  - **Interactive Demo**: `tests/demos/demo_audio_analysis.py` with 8 examples
  - **Total Test Count**: 942 tests passing, 33 skipped (up from 900 passed)

  **Example Usage:**

  ```python
  import coremusic as cm

  # Beat detection and tempo estimation
  analyzer = cm.AudioAnalyzer("song.wav")
  beat_info = analyzer.detect_beats()
  print(f"Tempo: {beat_info.tempo:.1f} BPM")
  print(f"Beats: {beat_info.beats[:5]}")  # First 5 beat times
  print(f"Downbeats: {beat_info.downbeats}")

  # Pitch detection and tracking
  pitch_info = analyzer.detect_pitch()
  print(f"Frequency: {pitch_info.frequency:.2f} Hz")
  print(f"MIDI Note: {pitch_info.midi_note}")
  print(f"Cents: {pitch_info.cents_offset:+.1f}")

  # Spectral analysis at specific time
  spectrum = analyzer.analyze_spectrum(time=1.0, window_size=0.1)
  print(f"Centroid: {spectrum['centroid']:.1f} Hz")
  print(f"Rolloff: {spectrum['rolloff']:.1f} Hz")
  print(f"Peaks: {spectrum['peaks'][:3]}")  # Top 3 peaks

  # MFCC extraction for timbre analysis
  mfcc = analyzer.extract_mfcc(n_mfcc=13)
  print(f"MFCC shape: {mfcc.shape}")  # (13, n_frames)

  # Key detection
  key, mode = analyzer.detect_key()
  print(f"Key: {key} {mode}")  # e.g., "C major"

  # Audio fingerprinting
  fingerprint = analyzer.get_audio_fingerprint()
  print(f"Fingerprint: {fingerprint[:64]}...")  # First 64 chars

  # Real-time pitch detection
  live_detector = cm.LivePitchDetector(sample_rate=44100.0, buffer_size=2048)
  for audio_chunk in stream:
      pitch_info = live_detector.process(audio_chunk)
      if pitch_info and pitch_info.confidence > 0.8:
          print(f"Pitch: {pitch_info.frequency:.2f} Hz")
  ```

  **Use Cases:**
  - Music information retrieval and analysis
  - Beat tracking for DJ software and auto-sync
  - Pitch detection for tuning and vocal analysis
  - Automatic key detection for harmonic mixing
  - Audio fingerprinting for content identification
  - MFCC extraction for machine learning features
  - Real-time pitch tracking for live performance
  - Spectral analysis for sound design and synthesis

- **Audio Slicing and Recombination** - Complete audio slicing framework for creative sample manipulation (October 2025)
  - **New Module**: `coremusic.audio.slicing` provides comprehensive audio slicing and recombination tools
  - **Slicing Methods**: 5 different slicing algorithms for various use cases
    - **Onset Detection**: Spectral flux-based onset detection for rhythmic material
    - **Transient Detection**: Envelope analysis with dB thresholding for dynamic changes
    - **Zero-Crossing Detection**: Glitch-free slicing at zero crossings
    - **Grid-Based Slicing**: Regular equal-duration divisions with optional beat alignment
    - **Manual Slicing**: User-specified time points for precise control
  - **Slice Dataclass** with properties for duration and sample count
  - **AudioSlicer Class** for detecting and extracting audio slices
    - Configurable sensitivity parameter (0.0-1.0)
    - Optional maximum slice count limiting
    - Minimum slice duration filtering
    - Export slices as individual audio files
  - **SliceCollection Class** with fluent API for slice manipulation
    - `shuffle()` - Randomize slice order
    - `reverse()` - Reverse slice sequence
    - `repeat(times)` - Duplicate slices
    - `filter(predicate)` - Filter slices by condition
    - `sort_by_duration()` - Sort by slice length
    - `select(indices)` - Select specific slices
    - `apply_pattern(pattern)` - Apply custom patterns
    - Method chaining support for complex operations
  - **SliceRecombinator Class** with 5 recombination strategies
    - **Original**: Maintain original order with crossfading
    - **Random**: Random selection and ordering
    - **Reverse**: Reversed order
    - **Pattern**: Custom index-based patterns
    - **Custom**: User-defined ordering functions
    - Crossfading algorithm for smooth transitions (configurable duration)
    - Optional normalization of output audio
  - **Comprehensive Test Coverage**: 50 tests in `tests/test_audio_slicing.py` (100% passing)
  - **Interactive Demo**: `tests/demos/demo_audio_slicing.py` with 9 examples
  - **Total Test Count**: 942 tests passing, 33 skipped (up from 905 passed)

  **Example Usage:**

  ```python
  import coremusic as cm

  # Slice using onset detection
  slicer = cm.AudioSlicer("drums.wav", method="onset", sensitivity=0.6)
  slices = slicer.detect_slices(min_slice_duration=0.05, max_slices=16)

  # Manipulate slices with fluent API
  collection = cm.SliceCollection(slices)
  shuffled = collection.filter(lambda s: s.duration > 0.1).shuffle().repeat(2)

  # Recombine with crossfading
  recombinator = cm.SliceRecombinator(shuffled)
  output = recombinator.recombine(method="random", crossfade_duration=0.01)
  recombinator.export("output.wav", method="pattern", pattern=[0, 2, 1, 3])

  # Grid slicing with beat alignment
  grid_slicer = cm.AudioSlicer("audio.wav", method="grid")
  slices = grid_slicer.detect_slices(divisions=16, tempo=120.0)

  # Zero-crossing for glitch-free slicing
  zc_slicer = cm.AudioSlicer("audio.wav", method="zero_crossing")
  slices = zc_slicer.detect_slices(target_slices=8, snap_to_zero=True)
  ```

  **Use Cases:**
  - Beat slicing for drum loops and rhythm manipulation
  - Creative sample recombination and glitch effects
  - Automatic audio segmentation for music analysis
  - Live performance sample triggering
  - Audio collage and mashup creation

- **Audio Visualization** - Comprehensive visualization tools for audio analysis (October 2025)
  - **New Module**: `coremusic.audio.visualization` provides matplotlib-based audio visualization
  - **WaveformPlotter Class** for waveform visualization
    - Basic waveform plotting with time axis
    - Optional RMS envelope overlay (configurable window size)
    - Optional peak envelope overlay
    - Time range zooming (plot specific sections)
    - Custom figure sizes and titles
    - Save to file (PNG, PDF, etc.) with configurable DPI
  - **SpectrogramPlotter Class** for time-frequency analysis
    - STFT-based spectrogram generation
    - Configurable window size and hop size
    - Multiple colormap support (viridis, magma, plasma, inferno)
    - Window function selection (hann, hamming, blackman)
    - dB scale with configurable min/max values
    - Save spectrograms with high quality
  - **FrequencySpectrumPlotter Class** for spectral analysis
    - Instant spectrum at specific time points
    - Average spectrum over time ranges
    - Logarithmic frequency scale
    - Configurable FFT window sizes (2048, 4096, 8192)
    - Frequency range filtering (min/max Hz)
    - Multiple window function support
  - **matplotlib Integration**: High-quality publication-ready plots
  - **Optional Dependency**: Gracefully handles missing matplotlib
  - **Comprehensive Test Coverage**: 37 tests in `tests/test_audio_visualization.py` (100% passing)
  - **Interactive Demo**: `tests/demos/demo_audio_visualization.py` with 11 examples
  - **Total Test Count**: 942 tests passing, 33 skipped (up from 905 passed)

  **Example Usage:**

  ```python
  import coremusic as cm

  # Plot waveform with envelopes
  plotter = cm.WaveformPlotter("audio.wav")
  fig, ax = plotter.plot(show_rms=True, show_peaks=True)
  plotter.save("waveform.png", dpi=150)

  # Generate spectrogram
  spec = cm.SpectrogramPlotter("audio.wav")
  fig, ax = spec.plot(window_size=2048, cmap="magma", min_db=-80)
  spec.save("spectrogram.png")

  # Frequency spectrum analysis
  spectrum = cm.FrequencySpectrumPlotter("audio.wav")

  # At specific time
  fig, ax = spectrum.plot(time=1.0, window_size=4096)

  # Averaged over time range
  fig, ax = spectrum.plot_average(time_range=(0, 5), hop_size=1024)
  spectrum.save("spectrum.png")

  # Complete workflow
  waveform = cm.WaveformPlotter("audio.wav")
  waveform.plot(time_range=(0.5, 1.5), show_rms=True)  # Zoom to specific range

  spec = cm.SpectrogramPlotter("audio.wav")
  spec.plot(window_size=1024, hop_size=256, cmap="plasma")
  ```

  **Use Cases:**
  - Audio analysis and debugging
  - Music production visualization
  - Scientific audio research
  - Educational demonstrations
  - Publication-quality figures
  - Real-time audio monitoring (with matplotlib animation)

- **OSStatus Error Translation** - Human-readable error messages with recovery suggestions (October 2025)
  - **New Module**: `coremusic.os_status` provides comprehensive OSStatus error code translation
  - **Error Code Coverage**: 100+ error codes from all CoreAudio frameworks
    - AudioHardware errors (13 codes): device, stream, property errors
    - AudioFile errors (14 codes): file I/O, format, permissions errors
    - AudioFormat errors (6 codes): format validation errors
    - AudioFileStream errors (12 codes): streaming parser errors
    - AudioCodec errors (9 codes): codec operation errors
    - AudioUnit errors (20 codes): plugin lifecycle and configuration errors
    - AudioQueue errors (23 codes): queue management errors
    - System errors (4 codes): parameter, memory, permission errors
  - **Translation Functions**:
    - `os_status_to_string(status)` - Convert OSStatus to "ErrorName: Description"
    - `get_error_suggestion(status)` - Get recovery suggestion for error
    - `format_os_status_error(status, operation)` - Complete formatted error message
    - `get_error_info(status)` - Get (name, description, suggestion) tuple
  - **Recovery Suggestions**: 30+ actionable suggestions for common errors
    - File errors: Check permissions, verify path exists, check file format
    - Hardware errors: Check device connection, wait for ready state
    - AudioUnit errors: Verify initialization, check format compatibility
    - Parameter errors: Validate ranges, check format parameters
  - **Enhanced Exception Classes**:
    - Added `CoreAudioError.from_os_status()` class method
    - Automatically formats error with name, description, and suggestion
    - Works with all exception subclasses (AudioFileError, AudioQueueError, etc.)
  - **FourCC Support**: Translates both integer codes and four-character codes
  - **Comprehensive Test Coverage**: 31 new tests in `tests/test_os_status.py` (100% passing)
  - **Zero Dependencies**: Pure Python implementation using only stdlib
  - **Total Test Count**: 735 tests passing, 45 skipped (up from 681 passed, 32 skipped)
    - Added 31 os_status tests
    - Enabled 4 previously skipped AudioQueue tests
    - Improved 19 test assertions across multiple test files

  **Example Usage:**

  ```python
  import coremusic as cm
  from coremusic import os_status

  # Translate error codes
  print(os_status.os_status_to_string(-43))
  # Output: kAudioFileFileNotFoundError: File not found

  # Get recovery suggestion
  suggestion = os_status.get_error_suggestion(-43)
  # Output: Verify the file path exists and is spelled correctly

  # Complete formatted error
  msg = os_status.format_os_status_error(-43, "open audio file")
  # Output: Failed to open audio file: kAudioFileFileNotFoundError (File not found)
  #         Suggestion: Verify the file path exists and is spelled correctly

  # Use with exceptions
  exc = cm.AudioFileError.from_os_status(-43, "load file")
  raise exc
  # AudioFileError: Failed to load file: kAudioFileFileNotFoundError: File not found.
  #                 Verify the file path exists and is spelled correctly
  ```

  **Before (cryptic numeric codes):**
  ```
  RuntimeError: AudioFileOpenURL failed with status: -43
  ```

  **After (human-readable with suggestion):**
  ```
  AudioFileError: Failed to open audio file: kAudioFileFileNotFoundError: File not found.
  Verify the file path exists and is spelled correctly
  ```

  **Impact:**
  - **Developers**: Much easier debugging with clear error names and actionable suggestions
  - **Users**: Better error messages guide them to fix issues themselves
  - **Support**: Reduced support burden with self-explanatory error messages
  - **Documentation**: Error codes now self-documenting

  **Implementation Details:**
  - **Capi Layer Integration**: 150+ error locations updated across `src/coremusic/capi.pyx`
    - Added `format_osstatus_error()` helper function
    - Integrated with `coremusic.log` module for structured error logging
    - All `RuntimeError` messages now include human-readable translations
  - **Objects Layer**: Enhanced `CoreAudioError.from_os_status()` class method
    - Automatically formats errors with name, description, and suggestion
    - Preserves status_code attribute for programmatic access
  - **Test Updates**: Comprehensive test suite updates to work with new error format
    - Updated `tests/test_objects_audio_queue.py`: Changed error detection from `"status: -50"` to `e.status_code == -50 or "paramErr" in str(e)`
    - Updated `tests/test_objects_comprehensive.py`: Similar paramErr error detection pattern
    - Updated `tests/test_coremidi.py`: Changed 18 assertions from `"failed with status"` to `"failed"`
    - Updated `tests/test_audiotoolbox_music_device.py`: Enhanced fixture error detection for userCanceledErr and InvalidFile
    - All tests now verify errors exist without depending on exact message format
  - **Logging Integration**: Structured logging with extra context
    - Error logs include status_code, operation, and suggestion fields
    - Controlled via DEBUG environment variable (DEBUG=0 disables logging)
  - **Demo Application**: `tests/demos/demo_os_status_errors.py` with 7 comprehensive examples
    - Basic error translation demonstration
    - Recovery suggestions showcase
    - Complete formatted error messages
    - Enhanced exception classes usage
    - Structured error information
    - Real-world scenario (file not found)
    - Error categories overview
  - **Backward Compatibility**: Status codes still preserved in exception attributes

### Changed

- **Unified AudioAnalyzer class** - Merged basic utility methods into comprehensive analysis class (October 2025)
  - **Issue**: Two separate `AudioAnalyzer` classes existed with naming conflict
    - `coremusic.audio.analysis.AudioAnalyzer` - Advanced music analysis (beat/pitch detection, MFCC, key detection)
    - `coremusic.audio.utilities.AudioAnalyzer` - Basic metrics (silence detection, peak, RMS)
  - **Solution**: Merged utility methods into analysis class as static methods
  - **New unified API**:
    - **Instance Methods** (advanced analysis, requires SciPy):
      - `detect_beats()` - Beat detection and tempo estimation
      - `detect_pitch()` - Pitch tracking over time
      - `analyze_spectrum()` - Spectral analysis
      - `extract_mfcc()` - MFCC extraction
      - `detect_key()` - Musical key detection
      - `get_audio_fingerprint()` - Audio fingerprinting
    - **Static Methods** (basic metrics, NumPy only):
      - `detect_silence()` - Find quiet regions in audio
      - `get_peak_amplitude()` - Maximum amplitude
      - `calculate_rms()` - RMS level calculation
      - `get_file_info()` - Comprehensive file metadata
  - **Benefits**:
    - Single import for all audio analysis: `from coremusic.audio.analysis import AudioAnalyzer`
    - Naming conflict resolved - one AudioAnalyzer class
    - Flexible API - choose static or instance methods based on needs
    - No breaking changes - existing code continues to work
  - **Example Usage**:
    ```python
    # Static API (no initialization, lightweight)
    silence = AudioAnalyzer.detect_silence("audio.wav", threshold_db=-40)
    peak = AudioAnalyzer.get_peak_amplitude("audio.wav")

    # Instance API (advanced analysis)
    analyzer = AudioAnalyzer("song.wav")
    beat_info = analyzer.detect_beats()
    key, mode = analyzer.detect_key()
    ```
  - **Migration**: Tests updated to import from `coremusic.audio.analysis`
  - **Verification**: All 1022 tests passing, type checking successful

- **Reorganized utilities module** - Moved `coremusic.utilities` to `coremusic.audio.utilities` (October 2025)
  - **Change**: Relocated utilities module for better package organization
    - From: `coremusic.utilities`
    - To: `coremusic.audio.utilities`
  - **Reason**: Utilities are audio-specific and belong in audio subpackage
  - **Updated imports**:
    - Main package: `from .audio.utilities import *` in `coremusic/__init__.py`
    - Audio package: Added utilities exports to `coremusic.audio.__init__.py`
    - Fixed all relative imports within utilities.py (`.` → `..`)
  - **Exports remain accessible**: All utilities still available via `import coremusic as cm`
  - **No breaking changes**: Existing user code continues to work
  - **Verification**: All 1022 tests passing, type checking successful

- **Improved test coverage for AudioQueue OO API** - Selective skipping instead of blanket test exclusion (October 2025)
  - **Issue**: All 16 tests in `test_objects_audio_queue.py` were skipped due to module-level `pytestmark`
  - **Root Cause**: Overly conservative assumption that all AudioQueue tests require audio hardware
  - **Fix**: Removed blanket skip marker and implemented selective skipping using fixture-based hardware detection
  - **Result**: 4 tests now passing (25% → 100% execution for non-hardware tests), 12 tests properly skip when hardware unavailable
  - **Tests now running without hardware**:
    - `test_audio_buffer_creation` - Pure Python object creation
    - `test_audio_buffer_properties` - Property access testing
    - `test_audio_queue_creation_with_format` - Object initialization
    - `test_audio_queue_error_handling` - Error handling for invalid formats
  - **Hardware-dependent tests** gracefully skip with clear message: "Audio hardware not available"
  - **Impact**: Better CI/headless environment coverage while preserving hardware functionality tests
  - **Verification**: All 681 tests passing, 32 skipped (no regressions)
  - **Updated error detection**: Changed from checking for string "status: -50" to checking status_code attribute or "paramErr" keyword
  - **Better compatibility**: Tests now work with new human-readable error format

- **Enhanced documentation for bytes parameters** - Added comprehensive usage examples to method docstrings (October 2025)
  - **Analysis**: Identified 6 methods accepting `bytes` parameters representing binary audio/MIDI data
  - **Confirmed**: All `bytes` parameters are correctly typed (binary data, not text):
    - `AudioFileStream.parse_bytes()` - Raw audio file format data (WAV/MP3/AAC headers)
    - `AudioConverter.convert()` - Raw PCM audio samples
    - `AudioConverter.convert_with_callback()` - Raw audio samples (already had example)
    - `AudioConverter.set_property()` - Binary property data (structs, ints)
    - `ExtendedAudioFile.write()` - Raw audio frame data
    - `MIDIOutputPort.send_data()` - MIDI protocol messages
  - **Documentation improvements**:
    - Added practical usage examples to 5 methods (1 already had examples)
    - Clarified binary nature of data with inline comments
    - Showed proper `struct.pack()` usage for creating binary data
    - Demonstrated MIDI protocol byte construction
    - Included context managers and realistic workflows
  - **Consistency**: All examples follow existing pattern from `convert_with_callback()`
  - **Verification**: All 681 tests passing (documentation-only changes, no functional impact)

### Fixed

- **Music device test fixture errors** - Enhanced error detection for security restrictions and invalid components (October 2025)
  - **Issue**: Fixture in `test_audiotoolbox_music_device.py` was showing ERROR instead of SKIPPED for unavailable/broken plugins
  - **Root Cause**: Error detection was checking for numeric code "-128" but new OSStatus translation changed format to "userCanceledErr"
  - **Fix**: Updated error detection to check for keyword "userCanceledErr" or "security restriction" instead of numeric codes
  - **Also handles**: kAudioUnitErr_InvalidFile (-10863) for broken third-party plugins
  - **Impact**: All 4 test errors converted to proper skips with clear messages
  - **Result**: Tests now gracefully skip when encountering:
    - Security-restricted components (userCanceledErr -128)
    - Invalid/broken plugin files (kAudioUnitErr_InvalidFile -10863)
    - Other component instantiation failures
  - **Final test count**: 735 passed, 45 skipped (up from 735 passed, 41 skipped, 4 errors)

- **AudioUnit factory presets crash** - Fixed critical bug in `audio_unit_get_factory_presets()` (src/coremusic/capi.pyx:1802)
  - **Root Cause**: Code was incorrectly treating `kAudioUnitProperty_FactoryPresets` return value as a CFArray of CFDictionaries
  - **Correct Implementation**: According to [Apple TechNote TN2157](https://developer.apple.com/library/archive/technotes/tn2157/_index.html), the property returns a CFArray of `AUPreset` structs
  - **Fix**: Changed implementation to cast array elements directly to `AUPreset*` pointers and access struct fields (`presetNumber`, `presetName`) instead of creating CFString keys and using `CFDictionaryGetValue()`
  - **Impact**: Eliminated crashes when querying factory presets from AudioUnits like AUDynamicsProcessor, AUDistortion, etc.
  - **Test Results**: Successfully discovered factory presets from 9 Apple plugins (48 total tested, 18.8% have presets)
  - Added missing `preset_name` variable declaration that was causing undefined behavior

- **Music device test fixture errors** - Improved error handling in `test_audiotoolbox_music_device.py`
  - **Issue**: Test fixture was encountering broken/unavailable third-party music device plugins returning error -10863 (`kAudioUnitErr_InvalidFile`)
  - **Fix**: Added -10863 to the list of errors that trigger test skip (alongside existing -128 handling)
  - **Impact**: Tests now properly skip instead of erroring when encountering incompatible plugins
  - **Result**: All 677 tests passing, 37 skipped, 0 errors

### Added

- **AudioUnit Host Enhancements** - Advanced audio format support, preset management, and plugin chaining
  - **AudioFormat Class** (`src/coremusic/audiounit_host.py:18-93`)
    - Support for multiple sample formats: `float32`, `float64`, `int16`, `int32`
    - Interleaved and non-interleaved buffer layout support
    - Properties: `bytes_per_sample`, `bytes_per_frame`
    - Format comparison and dictionary serialization
    - Type-safe format specification with string constants
  - **AudioFormatConverter Class** (`src/coremusic/audiounit_host.py:94-243`)
    - Automatic format conversion between any supported formats
    - Two-stage conversion pipeline: source → float32 interleaved → destination
    - Proper audio normalization to [-1.0, 1.0] range
    - Support for all format combinations (format, bit depth, channel layout)
    - Symmetric rounding for integer formats
  - **PresetManager Class** (`src/coremusic/audiounit_host.py:341-535`)
    - Complete preset lifecycle management (save/load/export/import)
    - JSON-based preset storage in `~/Library/Audio/Presets/coremusic/`
    - Preset metadata: name, description, plugin info, timestamp
    - Parameter state capture and restoration
    - Preset validation and compatibility checking
    - List, delete, export, and import operations
  - **AudioUnitChain Class** (`src/coremusic/audiounit_host.py:1169-1438`)
    - Sequential plugin processing with automatic routing
    - Dynamic chain building: add, insert, remove plugins
    - Automatic format conversion between plugins
    - Wet/dry mixing support (blend processed and original signals)
    - Plugin configuration by index
    - Context manager support for automatic cleanup
    - Method chaining for fluent API
  - **Enhanced AudioUnitPlugin** (`src/coremusic/audiounit_host.py:565-930`)
    - `set_audio_format()` - Configure plugin audio format
    - `process()` enhanced with format parameter for automatic conversion
    - `save_preset()`, `load_preset()`, `list_user_presets()` - Preset management
    - `delete_preset()`, `export_preset()`, `import_preset()` - Preset operations
    - `audio_format` property for format queries
  - **Comprehensive test coverage** - 37 new tests in `tests/test_audiounit_host_enhancements.py`
    - 5 AudioFormat tests (creation, properties, equality, serialization)
    - 7 AudioFormatConverter tests (all format combinations, interleaved/non-interleaved)
    - 6 PresetManager tests (save, load, list, delete, export, import)
    - 3 Plugin enhancement tests (format setting, conversion, integration)
    - 14 AudioUnitChain tests (creation, operations, processing, context manager)
    - 1 full workflow integration test
    - 27 tests passing, 10 skipped (plugins not available)
  - **Updated exports** - New classes exported from `coremusic` module
    - `AudioFormat`, `AudioFormatConverter`, `AudioUnitChain`, `PresetManager`
    - All classes available via `import coremusic as cm`
  - **Documentation updates**
    - `TODO.md` updated with completed features and usage examples
    - Test count updated to **736 tests passing** (100% success rate)
  - **Zero test regressions** - All 736 tests passing after enhancements

  **Example Usage:**

  ```python
  import coremusic as cm

  # Audio Format Support
  fmt = cm.PluginAudioFormat(44100.0, 2, cm.PluginAudioFormat.INT16, interleaved=True)
  plugin.set_audio_format(fmt)
  output = plugin.process(input_data, num_frames, fmt)

  # User Preset Management
  plugin.save_preset("My Reverb", "Large hall with 3s decay")
  plugin.load_preset("My Reverb")
  presets = plugin.list_user_presets()
  plugin.export_preset("My Reverb", "/path/to/export.json")
  plugin.import_preset("/path/to/preset.json")

  # AudioUnit Chain
  chain = cm.AudioUnitChain()
  chain.add_plugin("AUHighpass")
  chain.add_plugin("AUDelay")
  chain.add_plugin("AUReverb")
  chain.configure_plugin(0, {'Cutoff Frequency': 200.0})
  chain.configure_plugin(1, {'Delay Time': 0.5})
  output = chain.process(input_audio, num_frames, wet_dry_mix=0.8)

  # Or use context manager
  with cm.AudioUnitChain() as chain:
      chain.add_plugin("AUDelay")
      output = chain.process(input_data)
  ```

- **AudioUnit MIDI Support** - Complete MIDI control for AudioUnit instrument plugins
  - **MIDI Methods in AudioUnitPlugin class** (`src/coremusic/audiounit_host.py`)
    - `send_midi()` - Send raw MIDI messages to instrument plugins
    - `note_on()` - Send MIDI Note On with channel, note, velocity, and optional offset frames
    - `note_off()` - Send MIDI Note Off with channel, note, and optional velocity/offset
    - `control_change()` - Send MIDI Control Change (volume, pan, expression, etc.)
    - `program_change()` - Send MIDI Program Change for instrument selection (General MIDI)
    - `pitch_bend()` - Send MIDI Pitch Bend with 14-bit precision (0-16383)
    - `all_notes_off()` - Emergency stop all notes on a channel (MIDI CC 123)
    - Type checking ensures MIDI methods only work on instrument plugins (`aumu` type)
    - Sample-accurate MIDI scheduling with `offset_frames` parameter
  - **Full MIDI Specification Support**
    - All 128 MIDI notes (0-127)
    - All 128 velocity levels (0-127)
    - All 128 MIDI controllers (CC 0-127)
    - All 128 General MIDI programs (0-127)
    - 14-bit pitch bend precision (0-16383, center = 8192)
    - All 16 MIDI channels (0-15)
    - Sample-accurate timing for tight rhythmic patterns
  - **Comprehensive test coverage** - 19 new tests in `tests/test_audiounit_midi.py`
    - Basic MIDI operations (note on/off, chords, scales)
    - Velocity and note range testing across MIDI spec
    - Control Change messages (volume, pan, expression)
    - Program Change for instrument selection
    - Pitch Bend messages with smooth modulation
    - All Notes Off command
    - Multi-channel MIDI (all 16 channels)
    - Raw MIDI message sending
    - Sample-accurate scheduling with offset frames
    - Type checking (MIDI rejected on effect plugins)
    - Error handling and validation
    - Rapid note sequences (arpeggiator patterns)
    - Multi-channel orchestration
  - **Interactive demo application** - `tests/demos/audiounit_instrument_demo.py`
    - 8 comprehensive demonstrations of MIDI functionality
    - Plugin discovery (62 instrument plugins found)
    - Basic MIDI control (notes, chords, C major scale)
    - Instrument selection via General MIDI program changes
    - MIDI controller automation (volume fade, pan sweep)
    - Pitch bend demonstrations (smooth pitch modulation)
    - Multi-channel performance (4-channel orchestration example)
    - Arpeggiator patterns (rapid note sequences)
    - Interactive keyboard mapping demo
    - Integration with Apple DLSMusicDevice (built-in General MIDI synth)
  - **Updated documentation**
    - `docs/dev/audiounit_implementation.md` updated with MIDI sections and examples
    - MIDI instrument control examples
    - Multi-channel MIDI orchestration examples
    - Sample-accurate MIDI scheduling examples
    - Updated test coverage and demo information
  - **All 662 tests passing** (643 existing + 19 new MIDI tests)
  - **62 instrument plugins** discovered and working with MIDI control

  **Example Usage:**

  ```python
  import coremusic as cm
  import time

  # Load a General MIDI synthesizer
  with cm.AudioUnitPlugin.from_name("DLSMusicDevice", component_type='aumu') as synth:
      # Play a note
      synth.note_on(channel=0, note=60, velocity=100)  # Middle C
      time.sleep(1.0)
      synth.note_off(channel=0, note=60)

      # Play a chord
      notes = [60, 64, 67]  # C major (C, E, G)
      for note in notes:
          synth.note_on(channel=0, note=note, velocity=90)
      time.sleep(1.5)
      synth.all_notes_off(channel=0)

      # Change instrument (General MIDI)
      synth.program_change(channel=0, program=0)   # Acoustic Grand Piano
      synth.program_change(channel=0, program=40)  # Violin

      # Control volume with MIDI CC
      synth.control_change(channel=0, controller=7, value=100)  # Full volume
      synth.control_change(channel=0, controller=7, value=50)   # Half volume

      # Pitch bend
      synth.note_on(channel=0, note=60, velocity=100)
      synth.pitch_bend(channel=0, value=8192)   # Center (no bend)
      synth.pitch_bend(channel=0, value=12288)  # Bend up
      synth.pitch_bend(channel=0, value=8192)   # Back to center
      synth.note_off(channel=0, note=60)

  # Multi-channel orchestration
  with cm.AudioUnitPlugin.from_name("DLSMusicDevice", component_type='aumu') as synth:
      # Setup different instruments on different channels
      synth.program_change(channel=0, program=0)   # Piano
      synth.program_change(channel=1, program=48)  # Strings
      synth.program_change(channel=2, program=56)  # Trumpet

      # Play multi-channel arrangement
      synth.note_on(channel=0, note=60, velocity=90)  # Piano
      synth.note_on(channel=1, note=64, velocity=70)  # Strings
      synth.note_on(channel=2, note=72, velocity=80)  # Trumpet
      time.sleep(1.0)

      # Clean stop all channels
      for ch in range(3):
          synth.all_notes_off(channel=ch)
  ```

- **AudioPlayer.play() method** - Added `play()` as an intuitive alias for `start()` method
  - Both `player.play()` and `player.start()` now work identically
  - Improves API ergonomics and developer experience
  - Backward compatible - existing code using `start()` continues to work

- **Ableton Link Integration** - Complete tempo synchronization and beat grid support
  - **Link Cython wrapper** (`src/coremusic/link.pyx` and `link.pxd`)
    - `Clock` class - Platform-specific clock for Link timing
      - `micros()` - Get current time in microseconds
      - `ticks()` - Get current time in system ticks (mach_absolute_time)
      - `ticks_to_micros()` - Convert system ticks to microseconds
      - `micros_to_ticks()` - Convert microseconds to system ticks
    - `SessionState` class - Link timeline and transport state snapshot
      - Properties: `tempo`, `is_playing`
      - Beat/phase queries: `beat_at_time()`, `phase_at_time()`, `time_at_beat()`
      - Beat mapping: `request_beat_at_time()`, `force_beat_at_time()`
      - Transport control: `set_tempo()`, `set_is_playing()`, `time_for_is_playing()`
      - Convenience methods: `request_beat_at_start_playing_time()`, `set_is_playing_and_request_beat_at_time()`
    - `LinkSession` class - Main Link session for tempo synchronization
      - Properties: `enabled`, `num_peers`, `start_stop_sync_enabled`, `clock`
      - Session state capture: `capture_audio_session_state()`, `capture_app_session_state()`
      - Session state commit: `commit_audio_session_state()`, `commit_app_session_state()`
      - Realtime-safe audio thread operations with `nogil`
  - **AudioPlayer Link Integration** (`src/coremusic/capi.pyx`)
    - AudioPlayer now accepts optional `link_session` parameter
    - `link_session` property to access attached Link session
    - `get_link_timing(quantum)` method returns timing info dict (tempo, beat, phase, is_playing)
    - Python-layer timing queries for synchronized playback control
    - Link session reference kept alive to prevent garbage collection
  - **C++ Build Integration** (`setup.py`)
    - Link extension compiled with C++11 support
    - Include paths for Link library and ASIO standalone
    - LINK_PLATFORM_MACOSX define for macOS platform
  - **Comprehensive test coverage**
    - `test_link.py` - 25 tests covering all Link functionality
      - Clock operations (time queries, conversions, round-trip)
      - Session management (enable/disable, peers, transport sync)
      - State capture and commit (audio/app thread)
      - Tempo control and beat/phase calculations
      - Transport state management
      - Two-session synchronization tests
    - `test_link_audio_integration.py` - 9 tests for AudioPlayer integration
      - Player creation with/without Link session
      - Timing queries and updates
      - Tempo and transport state visibility
      - Multiple players sharing Link session
      - Reference lifecycle management
    - All 575 tests passing (566 existing + 9 new)
  - **Demo application** (`tests/demos/link_audio_demo.py`)
    - Complete Link + AudioPlayer workflow demonstration
    - Real-time beat/tempo monitoring during playback
    - Visual beat indicators and progress tracking
    - Example of synchronized audio playback
  - **High-Level Python API** (Phase 3 enhancements)
    - Context manager support for `LinkSession` - automatic enable/disable
    - `__enter__` and `__exit__` methods for `with` statement support
    - Exported from main `coremusic` package via `cm.link` module
    - Fully Pythonic API with properties, named arguments, informative `__repr__`
    - 19 additional tests for high-level API patterns
    - High-level demo (`tests/demos/link_high_level_demo.py`) with 6 examples
    - All 594 tests passing (566 existing + 9 AudioPlayer + 19 high-level API)
  - **Link + CoreMIDI Integration** (`src/coremusic/link_midi.py`)
    - `LinkMIDIClock` class - MIDI Clock messages synchronized to Link tempo
      - Sends 24 clock messages per quarter note per MIDI spec
      - Automatic tempo tracking when Link tempo changes
      - Sends MIDI Start/Stop messages
      - Runs in separate thread for realtime performance
    - `LinkMIDISequencer` class - Beat-accurate MIDI event scheduling
      - Schedule MIDI events at specific Link beat positions
      - `schedule_note()` - Schedule notes with automatic note-off
      - `schedule_cc()` - Schedule MIDI CC messages
      - `schedule_event()` - Schedule arbitrary MIDI messages
      - Events kept sorted by beat position
      - Thread-safe event scheduling
    - Time conversion utilities
      - `link_beat_to_host_time()` - Convert Link beats to mach_absolute_time
      - `host_time_to_link_beat()` - Convert host time to Link beats
      - Round-trip conversion with < 0.01 beat accuracy
    - MIDI constants (MIDI_CLOCK, MIDI_START, MIDI_STOP, MIDI_CLOCKS_PER_QUARTER_NOTE)
    - 20 comprehensive tests covering all functionality
    - Interactive demo (`tests/demos/link_midi_demo.py`) with 3 examples
    - All 614 tests passing (594 existing + 20 Link+MIDI integration)

  **Example Usage:**

  ```python
  import coremusic as cm

  # Basic Link usage with context manager
  with cm.link.LinkSession(bpm=120.0) as session:
      state = session.capture_app_session_state()
      print(f"Tempo: {state.tempo:.1f} BPM, Peers: {session.num_peers}")

  # AudioPlayer + Link integration
  with cm.link.LinkSession(bpm=120.0) as session:
      player = cm.AudioPlayer(link_session=session)
      player.load_file("audio.wav")
      player.setup_output()

      # Query Link timing
      timing = player.get_link_timing(quantum=4.0)
      print(f"Beat: {timing['beat']:.2f}, Tempo: {timing['tempo']:.1f} BPM")

      player.play()
      player.start()

  # MIDI Clock synchronized to Link
  from coremusic import link_midi

  client = cm.capi.midi_client_create("MIDI Clock")
  port = cm.capi.midi_output_port_create(client, "Clock Out")
  dest = cm.capi.midi_get_destination(0)

  with cm.link.LinkSession(bpm=120.0) as session:
      clock = link_midi.LinkMIDIClock(session, port, dest)
      clock.start()  # Sends MIDI clock messages
      time.sleep(10)
      clock.stop()

  # Beat-accurate MIDI sequencer
  with cm.link.LinkSession(bpm=120.0) as session:
      seq = link_midi.LinkMIDISequencer(session, port, dest)

      # Schedule notes at Link beat positions
      seq.schedule_note(beat=0.0, channel=0, note=60, velocity=100, duration=0.9)
      seq.schedule_note(beat=1.0, channel=0, note=64, velocity=100, duration=0.9)

      seq.start()
      time.sleep(5)
      seq.stop()
  ```

## [0.1.7]

### Added

- **CoreAudioClock API** - Complete audio/MIDI synchronization and timing services
  - **Low-level C API wrappers** in `capi.pyx`
    - `ca_clock_new()` - Create new clock instances
    - `ca_clock_dispose()` - Resource cleanup
    - `ca_clock_start()` / `ca_clock_stop()` - Playback control
    - `ca_clock_get_play_rate()` / `ca_clock_set_play_rate()` - Speed control
    - `ca_clock_get_current_time()` - Time queries with format support
    - Time format getter functions for seconds, beats, samples, host time
  - **High-level AudioClock class** with context manager support
    - Properties: `play_rate`, `is_running`, `is_disposed`
    - Methods: `start()`, `stop()`, `get_time_seconds()`, `get_time_beats()`, `get_time_samples()`, `get_time_host()`
    - Automatic resource management with `__enter__` and `__exit__`
  - **ClockTimeFormat constants** for time format specifications
    - `HOST_TIME` - mach_absolute_time()
    - `SAMPLES` - Audio sample count
    - `BEATS` - Musical beats
    - `SECONDS` - Seconds
    - `SMPTE_TIME` - SMPTE timecode
  - **Comprehensive test coverage** - 21 tests covering all functionality
    - Low-level API tests (create/dispose, start/stop, play rate, time formats)
    - High-level API tests (context manager, properties, time getters)
    - Timing accuracy verification (normal and half-speed)
    - Error handling and multiple simultaneous clocks
  - **Complete documentation**
    - Sphinx API reference with autodoc integration
    - Code examples in main index and getting started guide
    - Detailed docstrings with RST formatting
  - **Use cases**: DAWs, sequencers, MIDI sync, tempo control, audio/MIDI alignment

  **Example Usage:**

  ```python
  import coremusic as cm

  # High-level API
  with cm.AudioClock() as clock:
      clock.play_rate = 1.0  # Normal speed
      clock.start()

      # Get time in different formats
      seconds = clock.get_time_seconds()
      beats = clock.get_time_beats()
      samples = clock.get_time_samples()

      # Change speed for tempo sync
      clock.play_rate = 0.5  # Half speed

      clock.stop()

  # Low-level API
  import coremusic.capi as capi

  clock_id = capi.ca_clock_new()
  capi.ca_clock_start(clock_id)
  # ... operations ...
  capi.ca_clock_dispose(clock_id)
  ```

- **Full mypy type checking support**
  - Added comprehensive type hints across entire Python codebase
  - Configured strict mypy settings in `pyproject.toml`
  - Fixed all type errors in `scipy_utils.py`, `utilities.py`, `async_io.py`
  - Added `make typecheck` target to Makefile
  - All 516 tests passing with full type safety

- **AudioStreamBasicDescription parsing utility**
  - Added `parse_audio_stream_basic_description()` function to `utilities` module
  - Parses 40-byte ASBD structure from CoreAudio APIs into Python dictionary
  - Returns all format fields: sample_rate, format_id, channels, bit depth, etc.
  - Comprehensive documentation with structure layout and usage examples
  - 3 test cases verifying parsing, validation, and compatibility with OO API
  - Useful for functional API users who need to parse raw format data

  **Example Usage:**

  ```python
  import coremusic as cm
  import coremusic.capi as capi

  file_id = capi.audio_file_open_url("audio.wav")
  format_data = capi.audio_file_get_property(
      file_id,
      capi.get_audio_file_property_data_format()
  )
  asbd = cm.parse_audio_stream_basic_description(format_data)
  print(f"{asbd['sample_rate']} Hz, {asbd['channels_per_frame']} channels")
  capi.audio_file_close(file_id)
  ```

### Fixed

- **Sphinx documentation build warnings** - Eliminated all 41 warnings in documentation build
  - Fixed AudioClock docstring RST formatting (changed markdown code blocks to RST format)
  - Removed autofunction directives for non-exported capi functions
  - Updated API reference to guide users to `coremusic.capi` module for low-level functions
  - Updated audio file documentation examples to use correct import patterns
  - Fixed Makefile documentation targets to properly delegate to docs/Makefile
  - Documentation now builds cleanly with 0 warnings, 0 errors

### Changed

- **Pure Cython Audio Player Implementation** - Replaced C audio player with native Cython implementation
  - **Removed C dependencies**: Eliminated `audio_player.c`, `audio_player.h`, and `audio_player.pxd` files
  - **Simplified build process**: No separate C compilation needed, all audio playback in Cython
  - **Cleaner architecture**: Consistent with existing callback patterns in the codebase
  - **Same functionality**: All `AudioPlayer` methods work identically with same API
  - **Pure Cython render callback**: `audio_player_render_callback()` implemented as `cdef` function with `noexcept nogil`
  - **ExtAudioFile-based loading**: Uses already-wrapped ExtAudioFile APIs for audio file loading
  - **AudioUnit integration**: Native AudioUnit setup and control entirely in Cython
  - **Better maintainability**: All code in one language, easier to understand and extend
  - **Proven pattern**: Follows same approach as existing `audio_queue_output_callback` and `audio_converter_input_callback`
  - **Zero test regressions**: All 516 tests passing after migration
  - **Fixed build configuration**: Updated `setup.py` and `pyproject.toml` for pure Cython build

  **Technical Details:**
  - Render callback handles real-time audio rendering, looping, and playback state
  - Automatic format conversion to 44.1kHz stereo float32
  - Sample-rate conversion and chunked reading for large files
  - Full AudioUnit lifecycle management (initialize, start, stop, cleanup)
  - Proper memory management with automatic buffer cleanup

  **Impact:**
  - **Users**: No API changes - `AudioPlayer` works exactly the same
  - **Developers**: Simpler codebase with better maintainability
  - **Build**: Faster compilation without separate C sources

## [0.1.6]

### Changed

- **Namespace Refactoring** - Separated object-oriented API from functional C API for cleaner, more Pythonic interface
  - **Object-Oriented API is now the primary interface** - All high-level classes available directly from `import coremusic as cm`
  - **Functional C API moved to explicit namespace** - Low-level C functions now require `import coremusic.capi as capi`
  - **Cleaner main namespace** - `coremusic.*` now contains only Pythonic object-oriented classes and utilities
  - **Advanced users retain full access** - Complete functional API still available via `capi` submodule
  - **Re-exported base classes** - `CoreAudioObject` and `AudioPlayer` properly exported from main namespace
  - **Comprehensive migration** - 1,126 functional API calls migrated across 27 files (tests, demos, scripts)
  - **Zero test regressions** - All 516 tests passing after migration

  **Before (intermingled APIs):**

  ```python
  import coremusic as cm

  # Mix of OO and functional APIs in same namespace
  file = cm.AudioFile("audio.wav")  # OO class
  file_id = cm.audio_file_open_url("audio.wav")  # functional C API
  ```

  **After (clean separation):**

  ```python
  import coremusic as cm
  import coremusic.capi as capi

  # Object-oriented API (primary interface)
  file = cm.AudioFile("audio.wav")

  # Functional C API (advanced usage)
  file_id = capi.audio_file_open_url("audio.wav")
  ```

  **Impact:**
  - **Most users** - No changes needed if using OO API (`AudioFile`, `AudioQueue`, `AudioUnit`, etc.)
  - **Advanced users** - Add `import coremusic.capi as capi` and prefix functional calls with `capi.`
  - **SciPy utilities** - Already required explicit import: `import coremusic.scipy_utils as spu`

- Removed auto-import of scipy utilities in `__init__.py`

## [0.1.5]

- First pypi release for python 3.11 - 3.14 inclusive.

### Added

- sphinx docs, tutorials and examples.

- **SciPy Signal Processing Integration** - Seamless integration with SciPy for advanced audio DSP workflows
  - **Filter Design** (`scipy_utils.py`)
    - `design_butterworth_filter()` - Design Butterworth filters (lowpass, highpass, bandpass, bandstop)
    - `design_chebyshev_filter()` - Design Chebyshev Type I filters with configurable ripple
    - Support for all standard filter types with customizable order
  - **Filter Application**
    - `apply_filter()` - Generic filter application with zero-phase filtering option
    - `apply_scipy_filter()` - **NEW** Convenience wrapper accepting scipy.signal filter output directly
    - `apply_lowpass_filter()` - Convenient lowpass filtering
    - `apply_highpass_filter()` - Convenient highpass filtering
    - `apply_bandpass_filter()` - Convenient bandpass filtering
    - Automatic handling of mono and stereo audio
  - **Resampling**
    - `resample_audio()` - High-quality resampling using SciPy
    - Support for both FFT and polyphase methods
    - Automatic multi-channel handling
  - **Spectral Analysis**
    - `compute_spectrum()` - Power spectral density using Welch's method
    - `compute_fft()` - Fast Fourier Transform with windowing
    - `compute_spectrogram()` - Time-frequency analysis
    - Configurable window functions (hann, hamming, blackman, etc.)
  - **AudioSignalProcessor Class** - High-level interface for DSP workflows
    - Method chaining for fluent API (e.g., `.lowpass(1000).normalize().get_audio()`)
    - Built-in methods: `lowpass()`, `highpass()`, `bandpass()`, `resample()`, `normalize()`
    - Integrated spectral analysis: `spectrum()`, `fft()`, `spectrogram()`
    - `reset()` method to restore original audio
  - **SCIPY_AVAILABLE** flag for feature detection
  - **42 comprehensive tests** covering all SciPy functionality (including 7 tests for convenience API)
  - **Demo script** (`tests/demos/demo_scipy_integration.py`) with 6 detailed examples
  - **Complete NumPy/SciPy ecosystem integration** for scientific audio processing

  **Example Usage:**

  ```python
  import coremusic as cm
  import coremusic.scipy_utils as spu

  # Load and process audio
  with cm.AudioFile("audio.wav") as af:
      audio = af.read_as_numpy()
      sr = af.format.sample_rate

  # Use AudioSignalProcessor for chained operations
  processor = spu.AudioSignalProcessor(audio, sr)
  processed = (processor
              .highpass(50)      # Remove rumble
              .lowpass(15000)    # Remove ultrasonic
              .normalize(0.9)    # Normalize
              .get_audio())

  # Or use individual functions
  filtered = spu.apply_lowpass_filter(audio, cutoff=2000, sample_rate=sr)
  resampled = spu.resample_audio(audio, original_rate=sr, target_rate=48000)
  freqs, spectrum = spu.compute_spectrum(audio, sample_rate=sr)

  # Or use scipy.signal filters directly with convenience wrapper
  import scipy.signal
  filtered = spu.apply_scipy_filter(audio, scipy.signal.butter(5, 1000, 'low', fs=sr))
  ```

- **Complex Audio Conversion Support** - Full callback-based AudioConverter API for advanced audio format conversions
  - **Callback Infrastructure** in Cython layer (`src/coremusic/capi.pyx`)
    - `AudioConverterCallbackData` struct for passing data between Python and C callback
    - `audio_converter_input_callback()` - C callback function with `nogil` and `noexcept` for providing input data on demand
    - `audio_converter_fill_complex_buffer()` - Python wrapper for Apple's `AudioConverterFillComplexBuffer` API
    - Proper GIL management for thread-safe operation
    - Safe memory allocation/deallocation with automatic cleanup
  - **Enhanced AudioConverter class** (`src/coremusic/objects.py`)
    - `convert_with_callback()` method supporting all conversion types:
      - Sample rate changes (e.g., 44.1kHz → 48kHz, 48kHz → 96kHz)
      - Bit depth changes (e.g., 16-bit → 24-bit)
      - Channel count changes (stereo ↔ mono)
      - Combined conversions (e.g., 44.1kHz stereo → 48kHz mono)
    - Auto-calculation of output packet count based on sample rate ratio
    - Comprehensive documentation with usage examples
  - **Updated utilities** (`src/coremusic/utilities.py`)
    - `convert_audio_file()` now supports ALL conversion types (previously only channel count)
    - Automatically chooses between simple buffer API and callback API based on conversion type
    - Added `_formats_match()` helper function for format comparison
    - Removed NotImplementedError for complex conversions
  - **Comprehensive test coverage**
    - 6 new tests in `test_objects_audio_converter.py`:
      - Sample rate conversion (44.1kHz ↔ 48kHz)
      - Real file sample rate conversion with verification
      - Combined sample rate and channel conversion
      - Auto output packet count calculation
    - 3 previously skipped tests now enabled in `test_utilities.py`:
      - `test_convert_audio_file_sample_rate`
      - `test_convert_audio_file_bit_depth`
      - `test_convert_audio_file_combined_conversions`
    - All tests passing (474 passed, 36 skipped, 0 failures)
    - Duration preservation verified (< 0.000003s error for 2.743s audio)
  - **Documentation** in `docs/COMPLEX_AUDIO_CONVERSION.md`
    - Complete implementation guide with code examples
    - Technical details on callback mechanism and memory management
    - Usage examples and best practices
    - Implementation status updated

## [0.1.4]

### Added

- **Async I/O Support** - Complete async/await support for non-blocking audio operations
  - `AsyncAudioFile` class for asynchronous file reading with chunk streaming
  - `AsyncAudioQueue` class for non-blocking audio queue operations
  - Async context manager support (`async with`) for automatic resource cleanup
  - Async chunk streaming via `read_chunks_async()` - yields audio data without blocking event loop
  - Async packet reading via `read_packets_async()` for fine-grained control
  - NumPy integration with `read_as_numpy_async()` and `read_chunks_numpy_async()`
  - Executor-based approach using `asyncio.to_thread()` for CPU-bound operations
  - Convenience functions: `open_audio_file_async()`, `create_output_queue_async()`
  - Full backward compatibility - existing synchronous API completely untouched
  - Enables concurrent file processing and integration with modern async frameworks (FastAPI, aiohttp, etc.)

- **Comprehensive async test coverage**
  - `test_async_io.py` - 22 async tests covering all async functionality
  - Tests for async file operations (open, close, context managers)
  - Tests for async packet reading and chunk streaming
  - Tests for concurrent file access and processing pipelines
  - Tests for AudioQueue lifecycle management with async operations
  - Tests for NumPy integration with async streaming
  - Real-world async processing pipeline examples
  - 100% pass rate (22/22 tests passing when NumPy available)

- **Demo script for async I/O** (`demo_async_io.py`)
  - 6 comprehensive examples demonstrating async capabilities
  - Basic async file reading with format inspection
  - Streaming large files in chunks without blocking
  - Async AudioQueue creation and playback control
  - Concurrent file processing (batch operations)
  - Real-world processing pipeline (Read → Analyze → Save)
  - NumPy integration for signal processing workflows

- **High-Level Audio Processing Utilities** - Convenient utilities for common audio tasks
  - `AudioAnalyzer` class for audio analysis operations
    - `detect_silence()` - Detect silence regions in audio files with configurable threshold and duration
    - `get_peak_amplitude()` - Extract peak amplitude from audio files
    - `calculate_rms()` - Calculate RMS (Root Mean Square) amplitude
    - `get_file_info()` - Extract comprehensive file metadata (format, duration, sample rate, etc.)
    - All methods support both file paths and AudioFile objects
    - NumPy integration for efficient audio data processing
  - `AudioFormatPresets` class with common audio format presets
    - `wav_44100_stereo()` - CD quality WAV (44.1kHz, 16-bit, stereo)
    - `wav_44100_mono()` - Mono WAV (44.1kHz, 16-bit, mono)
    - `wav_48000_stereo()` - Pro audio WAV (48kHz, 16-bit, stereo)
    - `wav_96000_stereo()` - High-res WAV (96kHz, 24-bit, stereo)
  - `convert_audio_file()` - Simple file format conversion
    - Supports stereo ↔ mono conversion at same sample rate and bit depth
    - Automatic file copy for exact format matches
    - Raises NotImplementedError for complex conversions (guides users to AudioConverter)
  - `batch_convert()` - Batch convert multiple files with glob patterns
    - Supports custom output directory and file extension
    - Optional progress callback for UI integration
    - Automatic directory creation and file overwrite control
  - `trim_audio()` - Extract time ranges from audio files
    - Supports start and end time specification
    - Preserves audio format during trimming
  - `AudioEffectsChain` class for high-level AUGraph management
    - Pythonic wrapper for audio processing graphs with automatic resource management
    - Methods: `add_effect()`, `add_output()`, `connect()`, `open()`, `initialize()`, `start()`, `stop()`
    - Support for method chaining (e.g., `chain.open().initialize().start()`)
    - Context manager support for automatic cleanup
    - Node management with FourCC-based AudioUnit identification
  - `create_simple_effect_chain()` - Convenience function for quick effect chain creation
  - Comprehensive test coverage with 35 tests (28 passing, 7 skipped)
  - Demo script (`tests/demos/demo_utilities.py`) with 10 working examples

- **AudioUnit Name-Based Discovery** - Find and load AudioUnits by name instead of FourCC codes
  - `find_audio_unit_by_name()` - Search for AudioUnits by name (e.g., 'AUDelay')
    - Returns `AudioComponent` object (can create instances directly)
    - Case-insensitive substring matching by default
    - Returns `None` if no matching AudioUnit found
    - Iterates through all available AudioComponents using CoreAudio's `AudioComponentFindNext`
    - Example: `component = cm.find_audio_unit_by_name('AUDelay')`
  - `list_available_audio_units()` - List all available AudioUnits on the system
    - Returns list of dicts with 'name', 'type', 'subtype', 'manufacturer', 'flags'
    - Optional filtering by FourCC type code (e.g., 'aufx' for audio effects)
    - Discovers 676 AudioUnits on typical macOS system
    - Example: `units = cm.list_available_audio_units(filter_type='aufx')`
  - `get_audiounit_names()` - Get simple list of AudioUnit names
    - Returns list of strings (names only, lightweight)
    - Optional filtering by FourCC type code
    - Example: `names = cm.get_audiounit_names()`
  - `AudioEffectsChain.add_effect_by_name()` - Add effects to chain by name
    - Convenience method that automatically finds and adds AudioUnits
    - Example: `delay_node = chain.add_effect_by_name('AUDelay')`
  - Low-level C API wrappers in `src/coremusic/capi.pyx`:
    - `audio_component_copy_name()` - Get human-readable AudioComponent name
    - `audio_component_get_description()` - Get AudioComponentDescription
    - Updated `audio_component_find_next()` with iteration support
  - Proper CoreFoundation memory management with CFRelease for CFStringRef
  - Comprehensive test coverage with 11 tests (100% passing)
  - Documentation in `docs/audiounit_name_lookup.md` with usage examples
  - Demo examples in `tests/demos/demo_utilities.py` (Example 10)

### Fixed

- **Music device test fixture** - Improved error handling for component instantiation
  - Added graceful skip when `AudioComponentInstanceNew` returns status -128
  - Status -128 indicates macOS security restrictions preventing instantiation
  - Tests now properly skip instead of erroring when components cannot be instantiated
  - Improved test robustness across different macOS security configurations
  - Affects `test_audiotoolbox_music_device.py` fixture for music device unit tests

## [0.1.3]

### Added

- **AudioConverter API** - Complete audio format conversion framework
  - Functional API with 13 wrapper functions for AudioConverter operations
  - `audio_converter_new()`, `audio_converter_dispose()`, `audio_converter_convert_buffer()`
  - `audio_converter_get_property()`, `audio_converter_set_property()`, `audio_converter_reset()`
  - 6 property ID getter functions for converter configuration
  - Object-oriented `AudioConverter` class with automatic resource management
  - Context manager support for safe resource cleanup
  - Support for stereo↔mono conversion, bit depth changes, and format conversions

- **ExtendedAudioFile API** - High-level audio file I/O with automatic format conversion
  - Functional API with 14 wrapper functions for ExtendedAudioFile operations
  - `extended_audio_file_open_url()`, `extended_audio_file_create_with_url()`
  - `extended_audio_file_read()`, `extended_audio_file_write()`, `extended_audio_file_dispose()`
  - `extended_audio_file_get_property()`, `extended_audio_file_set_property()`
  - 7 property ID getter functions for file format access
  - Object-oriented `ExtendedAudioFile` class with context manager support
  - Automatic format conversion on read/write via client format property
  - Simplified file I/O compared to lower-level AudioFile API

- **AUGraph API** - Audio Unit graph framework for managing and connecting multiple AudioUnits
  - Functional API with 21 wrapper functions for AUGraph operations
  - `au_graph_new()`, `au_graph_dispose()`, `au_graph_open()`, `au_graph_close()`
  - `au_graph_initialize()`, `au_graph_uninitialize()`, `au_graph_start()`, `au_graph_stop()`
  - `au_graph_add_node()`, `au_graph_remove_node()`, `au_graph_get_node_count()`
  - `au_graph_connect_node_input()`, `au_graph_disconnect_node_input()`, `au_graph_update()`
  - 3 state query functions: `au_graph_is_open()`, `au_graph_is_initialized()`, `au_graph_is_running()`
  - CPU load monitoring: `au_graph_get_cpu_load()`, `au_graph_get_max_cpu_load()`
  - 5 error code getter functions for AUGraph-specific errors
  - Object-oriented `AUGraph` class with automatic resource management
  - Context manager support for safe graph lifecycle management
  - Node management with `AudioComponentDescription` integration
  - Connection management for building audio processing graphs
  - Method chaining support for fluent API (e.g., `graph.open().initialize()`)
  - Properties for state queries: `is_open`, `is_initialized`, `is_running`, `cpu_load`, `node_count`

- **Comprehensive test coverage** for new APIs
  - `test_audiotoolbox_audio_converter.py` - 12 functional API tests
  - `test_audiotoolbox_extended_audio_file.py` - 14 functional API tests
  - `test_objects_audio_converter.py` - 29 object-oriented wrapper tests
  - `test_augraph.py` - 16 AUGraph tests (4 functional, 11 OO, 1 integration)
  - Tests cover creation, conversion, I/O operations, property access, error handling
  - Real-world testing with actual audio files

- **Exception hierarchy** expanded
  - Added `AudioConverterError` for converter-specific exceptions
  - Added `AUGraphError` for graph operation exceptions
  - Proper error propagation with detailed error messages

### Changed

- Enhanced `AudioFormat` class integration with converter APIs
- Improved error handling consistency across audio conversion operations

### Fixed

- **Critical fix for AudioDevice string properties** - Added proper CFStringRef handling
  - Previously, `audio_object_get_property_data()` returned raw CFStringRef pointers instead of actual string content
  - Added new `audio_object_get_property_string()` function that properly dereferences CFStringRef using CoreFoundation APIs
  - Device names, UIDs, and manufacturer strings now correctly use CFStringGetCString for stable, proper string extraction
  - Fixes unstable device name/UID issues where properties returned random garbage on each read
  - All AudioDevice string properties (name, uid, manufacturer, model_uid) now work correctly
- Fixed UID string handling in `AudioDevice._get_property_string()` to strip both leading and trailing null bytes (changed from `.rstrip('\x00')` to `.strip('\x00')`)
- Improved `test_audio_device_manager_find_by_uid` test resilience to handle devices with inconsistent UID encoding

---

## [0.1.2]

### Added

- Object-oriented API layer with automatic resource management
  - Added `CoreAudioObject` base class with proper disposal
  - Added `AudioFile`, `AudioQueue`, `AudioUnit` classes with context manager support
  - Added `MIDIClient`, `MIDIPort` classes for MIDI operations
  - Added `AudioFormat`, `AudioComponentDescription` helper classes
  - Added comprehensive exception hierarchy with `CoreAudioError` base class

- API documentation file (API.md) with implementation status

- Dual API architecture supporting both functional and object-oriented patterns

- Enhanced package structure with proper **init**.py imports

- Comprehensive test coverage for object-oriented APIs
  - Added tests for AudioFile, AudioUnit, AudioQueue OO classes
  - Added MIDI object-oriented API tests
  - Added comprehensive integration tests

### Changed

- Updated README with dual API examples and migration guide
- Enhanced project description to reflect comprehensive framework coverage
- Improved developer experience documentation

### Fixed

- Resource management issues with automatic cleanup via Cython **dealloc**
- Memory leaks in audio operations through proper disposal patterns

---

## [0.1.0] - Previous Release

### Added

- Added namespaces to cimports

- Added a bunch of tests

- Renamed project from `cycoreaudio` to `coremusic`

- Added CoreMIDI wrapper

- Added CoreAudio wrapper
