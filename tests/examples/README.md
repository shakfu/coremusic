# coremusic Examples

Complete, working examples demonstrating coremusic capabilities.

This directory contains both standalone utility scripts and experimental modules that are **not part of the core coremusic package**.

## Directory Structure

```
tests/examples/
├── audio_converter.py    # Audio format converter utility
├── audio_inspector.py    # Audio file inspection utility
├── daw/                  # DAW-like timeline/track/clip framework
│   └── daw.py
├── generative/           # Generative music algorithms
│   ├── generative.py     # Arpeggiator, Euclidean, Melody generators
│   ├── markov.py         # Markov chain analysis and generation
│   ├── bayes.py          # Bayesian network music generation
│   └── cli.py            # CLI commands (not integrated with main CLI)
├── test_daw.py           # Tests for DAW module
├── test_music_bayes.py   # Tests for Bayesian module
├── test_music_generative.py  # Tests for generative module
└── test_music_markov.py  # Tests for Markov module
```

## Quick Start

All standalone examples can be run directly:

```bash
# From the project root
python tests/examples/audio_inspector.py tests/amen.wav

# Or make executable and run
chmod +x tests/examples/audio_inspector.py
./tests/examples/audio_inspector.py tests/amen.wav
```

## Standalone Utilities

### Audio Inspector
**File:** `audio_inspector.py`

Comprehensive audio file inspection tool that displays:
- File information (size, path)
- Format details (sample rate, channels, bit depth)
- Duration and frame count
- Quality classification
- Bitrate calculations

**Usage:**
```bash
python tests/examples/audio_inspector.py audio.wav
```

**Example Output:**
```
======================================================================
Audio File Inspector
======================================================================

FILE INFORMATION
----------------------------------------------------------------------
  Filename:     amen.wav
  File Size:    529.03 KB

FORMAT INFORMATION
----------------------------------------------------------------------
  Format ID:    lpcm
  Sample Rate:  44,100 Hz
  Channels:     2 (Stereo)
  Bit Depth:    16-bit

CLASSIFICATION
----------------------------------------------------------------------
  Quality:      CD Quality
  Bitrate:      1,411 kbps
```

### Audio Converter
**File:** `audio_converter.py`

Convert audio files between different sample rates while preserving quality.

**Features:**
- Sample rate conversion (44.1kHz, 48kHz, 96kHz, etc.)
- Duration preservation verification
- Format validation
- Progress reporting

**Usage:**
```bash
# Convert to 48kHz
python tests/examples/audio_converter.py input.wav output.wav --rate 48000

# Convert to 44.1kHz
python tests/examples/audio_converter.py input.wav output.wav --rate 44100
```

## Experimental Modules

These modules demonstrate advanced use cases but are not production-ready and may have different quality standards than the core wrappers.

### DAW Module (`daw/`)

DAW-like timeline/track/clip framework for arranging audio and MIDI.

### Generative Module (`generative/`)

Generative music algorithms including:
- **generative.py** - Arpeggiator, Euclidean rhythm, Melody generators
- **markov.py** - Markov chain analysis and generation
- **bayes.py** - Bayesian network music generation
- **cli.py** - CLI commands (standalone, not integrated with main CLI)

### Using Experimental Modules

These modules are not installed as part of coremusic. To use them:

1. Add the examples directory to your Python path:
   ```python
   import sys
   sys.path.insert(0, '/path/to/coremusic/tests/examples')
   ```

2. Import the modules:
   ```python
   from daw.daw import Timeline, Track, Clip
   from generative.generative import Arpeggiator, EuclideanGenerator
   ```

## Prerequisites

All examples require:
- coremusic installed and built
- macOS with CoreAudio
- Python 3.6+

Some examples have additional requirements:
- **NumPy**: For audio analysis examples
- **SciPy**: For signal processing examples
- **Matplotlib**: For visualization examples

Install optional dependencies:
```bash
pip install numpy scipy matplotlib
```

## Note

The experimental modules are provided for educational and experimental purposes. They may:
- Have incomplete test coverage
- Contain experimental APIs that may change
- Not follow the same quality standards as the core package

For production use, consider the core coremusic package which provides:
- CoreAudio/AudioToolbox bindings
- CoreMIDI bindings
- Ableton Link integration
- Music theory primitives (coremusic.music.theory)

## License

All examples are provided under the same MIT license as the coremusic project.
