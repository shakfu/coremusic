# Examples Gallery

Complete, working examples demonstrating coremusic capabilities.

## Example Categories

### Basic Examples

Essential examples for getting started:

- **Audio Player**: Simple audio file playback
- **Audio Converter**: Convert between audio formats
- **Audio Inspector**: Display detailed file information
- **MIDI Monitor**: Monitor MIDI input

### Audio Processing

Audio processing and manipulation:

- **Real-time Processor**: Process audio in real-time
- **Batch Converter**: Convert multiple files
- **Audio Analyzer**: Analyze audio characteristics
- **Waveform Generator**: Generate audio waveforms

### AudioUnit Examples

Working with AudioUnits:

- **AudioUnit Explorer**: Discover available AudioUnits
- **Effect Chain**: Chain multiple audio effects
- **Custom Processor**: Create custom AudioUnit
- **Parameter Controller**: Automate parameters

### MIDI Examples

MIDI processing and routing:

- **MIDI Router**: Route MIDI between devices
- **MIDI Transformer**: Transform MIDI messages
- **Virtual Keyboard**: Create virtual MIDI keyboard
- **MIDI Recorder**: Record MIDI sequences

### Advanced Examples

Advanced techniques and integration:

- **Multi-channel Processor**: Handle surround audio
- **Low Latency Streamer**: Minimal latency streaming
- **Audio Visualizer**: Real-time visualization
- **SciPy Integration**: Signal processing with SciPy

## Where the Examples Live

Every snippet in this documentation is a runnable program under
[`examples/`](https://github.com/shakfu/coremusic/tree/main/examples), arranged
by the page that includes it. They are executed by the test suite, so a snippet
that no longer runs fails the build rather than reaching you.

```bash
# Run one
python examples/tutorials/midi_basics/send_note.py

# Run all of them, as the test suite does
pytest tests/test_examples.py
```

Larger standalone programs live elsewhere in the repository:

- [`demos/`](https://github.com/shakfu/coremusic/tree/main/demos) - four
  end-to-end demos: an AudioUnit effect chain, MIDI rendered to WAV, a
  real-time tone, and a Link-synchronised sequencer. `make demos` runs them.
- [`tests/examples/`](https://github.com/shakfu/coremusic/tree/main/tests/examples) -
  utilities (audio inspector, converter) and experimental modules (a DAW-style
  timeline, generative algorithms) that are not part of the package.

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

## Example Template

Use this template for creating new examples:

```python
--8<-- "examples/examples/index/template.py:example"
```

## Quick Reference

### Common Example Patterns

**Simple audio playback:**

```python
--8<-- "examples/quickstart/play_audio.py:player"
```

**Format conversion:**

```python
--8<-- "examples/examples/index/convert_format.py:example"
```

**MIDI routing:**

```python
--8<-- "examples/examples/index/midi_router.py:example"
```

**Real-time processing:**

```python
--8<-- "examples/index/audiounit.py:example"
```

## Contributing Examples

We welcome example contributions! To add an example:

1. Create a standalone, working script
2. Add comprehensive docstrings
3. Include usage instructions
4. Handle errors gracefully
5. Add to the examples directory
6. Update this documentation

Example Guidelines:

- **Clear purpose**: Each example should demonstrate one concept
- **Self-contained**: Minimize external dependencies
- **Well-commented**: Explain non-obvious code
- **Error handling**: Handle common errors
- **Usage help**: Print usage if arguments are missing

## See Also

- [Step-by-step tutorials](../tutorials/index.md)
- [Recipe collection](../cookbook/index.md)
- [API reference](../api/index.md)
