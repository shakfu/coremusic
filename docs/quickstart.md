# Quick Start Guide

Get up and running with coremusic in 5 minutes.

## Installation

```bash
# Install from PyPI
pip install coremusic

# Or with uv
uv add coremusic
```

## Verify Installation

```bash
# Check CLI works
coremusic --version

# List audio devices
coremusic device list

# List MIDI devices
coremusic midi list
```

## Your First Script

Create a file called `hello_audio.py`:

```python
--8<-- "examples/quickstart/hello_audio.py:example"
```

Run it:

```bash
python hello_audio.py
```

## Common Tasks

### Play Audio

**Command Line:**

```bash
coremusic audio play music.wav
```

**Python:**

```python
--8<-- "examples/quickstart/play_audio.py:player"
```

### Record Audio

**Command Line:**

```bash
coremusic audio record -o recording.wav --duration 10
```

**Python:**

```python
--8<-- "examples/quickstart/record_audio.py:example"
```

### Convert Audio Format

**Command Line:**

```bash
# Convert to mono WAV
coremusic convert file input.wav output.wav --channels 1
```

**Python:**

```python
--8<-- "examples/quickstart/convert_audio.py:example"
```

### Apply Audio Effects

**Command Line:**

```bash
# Add reverb
coremusic plugin process AUReverb2 input.wav -o output.wav
```

**Python:**

```python
--8<-- "examples/quickstart/effects_chain.py:example"
```

### List Available Plugins

**Command Line:**

```bash
coremusic plugin list
```

**Python:**

```python
--8<-- "examples/quickstart/list_plugins.py:example"
```

### Monitor MIDI Input

**Command Line:**

```bash
coremusic midi monitor
```

**Python:**

```python
--8<-- "examples/quickstart/list_midi_sources.py:example"
```

### Send MIDI Notes

```python
--8<-- "examples/quickstart/send_midi_notes.py:example"
```

## API Patterns

### Context Managers (Recommended)

```python
--8<-- "examples/quickstart/api_patterns.py:context-manager"
```

### Error Handling

```python
--8<-- "examples/quickstart/api_patterns.py:error-handling"
```

### NumPy Integration

```python
--8<-- "examples/quickstart/api_patterns.py:numpy"
```

## CLI Command Reference

```text
Audio Commands:
  coremusic audio play <file>              Play audio file
  coremusic audio record -o <file>         Record audio

Device Commands:
  coremusic device list                    List audio devices
  coremusic device info <name>             Device details

Plugin Commands:
  coremusic plugin list                    List AudioUnits
  coremusic plugin process <name> <file>   Apply effect

MIDI Commands:
  coremusic midi list                      List MIDI devices
  coremusic midi monitor             Monitor MIDI input

Analysis Commands:
  coremusic audio info <file>              Audio file info
  coremusic analyze loudness <file>        LUFS measurement
```

## Next Steps

- [Getting Started](getting_started.md) - Detailed installation and setup
- [Tutorials](tutorials/index.md) - Step-by-step tutorials
- [Cookbook](cookbook/index.md) - Ready-to-use recipes
- [API Reference](api/index.md) - Complete API reference

## Getting Help

- Check the [API Reference](api/index.md) for detailed documentation
- See [Tutorials](tutorials/index.md) for worked examples
- Report issues at https://github.com/shakfu/coremusic/issues
