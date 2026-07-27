# Getting Started

This guide will help you get started with coremusic, from installation through your first audio and MIDI applications.

## Prerequisites

Before installing coremusic, ensure you have:

- **macOS**: CoreAudio and CoreMIDI frameworks are macOS-specific
- **Python 3.10+**: Python 3.10 or higher is required
- **Xcode Command Line Tools**: Required for framework headers

Install Xcode Command Line Tools:

```bash
xcode-select --install
```

## Installation

### From PyPI (Recommended)

```bash
pip install coremusic
```

Or with uv:

```bash
uv add coremusic
```

### From Source

1. Clone the repository:

   ```bash
   git clone https://github.com/shakfu/coremusic.git
   cd coremusic
   ```

2. Install with uv (recommended):

   ```bash
   uv sync
   ```

3. Build the extension:

   ```bash
   make
   ```

4. Verify installation:

   ```bash
   make test
   ```

## Understanding the Dual API

coremusic provides two complementary APIs that can be used together or independently:

### Functional API (Traditional)

The functional API provides direct access to CoreAudio C functions:

**Advantages:**

- Direct mapping to CoreAudio C APIs
- Maximum performance and control
- Familiar interface for CoreAudio developers
- Fine-grained resource management

**Use when:**

- Maximum performance is critical
- Porting existing CoreAudio C code
- Need fine-grained control over resource lifetimes
- Building low-level audio processing components

**Example:**

```python
--8<-- "examples/getting_started/dual_api.py:functional"
```

### Object-Oriented API (Modern)

The object-oriented API provides Pythonic wrappers with automatic resource management:

**Advantages:**

- Automatic cleanup with context managers
- Type safety with proper Python classes
- Pythonic patterns (properties, iteration, operators)
- Resource safety preventing memory leaks
- IDE autocompletion and type hints

**Use when:**

- Building new applications
- Rapid prototyping and development
- Team development where code safety is important
- Working with complex audio workflows

**Example:**

```python
--8<-- "examples/getting_started/dual_api.py:object-oriented"
```

## Your First Audio Application

### Audio File Information Tool

Let's create a simple tool to display audio file information:

```python
--8<-- "examples/getting_started/audio_info.py:example"
```

Save this as `audio_info.py` and run:

```bash
python audio_info.py path/to/audio.wav
```

### Simple Audio Player

Create a basic audio player:

```python
--8<-- "examples/getting_started/play_audio.py:example"
```

### Audio/MIDI Synchronization

Use AudioClock for synchronizing audio and MIDI with precise timing:

```python
--8<-- "examples/getting_started/clock_demo.py:example"
```

## Your First MIDI Application

### MIDI Device Lister

Create a tool to list all MIDI devices:

```python
--8<-- "examples/getting_started/list_midi_devices.py:example"
```

### Simple MIDI Monitor

Create a MIDI monitor that displays incoming messages:

```python
--8<-- "examples/getting_started/monitor_midi.py:example"
```

## Next Steps

Now that you've created your first applications, explore:

- [Tutorials](tutorials/index.md) - Step-by-step tutorials for common tasks
- [Cookbook](cookbook/index.md) - Ready-to-use recipes for audio processing
- [Examples](examples/index.md) - Complete example applications
- [API Reference](api/index.md) - Detailed API reference

## Common Patterns

### Context Managers

Always use context managers for automatic resource cleanup:

```python
--8<-- "examples/getting_started/patterns.py:context-managers"
```

### Error Handling

Handle errors appropriately:

```python
--8<-- "examples/getting_started/patterns.py:error-handling"
```

### Resource Management

When using the functional API, always clean up resources:

```python
--8<-- "examples/getting_started/patterns.py:functional-cleanup"
```

## Troubleshooting

### Build Errors

If you encounter build errors:

1. Ensure Xcode Command Line Tools are installed:

   ```bash
   xcode-select --install
   ```

2. Clean and rebuild:

   ```bash
   make clean
   make
   ```

3. If using uv, ensure dependencies are synced:

   ```bash
   uv sync --reinstall-package coremusic
   ```

### Runtime Errors

**"Module not found" errors:**

- Ensure you're running Python from the project directory
- Verify the extension was built: `ls src/coremusic/*.so`

**Audio playback issues:**

- Check audio file format is supported (WAV, AIFF, MP3, etc.)
- Verify audio file exists and is not corrupted
- Ensure macOS audio system is working

**MIDI issues:**

- Check MIDI devices are connected and powered on
- Verify MIDI devices appear in Audio MIDI Setup app
- Ensure no other application is exclusively using MIDI devices

## Getting Help

If you encounter issues:

1. Check the [API Reference](api/index.md) for detailed function documentation
2. Review the [Examples](examples/index.md) for working code samples
3. Search existing issues on GitHub
4. Create a new issue with:
   - Your macOS version
   - Python version
   - Complete error message
   - Minimal code to reproduce the issue
