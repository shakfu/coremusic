# Cookbook

Ready-to-use recipes for common audio and MIDI processing tasks.

## Recipe Overview

### File Operations

- **File I/O**: Common file reading and writing patterns
- **Batch Processing**: Process multiple files efficiently
- **Format Detection**: Detect and validate audio formats
- **Format Conversion**: Convert between audio formats

### Audio Processing

- **Volume Control**: Normalize audio, adjust levels
- **Fades**: Apply fade in/out effects
- **Resampling**: Change sample rates with automatic conversion
- **Mixing**: Mix multiple audio tracks
- **Slicing**: Split audio into chunks
- **Concatenation**: Merge multiple audio files

### Real-Time Audio

- **Recording**: Capture audio from input devices
- **Low-Latency Playback**: Minimal latency audio output
- **Level Monitoring**: Real-time audio level metering
- **Effects Processing**: Apply real-time audio effects

### AudioUnit Plugin Hosting

- **Plugin Discovery**: Find and list available AudioUnit plugins
- **Parameter Control**: Control plugin parameters and automation
- **Preset Management**: Save, load, and share plugin presets
- **Audio Format Support**: Process audio in multiple formats (float32/64, int16/32)
- **Plugin Chains**: Create multi-effect chains with automatic routing
- **MIDI Control**: Control instrument plugins with MIDI messages

### Ableton Link Integration

- **Tempo Synchronization**: Sync tempo across multiple applications
- **Beat-Accurate Timing**: Schedule events on specific beats
- **AudioPlayer Sync**: Synchronize audio playback with Link
- **MIDI Clock Sync**: Send MIDI clock messages synchronized to Link
- **Transport Control**: Control playback state across devices
- **Multi-Device Sync**: Connect multiple Link-enabled applications

### MIDI Processing

- **Device Discovery**: Find and list MIDI sources and destinations
- **MIDI Input**: Receive and process MIDI messages
- **MIDI Output**: Send MIDI messages to devices
- **MIDI Routing**: Route MIDI between devices and channels
- **MIDI Transformation**: Transpose, scale velocity, and transform MIDI data
- **MIDI Recording**: Record and playback MIDI sequences

### Integration

- **NumPy Integration**: Work with NumPy arrays for signal processing
- **SciPy Integration**: Use SciPy for advanced DSP
- **Async I/O**: Asynchronous audio file operations

## Quick Reference

### Common Patterns

**Read audio file:**

```python
--8<-- "examples/cookbook/index/common_patterns.py:read-file"
```

**Load AudioUnit plugin:**

```python
--8<-- "examples/cookbook/index/common_patterns.py:load-plugin"
```

**Create plugin chain:**

```python
--8<-- "examples/cookbook/index/common_patterns.py:plugin-chain"
```

**Sync with Ableton Link:**

```python
--8<-- "examples/cookbook/index/common_patterns.py:link"
```

**Send MIDI:**

```python
--8<-- "examples/cookbook/index/common_patterns.py:send-midi"
```

## Tips and Best Practices

### Performance

- Use chunk processing for large files
- Pre-allocate buffers when possible
- Minimize memory copies
- Use appropriate buffer sizes (1024-4096 frames typical)

### Resource Management

- Always use context managers (`with` statements)
- Close resources explicitly if not using context managers
- Handle exceptions to ensure cleanup
- Dispose of MIDI clients and ports properly

### Error Handling

- Check file existence before opening
- Validate audio formats
- Handle CoreAudio errors gracefully
- Provide meaningful error messages

### Thread Safety

- AudioUnits are not thread-safe by default
- Use locks when accessing shared resources
- Process audio on dedicated threads
- Avoid blocking the audio callback thread
