# Audio File Basics

This tutorial covers the fundamentals of working with audio files using coremusic.

## Prerequisites

- coremusic installed and built
- Basic Python knowledge
- An audio file to work with (WAV, AIFF, or MP3)

## Opening and Reading Files

### Using the Object-Oriented API

The recommended approach uses the `AudioFile` class with context managers:

```python
--8<-- "examples/tutorials/audio_file_basics/opening.py:context-manager"
```

**Advantages:**

- Automatic resource cleanup
- Exception-safe
- Pythonic and readable

### Using the Functional API

For more control, use the functional API:

```python
--8<-- "examples/tutorials/audio_file_basics/opening.py:manual"
```

## Getting File Information

### Duration and Format

```python
--8<-- "examples/tutorials/audio_file_basics/properties.py:basic"
```

**Output example:**

```text
Duration: 2.74 seconds
Total frames: 120960
Sample rate: 44100.0 Hz
Channels: 2
Bit depth: 16
Format ID: lpcm
```

### Detailed Format Information

```python
--8<-- "examples/tutorials/audio_file_basics/properties.py:format"
```

## Reading Audio Data

### Reading Packets

Audio data is read in packets (frames):

```python
--8<-- "examples/tutorials/audio_file_basics/reading.py:packets"
```

**Parameters:**

- `start_packet`: Starting packet number (0-indexed)
- `num_packets`: Number of packets to read

**Returns:**

- `data`: Raw audio data as bytes
- `packets_read`: Actual number of packets read

### Reading in Chunks

For large files, read in chunks to manage memory:

```python
--8<-- "examples/tutorials/audio_file_basics/reading.py:chunks"
```

### Reading Entire File

For smaller files, read everything at once:

```python
--8<-- "examples/tutorials/audio_file_basics/reading.py:whole-file"
```

## Working with Different Formats

### Detecting Format Type

```python
--8<-- "examples/tutorials/audio_file_basics/format_checks.py:detect"
```

### Checking Format Properties

```python
--8<-- "examples/tutorials/audio_file_basics/format_checks.py:properties"
```

## Error Handling

### Handling File Errors

Always handle potential errors:

```python
--8<-- "examples/tutorials/audio_file_basics/error_handling.py:safe-open"
```

### Validating Audio Files

```python
--8<-- "examples/tutorials/audio_file_basics/error_handling.py:validate"
```

## Complete Example

### Audio File Inspector

A complete tool that inspects audio files:

```python
--8<-- "examples/tutorials/audio_file_basics/inspect_audio.py:example"
```

Save as `inspect_audio.py` and run:

```bash
python inspect_audio.py audio.wav
```

**Example output:**

```text
Inspecting: audio.wav
File size: 529.03 KB

Format Information:
  Format ID: lpcm
  Sample Rate: 44100.0 Hz
  Channels: 2
  Bit Depth: 16
  Bytes/Frame: 4

Duration Information:
  Total Frames: 120,960
  Duration: 2.74 seconds
  Duration: 0.05 minutes

Classification:
  Quality: CD Quality
  Channel Type: Stereo
  Bitrate: 1411 kbps
```

## Next Steps

Now that you understand audio file basics, explore:

- [File Operations Cookbook](../cookbook/file_operations.md) - Common file operation recipes

## See Also

- [AudioFile API Reference](../api/audio_file.md) - Complete AudioFile API reference
