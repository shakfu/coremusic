# Audio File Operations

The audio file module provides functionality for reading and writing audio files in various formats.

## Object-Oriented API

### AudioFile Class

The `AudioFile` class provides high-level audio file operations with automatic resource management.

```python
--8<-- "examples/api/audio_file/snippet_01.py:example"
```

### Class Reference

::: coremusic.audio.AudioFile
    options:
      members: true
      show_bases: true

### AudioFormat Class

The `AudioFormat` class represents audio stream format information.

```python
--8<-- "examples/api/audio_file/snippet_02.py:example"
```

### Class Reference

::: coremusic.audio.AudioFormat
    options:
      members: true
      show_bases: true

## Functional API

The functional API provides direct access to CoreAudio file operations through
the `coremusic.capi` module.

!!! note
    The object-oriented `AudioFile` API is recommended for most use cases.
    Use the functional API only when you need fine-grained control.

### Opening and Closing Files

**Example:**

```python
--8<-- "examples/api/audio_file/snippet_03.py:example"
```

### Reading Audio Data

**Example:**

```python
--8<-- "examples/api/audio_file/snippet_04.py:example"
```

### File Properties

**Example:**

```python
--8<-- "examples/api/audio_file/snippet_05.py:example"
```

## Supported Formats

coremusic supports all audio formats supported by CoreAudio, including:

### Common Formats

- **WAV** (Waveform Audio File Format)
- **AIFF** (Audio Interchange File Format)
- **MP3** (MPEG-1 Audio Layer 3)
- **AAC** (Advanced Audio Coding)
- **ALAC** (Apple Lossless Audio Codec)
- **FLAC** (Free Lossless Audio Codec)

### Format IDs

Common format IDs (FourCC codes):

- `'lpcm'` - Linear PCM (uncompressed)
- `'aac '` - AAC
- `'.mp3'` - MP3
- `'alac'` - Apple Lossless
- `'flac'` - FLAC

### Format Flags

For Linear PCM, common format flags include:

- Float vs Integer
- Big Endian vs Little Endian
- Packed vs Aligned
- Signed vs Unsigned

Use the provided constant functions to get appropriate flags:

```python
--8<-- "examples/api/audio_file/snippet_06.py:example"
```

## Examples

### Read Entire Audio File

```python
--8<-- "examples/api/audio_file/read_audio_file.py:example"
```

### Process Audio in Chunks

```python
--8<-- "examples/api/audio_file/process_audio_chunks.py:example"
```

### Audio Format Conversion

```python
--8<-- "examples/api/audio_file/convert_audio_format.py:example"
```

## See Also

- [Common file operation recipes](../cookbook/file_operations.md)
