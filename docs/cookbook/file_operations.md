# File Operations

Common recipes for audio file operations.

## Reading Audio Files

### Read Entire File

```python
--8<-- "examples/cookbook/file_operations/read_audio_file.py:example"
```

### Read File in Chunks

For large files, read in manageable chunks:

```python
--8<-- "examples/cookbook/file_operations/read_audio_chunks.py:example"
```

### Read Specific Section

Read a specific time range from an audio file:

```python
--8<-- "examples/cookbook/file_operations/read_time_range.py:example"
```

## Writing Audio Files

### Convert to NumPy Array

```python
--8<-- "examples/cookbook/file_operations/audio_to_numpy.py:example"
```

## File Information

### Get File Metadata

```python
--8<-- "examples/cookbook/file_operations/get_audio_metadata.py:example"
```

### Write Metadata Tags

Write metadata tags (title, artist, album, etc.) to audio files. The file must
be opened with `writable=True`. Writable formats include CAF and AIFF (not WAV).

```python
--8<-- "examples/cookbook/file_operations/metadata_tags.py:example"
```

To write metadata to a WAV file, convert it to CAF first:

```bash
afconvert -f caff -d LEI16 input.wav output.caf
```

### Compare Audio Files

```python
--8<-- "examples/cookbook/file_operations/compare_audio_files.py:example"
```

## File Validation

### Validate Audio File

```python
--8<-- "examples/cookbook/file_operations/validate_audio_file.py:example"
```

### Check Format Support

```python
--8<-- "examples/cookbook/file_operations/is_format_supported.py:example"
```

## File Utilities

### Calculate Audio Statistics

```python
--8<-- "examples/cookbook/file_operations/calculate_audio_stats.py:example"
```

### Detect Silence

```python
--8<-- "examples/cookbook/file_operations/detect_silence.py:example"
```

### Format Human-Readable Info

```python
--8<-- "examples/cookbook/file_operations/format_duration.py:example"
```

## See Also

- [Audio File Basics](../tutorials/audio_file_basics.md) - Audio file fundamentals
- [AudioFile API](../api/audio_file.md) - AudioFile API reference
