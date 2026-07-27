# API Quickstart

A rapid introduction to coremusic's most commonly used APIs.

## Import Patterns

```python
--8<-- "examples/api/quickstart/snippet_01.py:example"
```

## Audio File Operations

### Read Audio File

```python
--8<-- "examples/api/quickstart/snippet_02.py:example"
```

### Get Audio Format

```python
--8<-- "examples/api/quickstart/snippet_03.py:example"
```

### Extended Audio File (Format Conversion)

```python
--8<-- "examples/api/quickstart/snippet_04.py:example"
```

## AudioUnit Operations

### Create Default Output

```python
--8<-- "examples/api/quickstart/snippet_05.py:example"
```

### Find and Create AudioUnit

```python
--8<-- "examples/api/quickstart/snippet_06.py:example"
```

## MIDI Operations

### List MIDI Devices

```python
--8<-- "examples/api/quickstart/snippet_07.py:example"
```

### Create MIDI Client

```python
--8<-- "examples/api/quickstart/snippet_08.py:example"
```

## Audio Queue Operations

### Create Output Queue

```python
--8<-- "examples/api/quickstart/snippet_09.py:example"
```

## Constants Usage

### Using Enum Constants

```python
--8<-- "examples/api/quickstart/snippet_10.py:example"
```

### Constants in API Calls

```python
--8<-- "examples/api/quickstart/snippet_11.py:example"
```

## Async Operations

### Async File Reading

```python
--8<-- "examples/api/quickstart/snippet_12.py:example"
```

## Error Handling

### Exception Hierarchy

```python
--8<-- "examples/api/quickstart/snippet_13.py:example"
```

## NumPy Integration

### Check Availability

```python
--8<-- "examples/api/quickstart/snippet_14.py:example"
```

### Memory-Mapped Files

```python
--8<-- "examples/api/quickstart/snippet_15.py:example"
```

## Quick Reference Table

| Class | Purpose |
|-------|---------|
| `cm.AudioFile` | Read audio files (WAV, AIFF, MP3, etc.) |
| `cm.ExtendedAudioFile` | Read with format conversion |
| `cm.AudioFormat` | Audio format description |
| `cm.AudioUnit` | Audio processing unit |
| `cm.AudioQueue` | Audio playback/recording queue |
| `cm.AudioConverter` | Convert between formats |
| `cm.MIDIClient` | MIDI client connection |
| `cm.AudioPlayer` | High-level audio playback |
| `cm.AsyncAudioFile` | Async file operations |
| `cm.AsyncAudioQueue` | Async queue operations |

## See Also

- [Full API reference](index.md)
- [Complete getting started guide](../getting_started.md)
- [Step-by-step tutorials](../tutorials/index.md)
