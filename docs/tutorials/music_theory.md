# Music Theory Primitives

This tutorial covers the music theory primitives available in the
`coremusic.music` module for working with notes, intervals, scales,
and chords.

## Overview

The music module provides:

- **Note**: Musical note with MIDI number, name, octave, and frequency
- **Interval**: Distance between two notes with quality (major, minor, perfect)
- **Scale**: 25+ scale types including modes and exotic scales
- **Chord**: 35+ chord types from triads to extended/altered chords

## Music Theory Basics

### Notes

The `Note` class represents a musical note:

```python
--8<-- "examples/tutorials/music_theory/notes.py:example"
```

### Intervals

The `Interval` class represents the distance between two notes:

```python
--8<-- "examples/tutorials/music_theory/intervals.py:example"
```

### Scales

The `Scale` class provides 25+ scale types:

```python
--8<-- "examples/tutorials/music_theory/scales.py:example"
```

### Chords

The `Chord` class provides 35+ chord types:

```python
--8<-- "examples/tutorials/music_theory/chords.py:example"
```

### Key Signatures

Work with key signatures and the circle of fifths:

```python
--8<-- "examples/tutorials/music_theory/key_signatures.py:example"
```

### Utility Functions

Convert between note names and MIDI numbers:

```python
--8<-- "examples/tutorials/music_theory/conversion.py:example"
```

## See Also

- [Async Audio](async_audio.md) - Asynchronous audio processing
- [Audio File Basics](audio_file_basics.md) - Working with audio files
