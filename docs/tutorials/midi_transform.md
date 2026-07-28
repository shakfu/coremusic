# MIDI Transformation Pipeline

This tutorial covers the MIDI transformation pipeline for loading, transforming,
and saving MIDI files using composable transformers.

## Overview

The `coremusic.midi.transform` module provides a pipeline architecture for
processing MIDI sequences. Transformers can be chained together to create
complex processing workflows.

**Key Features:**

- Load and save Standard MIDI Files
- Composable transformer pipeline
- 15+ built-in transformers for pitch, time, velocity, and filtering
- Reproducible results with seed parameters
- Fluent API for easy chaining

## Quick Start

```python
--8<-- "examples/tutorials/midi_transform/quick_start.py:example"
```

## Pipeline Basics

### Creating a Pipeline

A pipeline chains multiple transformers together:

```python
--8<-- "examples/tutorials/midi_transform/pipelines.py:creating"
```

### Using Individual Transformers

Each transformer can be used standalone:

```python
--8<-- "examples/tutorials/midi_transform/pipelines.py:individual"
```

## Pitch Transformers

### Transpose

Shift all notes by a fixed number of semitones:

```python
--8<-- "examples/tutorials/midi_transform/pitch.py:transpose"
```

### Invert

Mirror melody around a pivot note:

```python
--8<-- "examples/tutorials/midi_transform/pitch.py:invert"
```

### Harmonize

Add parallel intervals to create harmonies:

```python
--8<-- "examples/tutorials/midi_transform/pitch.py:harmonize"
```

## Time Transformers

### Quantize

Snap timing to a grid with optional swing:

```python
--8<-- "examples/tutorials/midi_transform/timing.py:quantize"
```

### TimeStretch

Change tempo by stretching or compressing time:

```python
--8<-- "examples/tutorials/midi_transform/timing.py:stretch"
```

### TimeShift

Move all events forward or backward in time:

```python
--8<-- "examples/tutorials/midi_transform/timing.py:shift"
```

### Reverse

Reverse the sequence (retrograde):

```python
--8<-- "examples/tutorials/midi_transform/timing.py:reverse"
```

## Velocity Transformers

### VelocityScale

Scale velocities by factor or to a range:

```python
--8<-- "examples/tutorials/midi_transform/velocity.py:scale"
```

### VelocityCurve

Apply a velocity curve for dynamic shaping:

```python
--8<-- "examples/tutorials/midi_transform/velocity.py:curve"
```

### Humanize

Add human-like timing and velocity variation:

```python
--8<-- "examples/tutorials/midi_transform/velocity.py:humanize"
```

## Filter Transformers

### NoteFilter

Filter notes by pitch, velocity, or channel:

```python
--8<-- "examples/tutorials/midi_transform/filters.py:notes"
```

### EventTypeFilter

Filter by MIDI event type:

```python
--8<-- "examples/tutorials/midi_transform/filters.py:events"
```

## Track Transformers

### ChannelRemap

Remap MIDI channels:

```python
--8<-- "examples/tutorials/midi_transform/tracks.py:channel"
```

### TrackMerge

Merge all tracks into one:

```python
--8<-- "examples/tutorials/midi_transform/tracks.py:merge"
```

### Arpeggiate

Convert chords to arpeggios:

```python
--8<-- "examples/tutorials/midi_transform/tracks.py:arpeggio"
```

## Convenience Functions

For common operations, convenience functions are available:

```python
--8<-- "examples/tutorials/midi_transform/functions.py:example"
```

## Complete Example

Here's a complete workflow processing a MIDI file:

```python
--8<-- "examples/tutorials/midi_transform/complete.py:example"
```

## See Also

- [Music Theory](music_theory.md) - Music theory fundamentals
- `coremusic.midi.utilities` - MIDI file I/O
- `coremusic.midi.link` - Ableton Link integration
