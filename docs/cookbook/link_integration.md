# Ableton Link Integration

Recipes for tempo synchronization using Ableton Link.

## Basic Link Usage

### Create Link Session

Start a Link session for tempo synchronization:

```python
--8<-- "examples/cookbook/link_integration/snippet_01.py:example"
```

### Query Tempo and Beat

Get current tempo and beat position:

```python
--8<-- "examples/cookbook/link_integration/snippet_02.py:example"
```

### Change Tempo

Modify tempo during playback:

```python
--8<-- "examples/cookbook/link_integration/snippet_03.py:example"
```

## AudioPlayer Integration

### Sync AudioPlayer to Link

Synchronize audio playback with Link:

```python
--8<-- "examples/cookbook/link_integration/snippet_04.py:example"
```

### Beat-Accurate Playback Start

Start playback on a specific beat:

```python
--8<-- "examples/cookbook/link_integration/snippet_05.py:example"
```

## MIDI Clock Sync

### Send MIDI Clock

Synchronize external MIDI devices to Link:

```python
--8<-- "examples/cookbook/link_integration/snippet_06.py:example"
```

### Beat-Accurate MIDI Sequencing

Schedule MIDI events at specific beat positions:

```python
--8<-- "examples/cookbook/link_integration/snippet_07.py:example"
```

## Multi-Device Sync

### Sync Multiple Applications

Connect multiple Link-enabled applications:

```python
--8<-- "examples/cookbook/link_integration/snippet_08.py:example"
```

### Transport Control

Control playback state across multiple devices:

```python
--8<-- "examples/cookbook/link_integration/snippet_09.py:example"
```

## Advanced Beat Mapping

### Map Timeline to Beats

Convert between sample positions and beat positions:

```python
--8<-- "examples/cookbook/link_integration/snippet_10.py:example"
```

### Request Beat Alignment

Align beat grid to specific events:

```python
--8<-- "examples/cookbook/link_integration/snippet_11.py:example"
```

### Tempo-Synced Loops

Create loops that stay synchronized:

```python
--8<-- "examples/cookbook/link_integration/snippet_12.py:example"
```

## Complete Example: Drum Machine

Full example of a Link-synchronized drum machine:

```python
--8<-- "examples/cookbook/link_integration/create_drum_pattern.py:example"
```

## Best Practices

### Session Management

Always use context managers:

```python
--8<-- "examples/cookbook/link_integration/snippet_14.py:example"
```

### State Capture and Commit

Capture state, modify, then commit:

```python
--8<-- "examples/cookbook/link_integration/snippet_15.py:example"
```

### Timing Precision

Use microsecond precision for accurate timing:

```python
--8<-- "examples/cookbook/link_integration/snippet_16.py:example"
```

### Thread Safety

Link operations are thread-safe, but use audio thread for time-critical operations:

```python
--8<-- "examples/cookbook/link_integration/snippet_17.py:example"
```

## See Also

- [API Reference](../api/index.md) - Complete API reference
- [AudioUnit Hosting](audiounit_hosting.md) - AudioUnit plugin hosting
- [MIDI Processing](midi_processing.md) - MIDI I/O and processing
- Ableton Link documentation: https://ableton.github.io/link/
