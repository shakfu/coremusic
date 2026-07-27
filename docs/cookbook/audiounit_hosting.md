# AudioUnit Plugin Hosting

Recipes for hosting and controlling AudioUnit plugins.

## Plugin Discovery

### List Available Plugins

Discover all AudioUnit plugins on the system:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_01.py:example"
```

### Load Plugin by Name

Load a specific plugin by name:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_02.py:example"
```

## Parameter Control

### List and Control Parameters

Discover and control plugin parameters:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_03.py:example"
```

### Automate Parameters

Automate parameter changes over time:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_04.py:example"
```

## Preset Management

### Factory Presets

Browse and load factory presets:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_05.py:example"
```

### User Presets

Save and load custom user presets:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_06.py:example"
```

### Export and Import Presets

Share presets between systems:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_07.py:example"
```

## Audio Format Support

### Custom Audio Formats

Process audio in different formats:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_08.py:example"
```

### Supported Formats

All supported audio formats:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_09.py:example"
```

## Plugin Chains

### Basic Chain

Create a simple plugin chain:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_10.py:example"
```

### Advanced Chain with Wet/Dry Mix

Control the balance between processed and original signal:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_11.py:example"
```

### Dynamic Chain Manipulation

Modify chain during processing:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_12.py:example"
```

## MIDI Control (Instruments)

### Basic Note Control

Play notes with AudioUnit instruments:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_13.py:example"
```

### Program Changes

Change instrument sounds using General MIDI:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_14.py:example"
```

### MIDI Controllers

Control parameters using MIDI CC messages:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_15.py:example"
```

### Pitch Bend

Apply pitch bend to notes:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_16.py:example"
```

### Multi-Channel Performance

Use multiple MIDI channels for complex arrangements:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_17.py:example"
```

## Complete Example: Reverb Effect

Full example processing audio with reverb:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_18.py:example"
```

## Best Practices

### Resource Management

Always use context managers for automatic cleanup:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_19.py:example"
```

### Error Handling

Handle plugin errors gracefully:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_20.py:example"
```

### Performance

Tips for optimal performance:

```python
--8<-- "examples/cookbook/audiounit_hosting/snippet_21.py:example"
```

## See Also

- [API Reference](../api/index.md) - Complete API reference
- [File Operations](file_operations.md) - File I/O recipes
- [Link Integration](link_integration.md) - Ableton Link tempo sync
