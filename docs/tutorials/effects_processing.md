# Effects Processing

This tutorial covers audio effects processing using AudioUnits with coremusic.

## Prerequisites

- coremusic installed and built
- Basic Python knowledge
- Audio files to process

## Understanding AudioUnits

AudioUnits are macOS audio plugins that process audio:

- **Effects (aufx)**: Modify audio (reverb, delay, EQ, compression)
- **Instruments (aumu)**: Generate audio from MIDI
- **Generators (augn)**: Generate audio (test tones, noise)
- **Mixers (aumx)**: Mix multiple audio streams

## Discovering Available Effects

### List All AudioUnits

```python
--8<-- "examples/tutorials/effects_processing/discover.py:list-all"
```

### List Effects Only

```python
--8<-- "examples/tutorials/effects_processing/discover.py:list-effects"
```

### Find Specific Effect

```python
--8<-- "examples/tutorials/effects_processing/discover.py:find"
```

### Using the CLI

```bash
# List all plugins
coremusic plugin list

# List effects only
coremusic plugin list --type aufx

# Get plugin info
coremusic plugin info AUDelay
```

## Creating an Effects Chain

### Simple Effect Chain

```python
--8<-- "examples/tutorials/effects_processing/chains.py:simple"
```

### Multiple Effects Chain

```python
--8<-- "examples/tutorials/effects_processing/chains.py:multi"
```

### Using Effect Descriptors

```python
--8<-- "examples/tutorials/effects_processing/chains.py:descriptors"
```

`AudioEffectsChain` builds an AUGraph, which runs live and feeds the output
device. When you want to push your own blocks through the same effects and get
the processed audio back, use `AudioUnitChain` from the plugin host:

```python
--8<-- "examples/tutorials/effects_processing/chains.py:plugin-chain"
```

## Processing Audio Files

### Using the CLI

```bash
# Apply effect to audio file
coremusic plugin process AUDelay input.wav -o output.wav

# Use a preset
coremusic plugin process AUDelay input.wav -o output.wav --preset "Long Delay"

# List available presets
coremusic plugin preset list AUDelay
```

### Programmatic Processing

```python
--8<-- "examples/tutorials/effects_processing/process_file.py:example"
```

## Configuring Effect Parameters

### Listing Parameters

```python
--8<-- "examples/tutorials/effects_processing/parameters.py:list"
```

### Setting Parameters

```python
--8<-- "examples/tutorials/effects_processing/parameters.py:configure"
```

### Using Presets

```python
--8<-- "examples/tutorials/effects_processing/parameters.py:presets"
```

## Real-Time Effects Processing

```python
--8<-- "examples/tutorials/effects_processing/realtime.py:example"
```

## Common Effect Configurations

### Reverb

```python
--8<-- "examples/tutorials/effects_processing/common_effects.py:reverb"
```

### Delay

```python
--8<-- "examples/tutorials/effects_processing/common_effects.py:delay"
```

### EQ

```python
--8<-- "examples/tutorials/effects_processing/common_effects.py:eq"
```

## Complete Example: Audio Processor

```python
--8<-- "examples/tutorials/effects_processing/audio_processor.py:example"
```

## Next Steps

- [AudioUnit Hosting Cookbook](../cookbook/audiounit_hosting.md) - Advanced AudioUnit hosting
- [Audio Playback](audio_playback.md) - Play processed audio
- [Real-Time Audio Cookbook](../cookbook/real_time_audio.md) - Real-time processing techniques

## See Also

- [API Reference](../api/index.md) - Complete API reference
- [CLI Guide](../guides/cli.md) - CLI plugin commands
