# API Reference

Complete API reference for coremusic. The package provides both functional and object-oriented APIs.

!!! note
    The object-oriented API is recommended for new applications due to automatic resource management and Pythonic interfaces.

## Object-Oriented API

High-level Pythonic wrappers with automatic resource management.

### AudioFile Class

::: coremusic.audio.AudioFile
    options:
      members: true
      show_bases: true

### AudioFormat Class

::: coremusic.audio.AudioFormat
    options:
      members: true
      show_bases: true

### AudioUnit Class

::: coremusic.audio.AudioUnit
    options:
      members: true
      show_bases: true

### AudioQueue Class

::: coremusic.audio.AudioQueue
    options:
      members: true
      show_bases: true

### AudioConverter Class

::: coremusic.audio.AudioConverter
    options:
      members: true
      show_bases: true

### MIDIClient Class

::: coremusic.midi.MIDIClient
    options:
      members: true
      show_bases: true

### MIDIInputPort Class

::: coremusic.midi.MIDIInputPort
    options:
      members: true
      show_bases: true

### MIDIOutputPort Class

::: coremusic.midi.MIDIOutputPort
    options:
      members: true
      show_bases: true

### MIDIEndpoint Class

::: coremusic.midi.MIDIEndpoint
    options:
      members: true
      show_bases: true

### Endpoint Discovery

::: coremusic.midi.get_sources

::: coremusic.midi.get_destinations

::: coremusic.midi.find_source

::: coremusic.midi.find_destination

### AudioClock Class

::: coremusic.audio.AudioClock
    options:
      members: true
      show_bases: true

### ClockTimeFormat

::: coremusic.audio.ClockTimeFormat
    options:
      members: true
      show_bases: true

### AudioEffectsChain Class

::: coremusic.audio.AudioEffectsChain
    options:
      members: true
      show_bases: true

## Functional API

Low-level C-style functions are available through the `coremusic.capi` module
for advanced use cases requiring direct access to CoreAudio frameworks.

!!! note
    The object-oriented API is recommended for most use cases. The functional
    API in `coremusic.capi` provides low-level access when needed.

For direct access to low-level functions:

```python
--8<-- "examples/api/index/functional.py:example"
```

## Error Handling

coremusic provides exception classes for different CoreAudio subsystems:

::: coremusic.exceptions.CoreAudioError
    options:
      members: true
      show_bases: true

::: coremusic.exceptions.AudioFileError
    options:
      members: true
      show_bases: true

::: coremusic.exceptions.AudioUnitError
    options:
      members: true
      show_bases: true

::: coremusic.exceptions.AudioQueueError
    options:
      members: true
      show_bases: true

::: coremusic.exceptions.AudioConverterError
    options:
      members: true
      show_bases: true

::: coremusic.exceptions.MIDIError
    options:
      members: true
      show_bases: true

::: coremusic.exceptions.MusicPlayerError
    options:
      members: true
      show_bases: true

::: coremusic.exceptions.AudioDeviceError
    options:
      members: true
      show_bases: true

::: coremusic.exceptions.AUGraphError
    options:
      members: true
      show_bases: true

## Utility Functions

Utility functions are available through `coremusic.capi` for FourCC conversion
and other low-level operations:

```python
--8<-- "examples/api/index/fourcc.py:example"
```
