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
import coremusic.capi as capi

# Low-level audio file operations
file_id = capi.audio_file_open_url("audio.wav")
# ... operations ...
capi.audio_file_close(file_id)

# Low-level clock operations
clock_id = capi.ca_clock_new()
capi.ca_clock_start(clock_id)
# ... operations ...
capi.ca_clock_dispose(clock_id)
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
import coremusic.capi as capi

# Convert FourCC string to integer
format_int = capi.fourchar_to_int('lpcm')

# Convert integer back to FourCC string
format_str = capi.int_to_fourchar(format_int)
```
