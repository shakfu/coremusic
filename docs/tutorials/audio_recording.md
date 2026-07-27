# Audio Recording

This tutorial covers recording audio from input devices using coremusic.

## Prerequisites

- coremusic installed and built
- A working audio input device (built-in microphone, USB audio interface, etc.)
- Basic Python knowledge

## Simple Recording

### Using the CLI

The easiest way to record is via the command line:

```bash
# Record for 10 seconds
coremusic audio record -o recording.wav --duration 10

# Record with specific settings
coremusic audio record -o recording.wav --duration 10 --sample-rate 48000 --channels 1

# List input devices
coremusic device list --input
```

### Using AudioRecorder

For programmatic recording:

```python
--8<-- "examples/tutorials/audio_recording/basic.py:example"
```

## Recording with Progress

Display recording progress:

```python
--8<-- "examples/tutorials/audio_recording/progress.py:example"
```

## Device Selection

### List Input Devices

```python
--8<-- "examples/tutorials/audio_recording/devices.py:list"
```

### Record from Specific Device

```python
--8<-- "examples/tutorials/audio_recording/devices.py:select"
```

## Recording Formats

### Different Sample Rates

```python
--8<-- "examples/tutorials/audio_recording/formats.py:high-quality"
```

### Mono Recording

```python
--8<-- "examples/tutorials/audio_recording/formats.py:mono"
```

## Real-Time Monitoring

`AudioRecorder` captures into a buffer; it does not report levels while it
runs. To meter the input live, read it with `AudioInputStream`, which hands
each block to a callback as it arrives:

```python
--8<-- "examples/tutorials/audio_recording/level_meter.py:example"
```

## Recording to NumPy Array

For processing, record directly to a NumPy array:

```python
--8<-- "examples/tutorials/audio_recording/numpy_capture.py:example"
```

## Error Handling

Handle recording errors gracefully:

```python
--8<-- "examples/tutorials/audio_recording/error_handling.py:example"
```

## Complete Example: Voice Recorder

A complete voice recorder application:

```python
--8<-- "examples/tutorials/audio_recording/voice_recorder.py:example"
```

## Troubleshooting

### No Input Device Found

1. Check System Preferences > Sound > Input
2. Ensure microphone permissions are granted
3. List devices with `coremusic device list --input`

### Recording is Silent

1. Check input device is selected correctly
2. Verify microphone is not muted
3. Test with Audio MIDI Setup app
4. Check input gain/volume

### Permission Denied

macOS requires microphone permission:

1. Go to System Preferences > Security & Privacy > Privacy
2. Select Microphone
3. Enable permission for Terminal or your Python app

## Next Steps

- [Audio Playback](audio_playback.md) - Play back your recordings
- [Audio Processing Cookbook](../cookbook/audio_processing.md) - Process recorded audio
- [Real-Time Audio Cookbook](../cookbook/real_time_audio.md) - Real-time audio monitoring

## See Also

- [API Reference](../api/index.md) - Complete API reference
- [CLI Guide](../guides/cli.md) - CLI recording commands
