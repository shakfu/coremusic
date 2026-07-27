# Audio Playback

This tutorial covers audio playback using coremusic, from simple file playback to real-time streaming.

## Prerequisites

- coremusic installed and built
- Basic Python knowledge
- Audio files to play (WAV, AIFF, MP3, M4A, etc.)

## Simple File Playback

### Using AudioPlayer (Recommended)

The `AudioPlayer` class provides the easiest way to play audio files:

```python
--8<-- "examples/tutorials/audio_playback/basic.py:example"
```

### Playback with Progress

Monitor playback progress with a progress bar:

```python
--8<-- "examples/tutorials/audio_playback/progress.py:example"
```

### Looping Playback

For continuous looping:

```python
--8<-- "examples/tutorials/audio_playback/looping.py:manual"
```

Or let the player loop for you, and stop it when you are done:

```python
--8<-- "examples/tutorials/audio_playback/looping.py:builtin"
```

## Using the CLI

The coremusic CLI provides quick playback:

```bash
# Simple playback
coremusic audio play music.wav

# Looping playback
coremusic audio play music.wav --loop

# List audio devices
coremusic device list
```

## Streaming Playback

`AudioPlayer` reads the file for you. When you need to produce the samples
yourself - a synthesiser, a decoder, a live effect - use
`AudioOutputStream`, which pulls blocks from a generator you supply:

```python
--8<-- "examples/tutorials/audio_playback/streaming.py:example"
```

The generator returns interleaved float32 as raw bytes and must return
promptly: it runs on the audio thread, where a slow call becomes an audible
dropout. `stream.latency` reports the round-trip latency of the configured
buffer size.

## Async Playback

For non-blocking playback in async applications:

```python
--8<-- "examples/tutorials/audio_playback/async_playback.py:example"
```

## Playback with Effects

Route audio through AudioUnit effects during playback:

```python
--8<-- "examples/tutorials/audio_playback/effects.py:live-chain"
```

That chain processes live input. To hear a *file* through an effect, render it
through the plugin and play the result:

```python
--8<-- "examples/tutorials/audio_playback/effects.py:offline"
```

## Device Selection

Play to a specific audio device:

```python
--8<-- "examples/tutorials/audio_playback/devices.py:list"
```

CoreAudio routes playback to the default output device, so selecting a device
means making it the default for the duration:

```python
--8<-- "examples/tutorials/audio_playback/devices.py:select"
```

## Volume Control

Control playback volume:

```python
--8<-- "examples/tutorials/audio_playback/devices.py:volume"
```

## Error Handling

Handle playback errors gracefully:

```python
--8<-- "examples/tutorials/audio_playback/error_handling.py:example"
```

## Complete Example: Music Player

A simple command-line music player:

```python
--8<-- "examples/tutorials/audio_playback/music_player.py:example"
```

## Next Steps

- [Audio Recording](audio_recording.md) - Record audio from input devices
- [Audio Processing Cookbook](../cookbook/audio_processing.md) - Process and manipulate audio
- [AudioUnit Hosting Cookbook](../cookbook/audiounit_hosting.md) - Use AudioUnit effects

## See Also

- [API Reference](../api/index.md) - Complete API reference
- [Real-Time Audio Cookbook](../cookbook/real_time_audio.md) - Real-time audio techniques
