# Real-Time Audio Recipes

Practical recipes for real-time audio processing, built on
`coremusic.audio.streaming`: `AudioInputStream` pulls blocks from the input
device, `AudioOutputStream` pushes blocks to the output device, and
`AudioProcessor` wires the two together through a function of yours.

Every callback below runs on the audio thread. Allocate nothing, block on
nothing, and return promptly - anything slower becomes an audible dropout,
counted in `overruns` and `underruns`.

## Record Audio from Input Device

Capture from the default input device, one block at a time.

```python
--8<-- "examples/cookbook/real_time_audio/record.py:example"
```

## Play Audio with Low Latency

Generate samples on demand. The buffer size sets the latency.

```python
--8<-- "examples/cookbook/real_time_audio/low_latency_output.py:example"
```

## Monitor Audio Levels

Track peak and RMS of the incoming signal.

```python
--8<-- "examples/cookbook/real_time_audio/level_monitor.py:example"
```

## Process Input to Output

Read the input, process it, and play the result.

```python
--8<-- "examples/cookbook/real_time_audio/processor.py:example"
```

## See Also

- [Audio Processing](audio_processing.md) - Audio processing recipes
- [File Operations](file_operations.md) - File I/O recipes
