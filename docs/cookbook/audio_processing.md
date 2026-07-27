# Audio Processing Recipes

Practical recipes for common audio processing tasks.

## Normalize Audio Volume

Normalize audio to target peak level.

```python
--8<-- "examples/cookbook/audio_processing/normalize_audio.py:example"
```

## Apply Fade In/Out

Add smooth fade effects to audio.

```python
--8<-- "examples/cookbook/audio_processing/apply_fades.py:example"
```

## Change Sample Rate

Resample audio to different sample rate using ExtendedAudioFile.

```python
--8<-- "examples/cookbook/audio_processing/resample_audio.py:example"
```

`convert_audio_file` wraps the converter. To drive it yourself - to process
while you resample, say - set a client format on the input and copy:

```python
--8<-- "examples/cookbook/audio_processing/resample_audio.py:manual"
```

## Mix Multiple Tracks

Mix multiple audio tracks into stereo output.

```python
--8<-- "examples/cookbook/audio_processing/mix_tracks.py:example"
```

## Split Audio into Chunks

Split long audio file into smaller segments.

```python
--8<-- "examples/cookbook/audio_processing/split_audio.py:fixed"
```

To cut on musical boundaries rather than a fixed grid, let `AudioSlicer`
detect onsets:

```python
--8<-- "examples/cookbook/audio_processing/split_audio.py:onsets"
```

## Merge Audio Files

Concatenate multiple audio files into one.

```python
--8<-- "examples/cookbook/audio_processing/merge_audio_files.py:example"
```

## See Also

- [File Operations](file_operations.md) - File I/O recipes
- [Performance Guide](../guides/performance.md) - Performance optimization
- [AudioFile API](../api/audio_file.md) - AudioFile API reference
