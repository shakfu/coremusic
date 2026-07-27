#!/usr/bin/env python3
"""Importing from the audio subpackage."""

# --8<-- [start:objects]
from coremusic.audio import (
    AudioConverter,
    AudioFile,
    AudioFormat,
    AudioQueue,
    AudioUnit,
    ExtendedAudioFile,
)

audio = AudioFile("audio.wav")
queue = AudioQueue.new_output(AudioFormat.pcm(44100.0, channels=2))
unit = AudioUnit.default_output()
# --8<-- [end:objects]

audio.dispose()
queue.dispose()
unit.dispose()

# --8<-- [start:async]
from coremusic.audio import AsyncAudioFile, AsyncAudioQueue

# Or from the module that defines them
from coremusic.audio.async_io import AsyncAudioFile
# --8<-- [end:async]

# --8<-- [start:analysis]
from coremusic.audio.analysis import (
    AudioAnalyzer,
    BeatInfo,
    LivePitchDetector,
    PitchInfo,
)

analyzer = AudioAnalyzer("audio.wav")
beats = analyzer.detect_beats()
print(f"{beats.tempo:.1f} BPM")
# --8<-- [end:analysis]

# --8<-- [start:slicing]
from coremusic.audio.slicing import AudioSlicer

slicer = AudioSlicer("audio.wav", method="onset")
slices = slicer.detect_slices()
print(f"{len(slices)} slices")
# --8<-- [end:slicing]

# --8<-- [start:visualization]
from coremusic.audio.visualization import (
    FrequencySpectrumPlotter,
    SpectrogramPlotter,
    WaveformPlotter,
)

plotter = WaveformPlotter("audio.wav")
plotter.save("waveform.png")
# --8<-- [end:visualization]

# --8<-- [start:audiounit-host]
from coremusic.audio.audiounit_host import (
    AudioUnitChain,
    AudioUnitHost,
    AudioUnitParameter,
    AudioUnitPlugin,
    AudioUnitPreset,
    PresetManager,
)

host = AudioUnitHost()
plugin = host.load_plugin("AUMatrixReverb", type="effect")
plugin.dispose()
# --8<-- [end:audiounit-host]
