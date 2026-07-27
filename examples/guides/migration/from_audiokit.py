#!/usr/bin/env python3
"""The AudioKit operations, done with coremusic."""

from coremusic.audio import AudioDeviceManager

have_output = bool(AudioDeviceManager.get_output_devices())

if have_output:
    # --8<-- [start:playback]
    from coremusic.base import AudioPlayer

    # High-level player
    player = AudioPlayer()
    player.load_file("audio.wav")
    player.setup_output()
    player.play()
    player.stop()
    # --8<-- [end:playback]

# --8<-- [start:queue]
from coremusic.audio import AudioFile, AudioQueue

# Or lower-level AudioQueue
with AudioFile("audio.wav") as audio:
    queue = AudioQueue.new_output(audio.format)

    # Allocate buffers and queue playback
    # (See the Real-Time Audio cookbook for a complete example)
    queue.dispose()
# --8<-- [end:queue]

# --8<-- [start:effects]
from coremusic.audio.audiounit_host import AudioUnitPlugin

input_data = bytes(512 * 2 * 4)  # 512 stereo frames of float32 silence

# Load a reverb AudioUnit
with AudioUnitPlugin.from_name("AUMatrixReverb") as reverb:
    reverb['Dry/Wet Mix'] = 50.0

    # Process audio
    output = reverb.process(input_data)
# --8<-- [end:effects]
