#!/usr/bin/env python3
"""Basic Chain."""

input_audio = bytes(1024 * 2 * 4)  # 1024 stereo frames of float32

# --8<-- [start:example]
from coremusic.audio.audiounit_host import AudioUnitChain

# Create chain with context manager
with AudioUnitChain() as chain:
    # Add plugins
    chain.add_plugin("AUHipass")
    chain.add_plugin("AUDelay")
    chain.add_plugin("AUMatrixReverb")

    # Configure each plugin
    chain.configure_plugin(0, {"Cutoff Frequency": 200.0})
    chain.configure_plugin(1, {"Delay Time": 0.5, "Feedback": 30.0})
    chain.configure_plugin(2, {"Dry/Wet Mix": 40.0})

    # Process audio through entire chain
    output = chain.process(input_audio, num_frames=1024)
# --8<-- [end:example]
