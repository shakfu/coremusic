#!/usr/bin/env python3
"""Advanced Chain with Wet/Dry Mix."""

input_audio = bytes(1024 * 2 * 4)  # 1024 stereo frames of float32

# --8<-- [start:example]
from coremusic.audio.audiounit_host import AudioUnitChain

with AudioUnitChain() as chain:
    chain.add_plugin("AUDelay")
    chain.add_plugin("AUMatrixReverb")

    chain.configure_plugin(0, {'Delay Time': 0.25})
    chain.configure_plugin(1, {'Dry/Wet Mix': 40.0})

    # Mix settings:
    # 0.0 = 100% dry (original signal)
    # 0.5 = 50% wet, 50% dry
    # 1.0 = 100% wet (fully processed)
    output = chain.process(input_audio, num_frames=1024, wet_dry_mix=0.7)
# --8<-- [end:example]
