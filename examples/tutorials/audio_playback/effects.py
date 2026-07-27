#!/usr/bin/env python3
"""Run a file through a reverb and out to the speakers."""

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_output_devices():
    print("No audio output device available.")
    raise SystemExit(0)

# --8<-- [start:live-chain]
import time

from coremusic.audio import AudioEffectsChain


def run_reverb_chain(seconds):
    """Build reverb -> output and run it."""
    chain = AudioEffectsChain()
    chain.open()

    reverb_node = chain.add_effect_by_name("AUReverb2")
    output_node = chain.add_output()
    chain.connect(reverb_node, output_node)

    chain.initialize()
    try:
        chain.start()
        time.sleep(seconds)
    finally:
        chain.stop()
        chain.dispose()


run_reverb_chain(0.5)
# --8<-- [end:live-chain]

# --8<-- [start:offline]
from coremusic.audio import AudioFile
from coremusic.audio.audiounit_host import AudioUnitPlugin

# To hear a file through an effect rather than a live input, render it:
# read the samples, push them through the plugin, and play or write the result.
with AudioFile("audio.wav") as audio:
    samples = audio.read_as_numpy()

with AudioUnitPlugin.from_name("AUMatrixReverb") as reverb:
    reverb['Dry/Wet Mix'] = 40.0
    block = bytes(512 * 2 * 4)  # one block of float32 stereo silence
    processed = reverb.process(block)
    print(f"processed {len(processed)} bytes")
# --8<-- [end:offline]
