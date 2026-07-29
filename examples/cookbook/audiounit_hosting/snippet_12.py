#!/usr/bin/env python3
"""Dynamic Chain Manipulation."""

audio_chunk1 = audio_chunk2 = audio_chunk3 = bytes(512 * 2 * 4)

# --8<-- [start:example]
from coremusic.audio.audiounit_host import AudioUnitChain

chain = AudioUnitChain()

# Add initial plugins
chain.add_plugin("AUHipass")
chain.add_plugin("AUMatrixReverb")

# Process some audio
output1 = chain.process(audio_chunk1)

# Insert plugin in the middle
chain.insert_plugin(1, "AUDelay")
chain.configure_plugin(1, {"Delay Time": 0.3})

# Process more audio with new chain
output2 = chain.process(audio_chunk2)

# Remove plugin
chain.remove_plugin(1)

# Process final audio
output3 = chain.process(audio_chunk3)

# Cleanup
chain.dispose()
# --8<-- [end:example]
