#!/usr/bin/env python3
"""Build an AUGraph-backed effect chain feeding the default output."""

# --8<-- [start:example]
from coremusic.audio import AudioEffectsChain

chain = AudioEffectsChain()
chain.open()

reverb = chain.add_effect_by_name("AUReverb2")
output = chain.add_output()
chain.connect(reverb, output)

chain.initialize()
chain.start()
# ... process audio through chain
chain.stop()
chain.dispose()
# --8<-- [end:example]
