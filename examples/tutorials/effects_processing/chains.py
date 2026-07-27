#!/usr/bin/env python3
"""Building effect chains, live and offline."""

# --8<-- [start:simple]
from coremusic.audio import AudioEffectsChain


def create_simple_chain():
    """Create a simple effect chain."""
    chain = AudioEffectsChain()
    chain.open()

    # Add effect by name
    delay_node = chain.add_effect_by_name("AUDelay")

    # Add output
    output_node = chain.add_output()

    # Connect effect to output
    chain.connect(delay_node, output_node)

    print(f"Created chain with {chain.node_count} nodes")
    return chain


chain = create_simple_chain()
chain.dispose()
# --8<-- [end:simple]

# --8<-- [start:multi]
from coremusic.audio import AudioEffectsChain


def create_multi_effect_chain():
    """Create chain with multiple effects."""
    chain = AudioEffectsChain()
    chain.open()

    # Add effects in series: EQ -> Compressor -> Reverb -> Output
    eq_node = chain.add_effect_by_name("AUGraphicEQ")
    comp_node = chain.add_effect_by_name("AUDynamicsProcessor")
    reverb_node = chain.add_effect_by_name("AUMatrixReverb")
    output_node = chain.add_output()

    # Connect: EQ -> Compressor -> Reverb -> Output
    chain.connect(eq_node, comp_node)
    chain.connect(comp_node, reverb_node)
    chain.connect(reverb_node, output_node)

    print("Created effects chain:")
    print("  Input -> EQ -> Compressor -> Reverb -> Output")

    return chain


chain = create_multi_effect_chain()
chain.dispose()
# --8<-- [end:multi]

# --8<-- [start:descriptors]
from coremusic.audio import create_simple_effect_chain


def create_chain_from_descriptors():
    """Create chain using explicit descriptors."""
    # Effect descriptors: (type, subtype, manufacturer)
    effects = [
        ("aufx", "dely", "appl"),  # Apple Delay
        ("aufx", "mrev", "appl"),  # Apple Matrix Reverb
    ]

    chain = create_simple_effect_chain(effects)

    print(f"Created chain with {chain.node_count} nodes")
    return chain


chain = create_chain_from_descriptors()
chain.dispose()
# --8<-- [end:descriptors]

# --8<-- [start:plugin-chain]
from coremusic.audio.audiounit_host import AudioUnitChain

# AudioEffectsChain builds an AUGraph, which routes live audio to the output
# device. To push blocks of your own through the same effects and get the
# result back, use AudioUnitChain.
block = bytes(512 * 2 * 4)  # 512 stereo frames of float32

with AudioUnitChain() as chain:
    chain.add_plugin("AUDelay")
    chain.add_plugin("AUMatrixReverb")
    processed = chain.process(block, wet_dry_mix=0.8)

print(f"{len(processed)} bytes out")
# --8<-- [end:plugin-chain]
