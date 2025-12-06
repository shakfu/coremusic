#!/usr/bin/env python3
"""Create an audio effects chain using AUGraph.

Usage:
    python create_chain.py
"""

import coremusic as cm


def main():
    with cm.AudioEffectsChain() as chain:
        # Add 3D mixer effect
        mixer_node = chain.add_effect("aumi", "3dem", "appl")
        print(f"Added 3D Mixer: node {mixer_node}")

        # Add output
        output_node = chain.add_output()
        print(f"Added Output: node {output_node}")

        # Connect
        chain.connect(mixer_node, output_node)
        print(f"Connected: {mixer_node} -> {output_node}")

        # Initialize
        try:
            chain.open().initialize()
            print(f"Chain initialized: {chain.node_count} nodes")
        except Exception as e:
            print(f"Init failed: {e}")


if __name__ == "__main__":
    main()
