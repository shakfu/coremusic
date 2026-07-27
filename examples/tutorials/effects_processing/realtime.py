#!/usr/bin/env python3
"""Run a live effects chain into the output device."""

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_output_devices():
    print("No audio output device available.")
    raise SystemExit(0)

# --8<-- [start:example]
import time

from coremusic.audio import AudioEffectsChain


class RealTimeEffectsProcessor:
    """Process audio in real-time with effects."""

    def __init__(self):
        self.chain = None
        self.running = False

    def setup(self, effect_names):
        """Set up effects chain."""
        self.chain = AudioEffectsChain()
        self.chain.open()

        # Add effects
        prev_node = None
        for name in effect_names:
            node = self.chain.add_effect_by_name(name)
            if node is None:
                print(f"Warning: Effect not found: {name}")
                continue

            if prev_node is not None:
                self.chain.connect(prev_node, node)
            prev_node = node

        # Add output
        output_node = self.chain.add_output()
        if prev_node:
            self.chain.connect(prev_node, output_node)

        # Initialize
        self.chain.initialize()

        print(f"Effects chain ready with {self.chain.node_count} nodes")

    def start(self):
        """Start real-time processing."""
        if self.chain:
            self.chain.start()
            self.running = True
            print("Effects processing started")

    def stop(self):
        """Stop processing."""
        if self.chain:
            self.chain.stop()
            self.running = False
            print("Effects processing stopped")

    def cleanup(self):
        """Clean up resources."""
        if self.chain:
            self.chain.dispose()
            self.chain = None


# Use the processor
processor = RealTimeEffectsProcessor()
processor.setup(["AUDelay", "AUMatrixReverb"])
processor.start()

# Let it run for a while
time.sleep(1)

processor.stop()
processor.cleanup()
# --8<-- [end:example]
