#!/usr/bin/env python3
"""A command-line file processor: input, output, and a list of effects."""

import sys

sys.argv = [sys.argv[0], "input.wav", "cli_processed.wav", "AUDelay", "AUMatrixReverb"]


# --8<-- [start:example]
import sys
from pathlib import Path

import numpy as np

import coremusic.capi as capi
from coremusic.audio import AudioFile, AudioFormat, ExtendedAudioFile
from coremusic.audio.audiounit_host import AudioUnitChain


class AudioProcessor:
    """Process audio files through a chain of effects."""

    BLOCK_FRAMES = 512

    def __init__(self):
        self.chain = None

    def setup_chain(self, effects):
        """Set up effects chain."""
        self.chain = AudioUnitChain()
        for effect in effects:
            self.chain.add_plugin(effect)

    def process_file(self, input_path, output_path):
        """Process audio file."""
        if not self.chain:
            raise RuntimeError("Chain not set up")

        print(f"Processing: {input_path}")
        print(f"Output: {output_path}")

        with AudioFile(input_path) as audio:
            samples = audio.read_as_numpy().astype(np.float32) / 32768.0
            channels = audio.format.channels_per_frame
            sample_rate = audio.format.sample_rate
            print(f"Duration: {audio.duration:.2f}s")

        blocks = []
        for start in range(0, len(samples), self.BLOCK_FRAMES):
            block = samples[start:start + self.BLOCK_FRAMES]
            if len(block) < self.BLOCK_FRAMES:
                block = np.pad(block, ((0, self.BLOCK_FRAMES - len(block)), (0, 0)))
            out = self.chain.process(block.tobytes(), num_frames=self.BLOCK_FRAMES)
            blocks.append(np.frombuffer(out, dtype=np.float32))

        result = np.concatenate(blocks)

        out_format = AudioFormat.pcm(
            sample_rate, channels=channels, bits=32, is_float=True
        )
        with ExtendedAudioFile.create(
            output_path, capi.fourchar_to_int('WAVE'), out_format
        ) as output:
            output.write(len(result) // channels, result.tobytes())

        print("Processing complete!")

    def cleanup(self):
        """Clean up resources."""
        if self.chain:
            self.chain.dispose()
            self.chain = None


def main():
    if len(sys.argv) < 3:
        print("Usage: python audio_processor.py <input.wav> <output.wav> [effects...]")
        print("Example: python audio_processor.py in.wav out.wav AUDelay AUMatrixReverb")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    effects = sys.argv[3:] or ["AUMatrixReverb"]

    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    processor = AudioProcessor()

    try:
        print(f"Setting up effects: {', '.join(effects)}")
        processor.setup_chain(effects)
        processor.process_file(input_file, output_file)
    finally:
        processor.cleanup()


if __name__ == "__main__":
    main()
# --8<-- [end:example]
