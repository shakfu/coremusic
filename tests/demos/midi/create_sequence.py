#!/usr/bin/env python3
"""Create a simple MIDI sequence.

Usage:
    python create_sequence.py [output_file]
"""

import sys
from coremusic.midi import MIDISequence


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/melody.mid"

    seq = MIDISequence(tempo=120.0)
    track = seq.add_track("Melody")
    track.channel = 0

    # C major scale
    notes = [60, 62, 64, 65, 67, 69, 71, 72]
    for i, note in enumerate(notes):
        track.add_note(i * 0.5, note, 100, 0.4)

    seq.save(output_path)

    print(f"Created: {output_path}")
    print(f"  Tempo: {seq.tempo} BPM")
    print(f"  Duration: {seq.duration:.2f}s")
    print(f"  Events: {len(track.events)}")


if __name__ == "__main__":
    main()
