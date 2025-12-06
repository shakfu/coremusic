#!/usr/bin/env python3
"""Create a multi-track MIDI composition.

Usage:
    python multi_track.py [output_file]
"""

import sys
from coremusic.midi import MIDISequence


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/composition.mid"

    seq = MIDISequence(tempo=120.0, time_signature=(4, 4))

    # Melody
    melody = seq.add_track("Melody")
    melody.channel = 0
    melody.add_program_change(0.0, 0)  # Piano
    for i, note in enumerate([60, 64, 67, 64, 60, 64, 67, 72]):
        melody.add_note(i * 0.5, note, 100, 0.4)

    # Bass
    bass = seq.add_track("Bass")
    bass.channel = 1
    bass.add_program_change(0.0, 33)  # Electric Bass
    for i, note in enumerate([48, 48, 43, 43]):
        bass.add_note(i * 1.0, note, 90, 0.9)

    # Drums
    drums = seq.add_track("Drums")
    drums.channel = 9
    for beat in range(8):
        if beat % 2 == 0:
            drums.add_note(beat * 0.5, 36, 110, 0.1)  # Kick
        else:
            drums.add_note(beat * 0.5, 38, 90, 0.1)  # Snare

    seq.save(output_path)

    print(f"Created: {output_path}")
    print(f"  Tracks: {len(seq.tracks)}")
    print(f"  Duration: {seq.duration:.2f}s")


if __name__ == "__main__":
    main()
