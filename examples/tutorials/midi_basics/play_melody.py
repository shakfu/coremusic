#!/usr/bin/env python3
"""Play a melody one note at a time."""

# --8<-- [start:example]
import time

from coremusic.midi import MIDIClient, get_destinations


def play_melody(notes, durations, tempo_bpm=120):
    """Play a simple melody."""
    with MIDIClient("Melody Player") as client:
        port = client.create_output_port("Output")

        destinations = get_destinations()
        if not destinations:
            print("No MIDI destinations available")
            return

        destination = destinations[0]

        # Calculate beat duration
        beat_duration = 60.0 / tempo_bpm

        for note, duration in zip(notes, durations, strict=True):
            port.send_data(destination, bytes([0x90, note, 100]))
            time.sleep(duration * beat_duration)
            port.send_data(destination, bytes([0x80, note, 0]))


# The opening phrase of "Twinkle Twinkle Little Star"
notes = [60, 60, 67, 67, 69, 69, 67]  # C C G G A A G
durations = [1, 1, 1, 1, 1, 1, 2]

play_melody(notes, durations, tempo_bpm=180)
# --8<-- [end:example]
