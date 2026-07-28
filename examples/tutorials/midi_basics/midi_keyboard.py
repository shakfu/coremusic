#!/usr/bin/env python3
"""Play MIDI notes from the computer keyboard."""

# --8<-- [start:example]
import sys
import termios
import time
import tty

from coremusic.midi import MIDIClient, get_destinations, note_off, note_on


class MIDIKeyboard:
    """Computer keyboard to MIDI converter."""

    # Map computer keys to MIDI notes
    KEY_MAP = {
        'a': 60,  # C4
        'w': 61,  # C#4
        's': 62,  # D4
        'e': 63,  # D#4
        'd': 64,  # E4
        'f': 65,  # F4
        't': 66,  # F#4
        'g': 67,  # G4
        'y': 68,  # G#4
        'h': 69,  # A4
        'u': 70,  # A#4
        'j': 71,  # B4
        'k': 72,  # C5
    }

    # A terminal reports key presses, not key releases, so each note is held
    # for a fixed time rather than until the key comes back up.
    NOTE_DURATION = 0.3

    def __init__(self):
        self.client = MIDIClient("MIDI Keyboard")
        self.port = self.client.create_output_port("Output")

        # Send to the first available destination, or publish our own so the
        # keyboard is usable with no hardware attached.
        destinations = get_destinations()
        if destinations:
            self.destination = destinations[0]
        else:
            self.destination = self.client.create_virtual_destination(
                "MIDI Keyboard Out"
            )

    def play(self, note, velocity=100):
        """Play one note."""
        self.port.send_data(self.destination, note_on(note, velocity))
        print(f"Note On: {note}")
        time.sleep(self.NOTE_DURATION)
        self.port.send_data(self.destination, note_off(note))

    def run(self):
        """Run keyboard input loop."""
        print("MIDI Keyboard")
        print("=" * 40)
        print("Keys: A-S-D-F-G-H-J-K = C-D-E-F-G-A-B-C")
        print("Black keys: W-E-T-Y-U")
        print("Press 'q' to quit")
        print()

        # Set terminal to raw mode
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setraw(sys.stdin.fileno())

            while True:
                char = sys.stdin.read(1).lower()

                if char == 'q':
                    break

                if char in self.KEY_MAP:
                    self.play(self.KEY_MAP[char])

        except KeyboardInterrupt:
            pass

        finally:
            # Restore terminal
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self.client.dispose()
            print("\nGoodbye!")


if sys.stdin.isatty():
    MIDIKeyboard().run()
else:
    print("Not running on a terminal - nothing to read keys from.")
# --8<-- [end:example]
