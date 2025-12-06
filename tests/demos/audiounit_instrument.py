#!/usr/bin/env python3
"""AudioUnit Instrument Plugin Demo

Demonstrates MIDI control of AudioUnit instrument plugins.

Features:
- Discover and load instrument plugins
- Send MIDI notes and control messages
- Program changes and multi-channel support
- Interactive instrument player
"""

import sys
import time
import coremusic as cm


def demo_discover_instruments():
    """Demo 1: Discover Instrument Plugins"""
    print("\n" + "=" * 70)
    print(" Demo 1: Discover Instrument Plugins")
    print("=" * 70)

    host = cm.AudioUnitHost()

    # Get all instruments
    instruments = host.discover_plugins(type='instrument')
    print(f"\nFound {len(instruments)} instrument plugins:")

    # Group by manufacturer
    by_mfr = {}
    for inst in instruments:
        mfr = inst['manufacturer']
        if mfr not in by_mfr:
            by_mfr[mfr] = []
        by_mfr[mfr].append(inst['name'])

    for mfr, plugins in sorted(by_mfr.items())[:10]:
        print(f"\n  {mfr}:")
        for plugin in plugins[:3]:
            print(f"    • {plugin}")
        if len(plugins) > 3:
            print(f"    ... and {len(plugins) - 3} more")

    print("\n✓ Demo complete")


def demo_basic_midi():
    """Demo 2: Basic MIDI Control"""
    print("\n" + "=" * 70)
    print(" Demo 2: Basic MIDI Control")
    print("=" * 70)

    host = cm.AudioUnitHost()

    print("\nLoading Apple DLSMusicDevice (General MIDI synth)...")
    with host.load_plugin("DLSMusicDevice", type='instrument') as synth:
        print(f"✓ Loaded: {synth.name}")

        # Play middle C
        print("\n  Playing middle C (note 60)...")
        synth.note_on(channel=0, note=60, velocity=100)
        time.sleep(0.5)
        synth.note_off(channel=0, note=60)

        # Play a C major chord
        print("  Playing C major chord (C-E-G)...")
        notes = [60, 64, 67]
        for note in notes:
            synth.note_on(channel=0, note=note, velocity=90)
        time.sleep(1.0)
        for note in notes:
            synth.note_off(channel=0, note=note)

        # Play ascending scale
        print("  Playing C major scale...")
        scale = [60, 62, 64, 65, 67, 69, 71, 72]
        for note in scale:
            synth.note_on(channel=0, note=note, velocity=80)
            time.sleep(0.2)
            synth.note_off(channel=0, note=note)

    print("\n✓ Demo complete")


def demo_program_changes():
    """Demo 3: Instrument Selection (Program Changes)"""
    print("\n" + "=" * 70)
    print(" Demo 3: Instrument Selection")
    print("=" * 70)

    host = cm.AudioUnitHost()

    # General MIDI instrument names
    gm_instruments = [
        (0, "Acoustic Grand Piano"),
        (24, "Nylon String Guitar"),
        (40, "Violin"),
        (48, "String Ensemble"),
        (56, "Trumpet"),
        (73, "Flute"),
    ]

    with host.load_plugin("DLSMusicDevice", type='instrument') as synth:
        print(f"\nLoaded: {synth.name}")

        for program, inst_name in gm_instruments:
            print(f"\n  Program {program:3d}: {inst_name}")
            synth.program_change(channel=0, program=program)

            # Play middle C with this instrument
            synth.note_on(channel=0, note=60, velocity=90)
            time.sleep(0.5)
            synth.note_off(channel=0, note=60)
            time.sleep(0.2)

    print("\n✓ Demo complete")


def demo_control_changes():
    """Demo 4: MIDI Controllers"""
    print("\n" + "=" * 70)
    print(" Demo 4: MIDI Controllers")
    print("=" * 70)

    host = cm.AudioUnitHost()

    with host.load_plugin("DLSMusicDevice", type='instrument') as synth:
        print(f"\nLoaded: {synth.name}")

        # Start a sustained note
        print("\n  Playing sustained note...")
        synth.note_on(channel=0, note=60, velocity=100)

        # Demonstrate volume control
        print("  Volume fade out...")
        for vol in range(127, 0, -10):
            synth.control_change(channel=0, controller=7, value=vol)
            time.sleep(0.1)

        print("  Volume fade in...")
        for vol in range(0, 127, 10):
            synth.control_change(channel=0, controller=7, value=vol)
            time.sleep(0.1)

        synth.note_off(channel=0, note=60)

        # Demonstrate pan control
        print("\n  Pan sweep (left to right)...")
        synth.note_on(channel=0, note=60, velocity=100)

        for pan in range(0, 127, 8):
            synth.control_change(channel=0, controller=10, value=pan)
            time.sleep(0.05)

        synth.note_off(channel=0, note=60)

    print("\n✓ Demo complete")


def demo_pitch_bend():
    """Demo 5: Pitch Bend"""
    print("\n" + "=" * 70)
    print(" Demo 5: Pitch Bend")
    print("=" * 70)

    host = cm.AudioUnitHost()

    with host.load_plugin("DLSMusicDevice", type='instrument') as synth:
        print(f"\nLoaded: {synth.name}")

        print("\n  Playing note with pitch bend...")

        # Start note
        synth.note_on(channel=0, note=60, velocity=100)
        time.sleep(0.3)

        # Bend up smoothly
        print("  Bending up...")
        for bend in range(8192, 16384, 512):
            synth.pitch_bend(channel=0, value=bend)
            time.sleep(0.02)

        time.sleep(0.2)

        # Bend back down
        print("  Bending down...")
        for bend in range(16384, 8192, -512):
            synth.pitch_bend(channel=0, value=bend)
            time.sleep(0.02)

        # Release note
        synth.note_off(channel=0, note=60)

        # Reset pitch bend to center
        synth.pitch_bend(channel=0, value=8192)

    print("\n✓ Demo complete")


def demo_multi_channel():
    """Demo 6: Multi-Channel Performance"""
    print("\n" + "=" * 70)
    print(" Demo 6: Multi-Channel Performance")
    print("=" * 70)

    host = cm.AudioUnitHost()

    with host.load_plugin("DLSMusicDevice", type='instrument') as synth:
        print(f"\nLoaded: {synth.name}")

        # Setup different instruments on different channels
        channels = [
            (0, 0, "Piano"),
            (1, 48, "Strings"),
            (2, 56, "Trumpet"),
            (3, 33, "Bass"),
        ]

        print("\n  Setting up channels:")
        for channel, program, name in channels:
            synth.program_change(channel=channel, program=program)
            print(f"    Channel {channel}: {name}")

        # Play chord progression with different instruments
        print("\n  Playing chord progression...")

        # C major chord
        print("  C major...")
        synth.note_on(channel=0, note=60, velocity=90)  # Piano: C
        synth.note_on(channel=1, note=64, velocity=70)  # Strings: E
        synth.note_on(channel=2, note=67, velocity=80)  # Trumpet: G
        synth.note_on(channel=3, note=36, velocity=100) # Bass: C low
        time.sleep(1.0)

        # Release
        for channel in range(4):
            synth.all_notes_off(channel=channel)
        time.sleep(0.2)

        # F major chord
        print("  F major...")
        synth.note_on(channel=0, note=65, velocity=90)  # Piano: F
        synth.note_on(channel=1, note=69, velocity=70)  # Strings: A
        synth.note_on(channel=2, note=72, velocity=80)  # Trumpet: C
        synth.note_on(channel=3, note=41, velocity=100) # Bass: F low
        time.sleep(1.0)

        # Release all
        for channel in range(4):
            synth.all_notes_off(channel=channel)

    print("\n✓ Demo complete")


def demo_arpeggiator():
    """Demo 7: Arpeggiator"""
    print("\n" + "=" * 70)
    print(" Demo 7: Arpeggiator")
    print("=" * 70)

    host = cm.AudioUnitHost()

    with host.load_plugin("DLSMusicDevice", type='instrument') as synth:
        print(f"\nLoaded: {synth.name}")

        # Set to bright synth sound
        synth.program_change(channel=0, program=81)  # Sawtooth synth

        print("\n  Playing arpeggiated chord...")

        # C major 7th arpeggio
        arpeggio = [60, 64, 67, 71]  # C, E, G, B

        # Play 4 octaves ascending and descending
        notes_sequence = arpeggio * 3
        notes_sequence += [72, 76, 79, 83]  # Higher octave
        notes_sequence += list(reversed(notes_sequence))

        for note in notes_sequence:
            synth.note_on(channel=0, note=note, velocity=90)
            time.sleep(0.08)
            synth.note_off(channel=0, note=note)

    print("\n✓ Demo complete")


def interactive_keyboard():
    """Demo 8: Interactive Keyboard"""
    print("\n" + "=" * 70)
    print(" Demo 8: Interactive Keyboard")
    print("=" * 70)

    host = cm.AudioUnitHost()

    # Keyboard mapping: computer keys to MIDI notes
    key_map = {
        'a': 60,  # C
        'w': 61,  # C#
        's': 62,  # D
        'e': 63,  # D#
        'd': 64,  # E
        'f': 65,  # F
        't': 66,  # F#
        'g': 67,  # G
        'y': 68,  # G#
        'h': 69,  # A
        'u': 70,  # A#
        'j': 71,  # B
        'k': 72,  # C (high)
    }

    print("\n  Keyboard mapping (press keys, 'q' to quit):")
    print("  a  w  s  e  d  f  t  g  y  h  u  j  k")
    print("  C  C# D  D# E  F  F# G  G# A  A# B  C")
    print("\n  [Currently non-interactive in demo mode]")

    with host.load_plugin("DLSMusicDevice", type='instrument') as synth:
        print(f"\nLoaded: {synth.name}")
        print("\n  Demo: Playing chromatic scale using key map...")

        for key, note in sorted(key_map.items(), key=lambda x: x[1]):
            print(f"  [{key}] ", end='', flush=True)
            synth.note_on(channel=0, note=note, velocity=90)
            time.sleep(0.15)
            synth.note_off(channel=0, note=note)

    print("\n\n✓ Demo complete")


def main():
    """Main demo runner"""
    print("\n" + "=" * 70)
    print(" CoreMusic AudioUnit Instrument Plugin Demo")
    print("=" * 70)
    print("\nDemonstrates MIDI control of AudioUnit instrument plugins:")
    print("  • Plugin discovery and loading")
    print("  • MIDI note on/off messages")
    print("  • Program changes (instrument selection)")
    print("  • Control changes (volume, pan, etc.)")
    print("  • Pitch bend")
    print("  • Multi-channel performance")
    print("  • Arpeggiator patterns")

    demos = [
        ("Discover Instruments", demo_discover_instruments),
        ("Basic MIDI Control", demo_basic_midi),
        ("Instrument Selection", demo_program_changes),
        ("MIDI Controllers", demo_control_changes),
        ("Pitch Bend", demo_pitch_bend),
        ("Multi-Channel Performance", demo_multi_channel),
        ("Arpeggiator", demo_arpeggiator),
        ("Interactive Keyboard", interactive_keyboard),
    ]

    while True:
        print("\n" + "=" * 70)
        print(" Demo Menu")
        print("=" * 70)
        for i, (name, _) in enumerate(demos, 1):
            print(f"  [{i}] {name}")
        print("  [a] Run all demos")
        print("  [q] Quit")

        choice = input("\nChoice: ").strip().lower()

        if choice == 'q':
            break
        elif choice == 'a':
            for name, demo_func in demos:
                demo_func()
                time.sleep(0.5)
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(demos):
                demos[idx][1]()
            else:
                print("Invalid choice")
        else:
            print("Invalid choice")

    print("\n" + "=" * 70)
    print(" Demo Complete!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  ✓ AudioUnit instruments support full MIDI control")
    print("  ✓ 16 independent MIDI channels")
    print("  ✓ 128 General MIDI program changes")
    print("  ✓ All standard MIDI messages supported")
    print("  ✓ Sample-accurate timing for tight sequences")
    print("\nNext steps:")
    print("  • Combine with Link for tempo-synced sequences")
    print("  • Route live MIDI input to AudioUnit instruments")
    print("  • Build a Python-based DAW or sequencer")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(0)
