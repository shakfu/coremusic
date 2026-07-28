#!/usr/bin/env python3
"""Beat-Accurate MIDI Sequencing."""

# --8<-- [start:example]
import time

from coremusic import capi, link
from coremusic.midi import link as link_midi

# Create MIDI output
client = capi.midi_client_create("Link Sequencer")
port = capi.midi_output_port_create(client, "Seq Out")
dest = capi.midi_destination_create(client, "Link Demo Destination")

# Create Link session
with link.LinkSession(bpm=120.0) as session:
    # Create MIDI sequencer
    sequencer = link_midi.LinkMIDISequencer(session, port, dest)

    # Schedule notes at specific beats
    # Beat 0: C (60)
    sequencer.schedule_note(beat=0.0, channel=0, note=60, velocity=100, duration=0.9)

    # Beat 1: E (64)
    sequencer.schedule_note(beat=1.0, channel=0, note=64, velocity=100, duration=0.9)

    # Beat 2: G (67)
    sequencer.schedule_note(beat=2.0, channel=0, note=67, velocity=100, duration=0.9)

    # Beat 3: C (72)
    sequencer.schedule_note(beat=3.0, channel=0, note=72, velocity=100, duration=0.9)

    # Schedule CC automation
    sequencer.schedule_cc(beat=0.0, channel=0, controller=7, value=100)  # Volume
    sequencer.schedule_cc(beat=2.0, channel=0, controller=7, value=80)

    # Start sequencer
    sequencer.start()
    print("Sequencer started")

    # Let it play
    time.sleep(5)

    # Stop sequencer
    sequencer.stop()

# Cleanup
# Cleanup: disposing the client also disposes its ports and endpoints
capi.midi_client_dispose(client)
# --8<-- [end:example]
