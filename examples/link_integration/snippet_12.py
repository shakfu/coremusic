#!/usr/bin/env python3
"""Beat-Accurate MIDI Sequencing."""

# --8<-- [start:example]
import time

from coremusic import capi, link
from coremusic.midi import link as link_midi

# Setup MIDI
client = capi.midi_client_create("Sequencer")
port = capi.midi_output_port_create(client, "Seq Out")
destination = capi.midi_destination_create(client, "Link Demo Destination")

with link.LinkSession(bpm=120.0) as session:
    # Create sequencer
    seq = link_midi.LinkMIDISequencer(session, port, destination)

    # Schedule a C major arpeggio (one note per beat)
    seq.schedule_note(beat=0.0, channel=0, note=60, velocity=100, duration=0.9)  # C4
    seq.schedule_note(beat=1.0, channel=0, note=64, velocity=100, duration=0.9)  # E4
    seq.schedule_note(beat=2.0, channel=0, note=67, velocity=100, duration=0.9)  # G4
    seq.schedule_note(beat=3.0, channel=0, note=72, velocity=100, duration=0.9)  # C5

    print(f"Scheduled {len(seq.events)} MIDI events")

    # Start sequencer
    seq.start()
    print("Sequencer running...")

    # Monitor playback
    for i in range(20):
        state = session.capture_app_session_state()
        current_time = session.clock.micros()
        beat = state.beat_at_time(current_time, 4.0)

        # Show which beat we're on
        beat_num = int(beat) % 4
        indicators = ["●" if i == beat_num else "○" for i in range(4)]
        print(f"{' '.join(indicators)}  Beat: {beat:7.2f}", end='\r')

        time.sleep(0.5)

    # Stop sequencer
    seq.stop()

# Cleanup: disposing the client also disposes its ports and endpoints
capi.midi_client_dispose(client)
# --8<-- [end:example]
