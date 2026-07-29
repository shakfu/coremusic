#!/usr/bin/env python3
"""Combined Audio + MIDI Synchronized."""

# --8<-- [start:example]
import time

from coremusic import capi, link
from coremusic.base import AudioPlayer
from coremusic.midi import link as link_midi

# Setup MIDI
client = capi.midi_client_create("Audio+MIDI")
port = capi.midi_output_port_create(client, "Out")
destination = capi.midi_destination_create(client, "Link Demo Destination")

# Share one Link session for both audio and MIDI
with link.LinkSession(bpm=120.0) as session:
    # Setup audio player
    player = AudioPlayer(link_session=session)
    player.load_file("drums.wav")
    player.setup_output()

    # Setup MIDI sequencer
    seq = link_midi.LinkMIDISequencer(session, port, destination)

    # Schedule bass notes every beat
    for beat in range(16):
        note = 36 if beat % 4 == 0 else 38  # Kick and snare pattern
        seq.schedule_note(
            beat=float(beat),
            channel=9,  # MIDI drum channel
            note=note,
            velocity=100,
            duration=0.9,
        )

    # Start both audio and MIDI
    print("Starting synchronized audio + MIDI playback...")

    player.play()
    player.start()
    seq.start()

    # Monitor both
    for _i in range(40):
        timing = player.get_link_timing(quantum=4.0)
        progress = player.get_progress()

        print(
            f"Beat: {timing['beat']:7.2f} | "
            f"Audio: {progress * 100:5.1f}% | "
            f"Tempo: {timing['tempo']:6.1f} BPM",
            end="\r",
        )

        time.sleep(0.5)

    # Stop both
    player.stop()
    seq.stop()

# Cleanup: disposing the client also disposes its ports and endpoints
capi.midi_client_dispose(client)
# --8<-- [end:example]
