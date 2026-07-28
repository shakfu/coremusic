#!/usr/bin/env python3
"""The mido operations, done with coremusic."""

# --8<-- [start:ports]
from coremusic.midi import MIDIClient, get_destinations, note_on

# List ports
for destination in get_destinations():
    print(destination.name)

# Send MIDI
client = MIDIClient("MyApp")
port = client.create_output_port("Output")

destinations = get_destinations()
dest = destinations[0] if destinations else client.create_virtual_destination("Out")

# Send note on
message = note_on("C4", 100)  # Channel 0, note 60, velocity 100
port.send_data(dest, message)

client.dispose()
# --8<-- [end:ports]

# --8<-- [start:files]
from coremusic.midi import MIDISequence

# Load a MIDI file
sequence = MIDISequence.load("song.mid")
for track in sequence.tracks:
    print(f"{track.name}: {len(track.events)} events")

# Create a new one
sequence = MIDISequence()
track = sequence.add_track("Melody")
track.add_note(time=0.0, note=60, velocity=100, duration=1.0)
sequence.save("output.mid")
# --8<-- [end:files]

# --8<-- [start:coreaudio-sequence]
from coremusic.midi import MusicSequence

# The CoreAudio sequencer, for playback through MusicPlayer
sequence = MusicSequence()
sequence.load_from_file("song.mid")

for i in range(sequence.track_count):
    track = sequence.get_track(i)
    # Access track data

# Or build one
sequence = MusicSequence()
track = sequence.new_track()
track.add_midi_note(time=0.0, channel=0, note=60, velocity=100, duration=1.0)
# --8<-- [end:coreaudio-sequence]
