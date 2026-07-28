#!/usr/bin/env python3
"""Porting CoreAudio C code: the object layer and the functional layer."""

# --8<-- [start:audiofile-oo]
from coremusic.audio import AudioFile

# Open audio file
with AudioFile("audio.wav") as audio:
    # Get format
    audio_format = audio.format

    # Read packets
    data, count = audio.read_packets(0, 1024)

# Automatic cleanup via context manager
# --8<-- [end:audiofile-oo]

# --8<-- [start:audiofile-functional]
from coremusic import capi

# Open
file_id = capi.audio_file_open_url("audio.wav")

# Get format
format_data = capi.audio_file_get_property(
    file_id,
    capi.get_audio_file_property_data_format()
)

# Read
data, count = capi.audio_file_read_packets(file_id, 0, 1024)

# Close
capi.audio_file_close(file_id)
# --8<-- [end:audiofile-functional]

# --8<-- [start:audiounit]
from coremusic.audio import AudioUnit

# Find and create the default output unit
unit = AudioUnit.default_output()
unit.initialize()
unit.start()
unit.stop()
unit.dispose()
# --8<-- [end:audiounit]

# --8<-- [start:audiounit-functional]
from coremusic import capi

desc = {
    'type': capi.fourchar_to_int('auou'),
    'subtype': capi.fourchar_to_int('def '),
    'manufacturer': capi.fourchar_to_int('appl'),
}

comp = capi.audio_component_find_next(desc)
unit = capi.audio_component_instance_new(comp)
capi.audio_unit_initialize(unit)
capi.audio_output_unit_start(unit)
capi.audio_output_unit_stop(unit)
capi.audio_component_instance_dispose(unit)
# --8<-- [end:audiounit-functional]

# --8<-- [start:midi]
from coremusic import capi
from coremusic.midi import note_on

# Create MIDI client
client = capi.midi_client_create("MyClient")

# Create output port
output_port = capi.midi_output_port_create(client, "Output")

# Get a destination
dest = capi.midi_destination_create(client, "MyClient Destination")

# Send note
message = note_on("C4", 100)
capi.midi_send_data(output_port, dest, message)

capi.midi_client_dispose(client)
# --8<-- [end:midi]
