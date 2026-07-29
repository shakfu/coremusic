#!/usr/bin/env python3
"""The patterns the cookbook recipes are built from."""

# --8<-- [start:read-file]
from coremusic.audio import AudioFile

with AudioFile("audio.wav") as audio:
    data, count = audio.read_packets(0, 1024)
    samples = audio.read_as_numpy()  # or the whole file at once
# --8<-- [end:read-file]

# A block of silence to push through the plugins below: 512 stereo frames of
# 32-bit float, which is the format the host uses unless told otherwise.
input_data = bytes(512 * 2 * 4)

# --8<-- [start:load-plugin]
from coremusic.audio.audiounit_host import AudioUnitPlugin

with AudioUnitPlugin.from_name("AUDelay") as plugin:
    plugin["Delay Time"] = 0.5
    output = plugin.process(input_data)
# --8<-- [end:load-plugin]

# --8<-- [start:plugin-chain]
from coremusic.audio.audiounit_host import AudioUnitChain

with AudioUnitChain() as chain:
    chain.add_plugin("AUHipass")
    chain.add_plugin("AUMatrixReverb")
    output = chain.process(input_data, wet_dry_mix=0.8)
# --8<-- [end:plugin-chain]

# --8<-- [start:link]
from coremusic import link

session = link.LinkSession(bpm=120.0)
session.enabled = True

state = session.capture_app_session_state()
print(f"Tempo: {state.tempo:.1f} BPM")
print(f"Peers: {session.num_peers}")

session.enabled = False
# --8<-- [end:link]

# --8<-- [start:send-midi]
from coremusic import capi
from coremusic.midi import note_on

client = capi.midi_client_create("Output")
port = capi.midi_output_port_create(client, "Out")
dest = capi.midi_destination_create(client, "Cookbook Destination")

# Send Note On
capi.midi_send_data(port, dest, note_on("C4", 100))

capi.midi_client_dispose(client)
# --8<-- [end:send-midi]
