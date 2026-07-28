#!/usr/bin/env python3
"""Every channel voice message, built and shown against its wire bytes."""

# --8<-- [start:notes]
from coremusic.midi import note_off, note_on

# A note may be a MIDI number, a name, or a Note
note_on(60, 100)  # b"\x90\x3c\x64"
note_on("C4", 100)  # the same message
note_on("F#3", 100)  # sharps and flats both parse

# Channel is keyword-only, so it can never be mistaken for a note
note_on("C4", 100, channel=2)  # b"\x92\x3c\x64"

note_off("C4")  # b"\x80\x3c\x00", release velocity 0
note_off("C4", 64)  # with an explicit release velocity

# Note numbers run 0-127. Middle C is 60 ("C4"), A440 is 69 ("A4").
# --8<-- [end:notes]

# --8<-- [start:control-change]
from coremusic.constants import MIDIControlChange
from coremusic.midi import all_notes_off, all_sound_off, control_change

control_change(MIDIControlChange.MODULATION, 64)  # mod wheel to 50%
control_change(MIDIControlChange.VOLUME, 100)
control_change(MIDIControlChange.PAN, 64)  # centred
control_change(MIDIControlChange.SUSTAIN_PEDAL, 127)  # sustain down
control_change(MIDIControlChange.SUSTAIN_PEDAL, 0)  # sustain up

# The two panic messages differ: All Notes Off releases held notes and lets
# their release tails ring, All Sound Off cuts the channel dead.
all_notes_off()  # CC 123
all_sound_off()  # CC 120
# --8<-- [end:control-change]

# --8<-- [start:program-change]
from coremusic.midi import program_change

program_change(0)  # b"\xc0\x00", program 0 (piano)
program_change(48)  # program 48 (strings)

# Program Change carries one data byte, so the message is two bytes long
assert len(program_change(0)) == 2
# --8<-- [end:program-change]

# --8<-- [start:pitch-bend]
from coremusic.midi import PITCH_BEND_CENTER, PITCH_BEND_MAX, pitch_bend

# Pitch bend is 14-bit, split across two 7-bit bytes, least significant first.
# pitch_bend() does the split for you.
pitch_bend(PITCH_BEND_CENTER)  # 8192, no bend
pitch_bend(0)  # fully down
pitch_bend(PITCH_BEND_MAX)  # 16383, fully up

# Reassembling the halves gives the original value back
message = pitch_bend(12000)
assert message[1] | (message[2] << 7) == 12000
# --8<-- [end:pitch-bend]

# --8<-- [start:aftertouch]
from coremusic.midi import channel_aftertouch, poly_aftertouch

poly_aftertouch("C4", 64)  # pressure on one held note
channel_aftertouch(64)  # pressure on every held note, two bytes
# --8<-- [end:aftertouch]

# --8<-- [start:validation]
from coremusic.midi import note_on as _note_on

# Out-of-range values raise rather than wrapping silently. A velocity of 200
# masked to 7 bits would become 72, and a data byte above 127 reads as a status
# byte, desynchronising everything after it.
try:
    _note_on(60, 200)
except ValueError as e:
    print(e)  # velocity must be 0-127, got 200
# --8<-- [end:validation]

print("all message reference examples built successfully")
