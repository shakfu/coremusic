#!/usr/bin/env python3
"""Reading file properties and format details."""

# --8<-- [start:basic]
from coremusic.audio import AudioFile

with AudioFile("audio.wav") as audio:
    # Basic information
    print(f"Duration: {audio.duration:.2f} seconds")
    print(f"Total packets: {audio.packet_count}")

    # Format details
    fmt = audio.format
    print(f"Sample rate: {fmt.sample_rate} Hz")
    print(f"Channels: {fmt.channels_per_frame}")
    print(f"Bit depth: {fmt.bits_per_channel}")
    print(f"Format ID: {fmt.format_id}")
# --8<-- [end:basic]

# --8<-- [start:format]
from coremusic.audio import AudioFile


def display_audio_format(fmt):
    """Display detailed format information."""
    print("Format Information:")
    print(f"  Sample Rate: {fmt.sample_rate} Hz")
    print(f"  Format ID: {fmt.format_id}")
    print(f"  Channels: {fmt.channels_per_frame}")
    print(f"  Bits/Channel: {fmt.bits_per_channel}")
    print(f"  Bytes/Frame: {fmt.bytes_per_frame}")
    print(f"  Bytes/Packet: {fmt.bytes_per_packet}")
    print(f"  Frames/Packet: {fmt.frames_per_packet}")
    print(f"  Format Flags: 0x{fmt.format_flags:08X}")


with AudioFile("audio.wav") as audio:
    display_audio_format(audio.format)
# --8<-- [end:format]
