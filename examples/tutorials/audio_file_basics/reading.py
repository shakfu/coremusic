#!/usr/bin/env python3
"""Reading audio data: a slice, in chunks, or all of it."""

# --8<-- [start:packets]
from coremusic.audio import AudioFile

with AudioFile("audio.wav") as audio:
    # Read first 1000 packets
    data, packets_read = audio.read_packets(0, 1000)

    print(f"Read {packets_read} packets")
    print(f"Data size: {len(data)} bytes")
# --8<-- [end:packets]


def process_chunk(chunk):
    pass


# --8<-- [start:chunks]
from coremusic.audio import AudioFile


def read_file_in_chunks(filepath, chunk_size=4096):
    """Read audio file in chunks."""
    with AudioFile(filepath) as audio:
        total_packets = audio.packet_count
        current_packet = 0

        while current_packet < total_packets:
            # Calculate remaining packets
            remaining = total_packets - current_packet
            to_read = min(chunk_size, remaining)

            # Read chunk
            data, count = audio.read_packets(current_packet, to_read)
            if count == 0:
                break

            # Process chunk
            yield data

            current_packet += count


# Use the generator
for chunk in read_file_in_chunks("audio.wav"):
    process_chunk(chunk)
# --8<-- [end:chunks]

# --8<-- [start:whole-file]
from coremusic.audio import AudioFile


def load_audio_file(filepath):
    """Load entire audio file into memory."""
    with AudioFile(filepath) as audio:
        # Read all packets
        data, count = audio.read_packets(0, audio.packet_count)

        return {
            "data": data,
            "sample_rate": audio.format.sample_rate,
            "channels": audio.format.channels_per_frame,
            "bits_per_channel": audio.format.bits_per_channel,
            "duration": audio.duration,
        }


# Load and use
audio_data = load_audio_file("audio.wav")
print(f"Loaded {len(audio_data['data'])} bytes")
# --8<-- [end:whole-file]
