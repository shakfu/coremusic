#!/usr/bin/env python3
"""Feed the output device from a pull generator."""

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_output_devices():
    print("No audio output device available.")
    raise SystemExit(0)

# --8<-- [start:example]
import time

from coremusic.audio import AudioFile
from coremusic.audio.streaming import AudioOutputStream


def make_file_generator(path, channels):
    """Return generator(frame_count) -> bytes, reading a file on demand."""
    audio = AudioFile(path)
    audio.open()
    position = 0
    total = audio.packet_count
    scale = 1.0 / 32768.0

    def generate(frame_count):
        nonlocal position
        import struct

        data, count = audio.read_packets(position, min(frame_count, total - position))
        position += count
        if count == 0:
            return b""

        # The file is 16-bit; the stream wants interleaved float32
        samples = struct.unpack(f"<{len(data) // 2}h", data)
        return struct.pack(f"<{len(samples)}f", *[s * scale for s in samples])

    return generate, audio


generate, audio = make_file_generator("audio.wav", channels=2)

stream = AudioOutputStream(channels=2, sample_rate=44100.0, buffer_size=512)
stream.set_generator(generate)
stream.start()

print(f"Streaming, latency {stream.latency * 1000:.1f}ms")
time.sleep(1.0)

stream.stop()
audio.close()
# --8<-- [end:example]
