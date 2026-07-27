#!/usr/bin/env python3
"""Complete Example: Reverb Effect."""

# --8<-- [start:example]
import coremusic.capi as capi
from coremusic.audio import AudioFile, AudioFormat, ExtendedAudioFile
from coremusic.audio.audiounit_host import AudioUnitPlugin, PluginAudioFormat

# Load audio file
with AudioFile("input.wav") as audio_file:
    # Read audio data
    audio_data, frame_count = audio_file.read_packets(0, audio_file.packet_count)

    # Get audio format
    sample_rate = audio_file.format.sample_rate
    channels = audio_file.format.channels_per_frame

# Create plugin format
fmt = PluginAudioFormat(
    sample_rate=sample_rate,
    channels=channels,
    sample_format=PluginAudioFormat.FLOAT32,
    interleaved=True
)

# Process with reverb
with AudioUnitPlugin.from_name("AUMatrixReverb") as reverb:
    reverb.set_audio_format(fmt)

    # Configure reverb
    # Reverbs differ in what they expose; a factory preset is portable
    for preset in reverb.factory_presets:
        if preset.name == 'Large Room':
            reverb.load_factory_preset(preset)
            break

    # Process audio
    output_data = reverb.process(audio_data, num_frames=frame_count, audio_format=fmt)

# Save processed audio
out_format = AudioFormat.pcm(sample_rate, channels=channels, bits=32, is_float=True)
with ExtendedAudioFile.create(
    "output.wav", capi.fourchar_to_int("WAVE"), out_format
) as output_file:
    output_file.write(frame_count, output_data)

print("Processing complete!")
# --8<-- [end:example]
