#!/usr/bin/env python3
"""The constant enums."""

# --8<-- [start:example]
from coremusic.constants import (
    AudioFileProperty,
    AudioFormatID,
    AudioUnitProperty,
    AudioUnitScope,
)

# Audio file properties
prop_id = AudioFileProperty.DATA_FORMAT
prop_id = AudioFileProperty.ESTIMATED_DURATION

# Audio format IDs
fmt_id = AudioFormatID.LINEAR_PCM
fmt_id = AudioFormatID.MPEG4_AAC

# AudioUnit properties
au_prop = AudioUnitProperty.STREAM_FORMAT
au_prop = AudioUnitProperty.SAMPLE_RATE

# AudioUnit scopes
scope = AudioUnitScope.INPUT
scope = AudioUnitScope.OUTPUT
# --8<-- [end:example]
