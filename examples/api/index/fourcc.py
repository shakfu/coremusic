#!/usr/bin/env python3
"""FourCC Conversion."""

# --8<-- [start:example]
from coremusic import capi

# Convert FourCC string to integer
format_int = capi.fourchar_to_int('lpcm')

# Convert integer back to FourCC string
format_str = capi.int_to_fourchar(format_int)
# --8<-- [end:example]
