#!/usr/bin/env python3
"""Import Patterns."""

# --8<-- [start:example]
# Main package - object-oriented API (recommended)

# Low-level functional API

# Constants (preferred over capi getter functions)
# Optional integrations
import coremusic.utils.scipy as spu  # SciPy integration (requires scipy)
from coremusic.constants import AudioFileProperty, AudioFormatID
# --8<-- [end:example]
