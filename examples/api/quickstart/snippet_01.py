#!/usr/bin/env python3
"""Import Patterns."""

# --8<-- [start:example]
# Main package - object-oriented API (recommended)

# Low-level functional API

# Constants (preferred over capi getter functions)
from coremusic.constants import AudioFileProperty, AudioFormatID

# Optional integrations
import coremusic.utils.scipy as spu  # SciPy integration (requires scipy)
# --8<-- [end:example]
