"""Base classes and utilities for coremusic objects.

This module provides shared infrastructure used across all object modules.
"""

from __future__ import annotations

from coremusic import capi

# Check if NumPy is available
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

# Re-export base class from capi
CoreAudioObject = capi.CoreAudioObject
AudioPlayer = capi.AudioPlayer

__all__ = [
    "NUMPY_AVAILABLE",
    "AudioPlayer",
    "CoreAudioObject",
    "np",
]
