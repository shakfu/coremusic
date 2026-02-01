"""Base classes and utilities for coremusic objects.

This module provides shared infrastructure used across all object modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import capi

# Check if NumPy is available
try:
    import numpy as np
    from numpy.typing import NDArray

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore
    if TYPE_CHECKING:
        from numpy.typing import NDArray

# Re-export base class from capi
CoreAudioObject = capi.CoreAudioObject
AudioPlayer = capi.AudioPlayer

__all__ = [
    "CoreAudioObject",
    "AudioPlayer",
    "NUMPY_AVAILABLE",
    "np",
]
