#!/usr/bin/env python3
"""CoreMusic utilities package.

This package contains various utility modules for CoreMusic:
- scipy: SciPy-based audio signal processing utilities (optional)
- fourcc: FourCC (four-character code) conversion utilities
"""

# Import utilities module for backward compatibility
from . import scipy
from . import fourcc

__all__ = [
    "scipy",
    "fourcc",
]
