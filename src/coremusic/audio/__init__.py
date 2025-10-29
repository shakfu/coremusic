#!/usr/bin/env python3
"""CoreMusic audio package.

This package contains audio-related modules:
- async_io: Asynchronous I/O classes for audio operations
"""

# Import async I/O classes
from .async_io import *

__all__ = [
    "AsyncAudioFile",
    "AsyncAudioQueue",
]
