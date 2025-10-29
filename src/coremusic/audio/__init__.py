#!/usr/bin/env python3
"""CoreMusic audio package.

This package contains audio-related modules:
- async_io: Asynchronous I/O classes for audio operations
- streaming: Real-time audio streaming and processing
"""

# Import async I/O classes
from .async_io import *

# Import streaming classes
from .streaming import *

__all__ = [
    # Async I/O
    "AsyncAudioFile",
    "AsyncAudioQueue",
    # Streaming
    "AudioInputStream",
    "AudioOutputStream",
    "AudioProcessor",
    "StreamGraph",
    "StreamNode",
    "create_loopback",
]
