#!/usr/bin/env python3
"""CoreMusic MIDI package.

This package contains MIDI-related modules:
- link: Ableton Link + MIDI integration (LinkMIDIClock, LinkMIDISequencer)
"""

# Import link module
from . import link

__all__ = [
    "link",
]
