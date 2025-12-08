"""User-friendly name to internal code mappings.

These mappings translate between human-readable CLI arguments
and internal CoreAudio FourCC codes / constants.
"""

from __future__ import annotations

from typing import Dict

# Audio format mappings (user input -> format_id)
FORMAT_NAMES: Dict[str, str] = {
    # Uncompressed
    "pcm": "lpcm",
    "linear-pcm": "lpcm",
    "wav": "lpcm",
    "aiff": "lpcm",
    # Compressed lossy
    "aac": "aac ",
    "mp3": ".mp3",
    # Compressed lossless
    "alac": "alac",
    "apple-lossless": "alac",
    "flac": "flac",
}

# Reverse mapping for display
FORMAT_DISPLAY: Dict[str, str] = {
    "lpcm": "Linear PCM",
    "aac ": "AAC",
    ".mp3": "MP3",
    "alac": "Apple Lossless",
    "flac": "FLAC",
    "aiff": "AIFF",
    "caff": "CAF",
}

# AudioUnit plugin type mappings
PLUGIN_TYPES: Dict[str, str] = {
    "effect": "aufx",
    "effects": "aufx",
    "fx": "aufx",
    "instrument": "aumu",
    "instruments": "aumu",
    "synth": "aumu",
    "generator": "augn",
    "generators": "augn",
    "output": "auou",
    "midi-processor": "aumf",
    "midi": "aumf",
}

# Reverse mapping for display
PLUGIN_TYPE_DISPLAY: Dict[str, str] = {
    "aufx": "Effect",
    "aumu": "Instrument",
    "augn": "Generator",
    "auou": "Output",
    "aumf": "MIDI Processor",
}

# Channel configuration display
CHANNEL_NAMES: Dict[int, str] = {
    1: "Mono",
    2: "Stereo",
    4: "Quadraphonic",
    6: "5.1 Surround",
    8: "7.1 Surround",
}


def get_format_id(user_input: str) -> str:
    """Convert user-friendly format name to format_id."""
    normalized = user_input.lower().strip()
    return FORMAT_NAMES.get(normalized, normalized)


def get_format_display(format_id: str) -> str:
    """Get display name for format_id."""
    return FORMAT_DISPLAY.get(format_id, format_id)


def get_plugin_type(user_input: str) -> str:
    """Convert user-friendly plugin type to FourCC."""
    normalized = user_input.lower().strip()
    return PLUGIN_TYPES.get(normalized, normalized)


def get_plugin_type_display(type_code: str) -> str:
    """Get display name for plugin type code."""
    return PLUGIN_TYPE_DISPLAY.get(type_code, type_code)


def get_channel_display(channels: int) -> str:
    """Get display name for channel count."""
    return CHANNEL_NAMES.get(channels, f"{channels}-channel")
