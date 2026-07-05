"""Shortcut functions for common coremusic operations.

This module provides simple, one-liner functions for common audio tasks.
These are thin wrappers around the object-oriented API designed for
quick scripting and interactive use.

Example:
    import coremusic as cm

    # Quick playback
    cm.play("song.wav")  # Blocking
    cm.play("song.wav", loop=True)  # Loop until interrupted

    # Quick analysis
    tempo = cm.analyze_tempo("song.wav")
    key, mode = cm.analyze_key("song.wav")

    # Quick conversion
    cm.convert("input.wav", "output.mp3")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .base import AudioPlayer

__all__ = [
    "play",
    "play_background",
    "play_async",
    "convert",
    "analyze_tempo",
    "analyze_key",
    "analyze_loudness",
    "get_duration",
    "get_info",
    "list_devices",
    "list_plugins",
]


def play(
    path: str | Path,
    *,
    loop: bool = False,
    volume: float = 1.0,
    block: bool = True,
) -> AudioPlayerHandle | None:
    """Play an audio file.

    Args:
        path: Path to audio file (WAV, AIFF, MP3, etc.)
        loop: If True, loop playback until stopped
        volume: Playback volume (0.0 to 1.0)
        block: If True, wait for playback to complete

    Returns:
        If block=False, returns an AudioPlayerHandle to control playback.
        If block=True, returns None after playback completes.

    Example:
        >>> import coremusic as cm
        >>> cm.play("song.wav")  # Play and wait
        >>> cm.play("song.wav", loop=True, block=False)  # Background loop
        <AudioPlayerHandle playing="song.wav">
    """
    from .base import AudioPlayer

    player = AudioPlayer()
    player.load_file(str(path))
    player.setup_output()

    if volume != 1.0:
        player.set_volume(volume)

    if loop:
        player.set_looping(True)

    player.play()

    if block:
        try:
            while player.is_playing():
                time.sleep(0.1)
        except KeyboardInterrupt:
            player.stop()
        return None
    else:
        return AudioPlayerHandle(player, str(path))


class AudioPlayerHandle:
    """Handle for controlling non-blocking audio playback.

    Returned by play() when block=False.
    """

    def __init__(self, player: "AudioPlayer", path: str):
        self._player = player
        self._path = path

    def stop(self) -> None:
        """Stop playback."""
        self._player.stop()

    def pause(self) -> None:
        """Pause playback."""
        self._player.pause()

    def resume(self) -> None:
        """Resume playback."""
        self._player.play()

    @property
    def is_playing(self) -> bool:
        """Check if still playing."""
        result: bool = self._player.is_playing()
        return result

    @property
    def position(self) -> float:
        """Current playback position in seconds."""
        result: float = self._player.get_time()
        return result

    def wait(self) -> None:
        """Block until playback completes."""
        try:
            while self._player.is_playing():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self._player.stop()

    def __repr__(self) -> str:
        status = "playing" if self.is_playing else "stopped"
        return f"<AudioPlayerHandle {status}={self._path!r}>"


def play_background(
    path: str | Path,
    *,
    loop: bool = False,
    volume: float = 1.0,
) -> AudioPlayerHandle:
    """Play an audio file in the background (non-blocking).

    Equivalent to play(path, block=False).

    Args:
        path: Path to audio file
        loop: If True, loop playback until stopped
        volume: Playback volume (0.0 to 1.0)

    Returns:
        AudioPlayerHandle to control playback

    Example:
        >>> handle = cm.play_background("song.wav")
        >>> # Do other work...
        >>> handle.stop()  # Stop when done
    """
    result = play(path, loop=loop, volume=volume, block=False)
    assert result is not None
    return result


def play_async(
    path: str | Path,
    *,
    loop: bool = False,
    volume: float = 1.0,
) -> AudioPlayerHandle:
    """Deprecated: use play_background() instead."""
    import warnings

    warnings.warn(
        "play_async() is deprecated, use play_background() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return play_background(path, loop=loop, volume=volume)


def convert(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sample_rate: float | None = None,
    channels: int | None = None,
    bit_depth: int | None = None,
) -> None:
    """Convert an audio file to another format.

    The output format is determined by the file extension.

    Args:
        input_path: Path to input audio file
        output_path: Path for output file (.wav, .aiff, .m4a, etc.)
        sample_rate: Target sample rate (None = keep original)
        channels: Target channel count (None = keep original)
        bit_depth: Target bit depth (None = keep original)

    Supported output containers: .wav, .aiff, .caf (lossless PCM). Compressed
    formats such as .mp3, .m4a, and .flac raise a clear error (macOS cannot
    encode them through this path).

    Example:
        >>> cm.convert("input.wav", "output.aiff")
        >>> cm.convert("stereo.wav", "mono.wav", channels=1)
        >>> cm.convert("hires.wav", "cd.wav", sample_rate=44100, bit_depth=16)
    """
    from .audio.core import AudioFormat
    from .audio.utilities import convert_audio_file

    # Read the source format as the base for the output format. Building it
    # unconditionally (not only when kwargs are given) avoids passing None to
    # convert_audio_file, which expects a concrete AudioFormat.
    from .audio import AudioFile

    with AudioFile(str(input_path)) as f:
        src = f.format

    out_channels = channels or src.channels_per_frame
    out_bits = bit_depth or src.bits_per_channel
    # Compute the PCM frame size; a zero bytes_per_frame yields an invalid ASBD
    # that AudioConverterNew rejects, which previously broke sample-rate and
    # bit-depth conversion.
    bytes_per_frame = (out_bits // 8) * out_channels

    output_format = AudioFormat(
        sample_rate=sample_rate or src.sample_rate,
        format_id=src.format_id,
        format_flags=src.format_flags,
        bytes_per_packet=bytes_per_frame,
        frames_per_packet=1,
        bytes_per_frame=bytes_per_frame,
        channels_per_frame=out_channels,
        bits_per_channel=out_bits,
    )

    convert_audio_file(str(input_path), str(output_path), output_format)


def analyze_tempo(path: str | Path) -> float:
    """Detect the tempo (BPM) of an audio file.

    Args:
        path: Path to audio file

    Returns:
        Tempo in beats per minute (BPM)

    Example:
        >>> tempo = cm.analyze_tempo("song.wav")
        >>> print(f"Tempo: {tempo:.1f} BPM")
        Tempo: 120.0 BPM
    """
    from .audio.analysis import AudioAnalyzer

    analyzer = AudioAnalyzer(str(path))
    beat_info = analyzer.detect_beats()
    return beat_info.tempo


def analyze_key(path: str | Path) -> tuple[str, str]:
    """Detect the musical key of an audio file.

    Args:
        path: Path to audio file

    Returns:
        Tuple of (key, mode) e.g. ("C", "major") or ("A", "minor")

    Example:
        >>> key, mode = cm.analyze_key("song.wav")
        >>> print(f"Key: {key} {mode}")
        Key: G major
    """
    from .audio.analysis import AudioAnalyzer

    analyzer = AudioAnalyzer(str(path))
    return analyzer.detect_key()


def analyze_loudness(path: str | Path) -> float:
    """Measure the integrated loudness of an audio file in LUFS.

    Args:
        path: Path to audio file

    Returns:
        Integrated loudness in LUFS (Loudness Units Full Scale)

    Example:
        >>> loudness = cm.analyze_loudness("song.wav")
        >>> print(f"Loudness: {loudness:.1f} LUFS")
        Loudness: -14.0 LUFS
    """
    from .audio.analysis import AudioAnalyzer

    analyzer = AudioAnalyzer(str(path))
    return analyzer.calculate_loudness()


def get_duration(path: str | Path) -> float:
    """Get the duration of an audio file in seconds.

    Args:
        path: Path to audio file

    Returns:
        Duration in seconds

    Example:
        >>> duration = cm.get_duration("song.wav")
        >>> print(f"Duration: {duration:.2f}s")
        Duration: 180.50s
    """
    from .audio import AudioFile

    with AudioFile(str(path)) as f:
        return f.duration


def get_info(path: str | Path) -> dict[str, Any]:
    """Get information about an audio file.

    Args:
        path: Path to audio file

    Returns:
        Dictionary with file information:
        - duration: Duration in seconds
        - sample_rate: Sample rate in Hz
        - channels: Number of channels
        - bit_depth: Bits per sample
        - format: Format ID (e.g., 'lpcm', 'aac')

    Example:
        >>> info = cm.get_info("song.wav")
        >>> print(info)
        {'duration': 180.5, 'sample_rate': 44100.0, 'channels': 2, ...}
    """
    from .audio import AudioFile

    with AudioFile(str(path)) as f:
        fmt = f.format
        return {
            "duration": f.duration,
            "sample_rate": fmt.sample_rate,
            "channels": fmt.channels_per_frame,
            "bit_depth": fmt.bits_per_channel,
            "format": fmt.format_id,
            "path": str(path),
        }


def list_devices(
    *, input_only: bool = False, output_only: bool = False
) -> list[dict[str, Any]]:
    """List available audio devices.

    Args:
        input_only: If True, only list input devices
        output_only: If True, only list output devices

    Returns:
        List of device info dictionaries

    Example:
        >>> devices = cm.list_devices()
        >>> for d in devices:
        ...     print(d['name'])
    """
    from .audio import AudioDeviceManager

    manager = AudioDeviceManager()

    if input_only:
        devices = manager.get_input_devices()
    elif output_only:
        devices = manager.get_output_devices()
    else:
        devices = manager.get_all_devices()

    result = []
    for d in devices:
        try:
            result.append(
                {
                    "name": d.name,
                    "uid": d.uid,
                    "manufacturer": d.manufacturer,
                }
            )
        except Exception as e:
            logger.debug("Failed to query device properties: %s", e)
    return result


def list_plugins(*, type: str | None = None) -> list[dict[str, Any]]:
    """List available AudioUnit plugins.

    Args:
        type: Filter by type ('effect', 'instrument', 'generator', etc.)
              None returns all plugins.

    Returns:
        List of plugin info dictionaries

    Example:
        >>> effects = cm.list_plugins(type='effect')
        >>> for p in effects:
        ...     print(p['name'])
    """
    from .audio.utilities import list_available_audio_units

    plugins = list_available_audio_units()

    if type:
        type_map = {
            "effect": "aufx",
            "instrument": "aumu",
            "generator": "augn",
            "mixer": "aumx",
            "panner": "aupn",
            "music_effect": "aumf",
        }
        filter_type = type_map.get(type.lower(), type)
        plugins = [p for p in plugins if p.get("type") == filter_type]

    return plugins


def render_midi(
    plugin: str,
    midi_path: str | Path,
    output_path: str | Path,
    *,
    sample_rate: float = 44100.0,
    channels: int = 2,
    tail_seconds: float = 1.0,
    preset: str | None = None,
) -> str:
    """Render a MIDI file through an instrument plugin to an audio file.

    Args:
        plugin: Instrument plugin name (case-insensitive, partial match)
        midi_path: Path to the input MIDI file
        output_path: Path for the rendered .wav output
        sample_rate: Render sample rate in Hz
        channels: Output channel count
        tail_seconds: Extra seconds rendered after the last event (note tails)
        preset: Optional factory preset name to load before rendering

    Returns:
        The output path as a string

    Example:
        >>> cm.render_midi("DLSMusicDevice", "song.mid", "song.wav")
    """
    from .audio.audiounit_host import render_midi_file

    return render_midi_file(
        plugin,
        midi_path,
        output_path,
        sample_rate=sample_rate,
        channels=channels,
        tail_seconds=tail_seconds,
        preset=preset,
    )
