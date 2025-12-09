#!/usr/bin/env python3
"""Shared pytest fixtures for coremusic tests.

This module provides common fixtures for test data files, avoiding
duplication across test modules.

Constants can be imported directly:
    from conftest import AMEN_WAV_PATH, CANON_MID_PATH
"""

import os
import pytest

# Base paths for test data
TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, "data")
MIDI_DIR = os.path.join(DATA_DIR, "midi")
WAV_DIR = os.path.join(DATA_DIR, "wav")

# Common file paths - can be imported directly
AMEN_WAV_PATH = os.path.join(WAV_DIR, "amen.wav")
CANON_MID_PATH = os.path.join(MIDI_DIR, "canon.mid")
DEMO_MID_PATH = os.path.join(MIDI_DIR, "demo.mid")
GROOVE_DIR = os.path.join(MIDI_DIR, "groove")
CLASSICAL_DIR = os.path.join(MIDI_DIR, "classical")

# Legacy path aliases for backwards compatibility
AMEN_WAV = AMEN_WAV_PATH
CANON_MID = CANON_MID_PATH
DEMO_MID = DEMO_MID_PATH


# ============================================================================
# Audio File Fixtures
# ============================================================================


@pytest.fixture
def amen_wav_path():
    """Fixture providing path to amen.wav test file."""
    if not os.path.exists(AMEN_WAV_PATH):
        pytest.skip(f"Test audio file not found: {AMEN_WAV_PATH}")
    return AMEN_WAV_PATH


@pytest.fixture
def test_wav_path(amen_wav_path):
    """Alias for amen_wav_path for backwards compatibility."""
    return amen_wav_path


@pytest.fixture
def test_audio():
    """Generate test audio signal for signal processing tests.

    Returns a tuple of (audio_data, sample_rate) with a 440Hz sine wave.
    """
    try:
        import numpy as np
    except ImportError:
        pytest.skip("NumPy required for audio signal fixtures")

    sample_rate = 44100
    duration = 0.2
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t)
    return audio, sample_rate


@pytest.fixture
def test_signal_mono():
    """Generate mono test signal for filter tests.

    Returns a tuple of (signal, sample_rate) with 440Hz and 2000Hz components.
    """
    try:
        import numpy as np
    except ImportError:
        pytest.skip("NumPy required for audio signal fixtures")

    sample_rate = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t)
    return signal, sample_rate


@pytest.fixture
def test_signal_stereo():
    """Generate stereo test signal for filter tests.

    Returns a tuple of (signal, sample_rate) with stereo audio.
    """
    try:
        import numpy as np
    except ImportError:
        pytest.skip("NumPy required for audio signal fixtures")

    sample_rate = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration))
    left = np.sin(2 * np.pi * 440 * t)
    right = np.sin(2 * np.pi * 880 * t)
    signal = np.column_stack([left, right])
    return signal, sample_rate


@pytest.fixture
def test_signal():
    """Generate test signal with known frequency for spectral analysis.

    Returns a tuple of (signal, sample_rate, frequency).
    """
    try:
        import numpy as np
    except ImportError:
        pytest.skip("NumPy required for audio signal fixtures")

    sample_rate = 44100
    duration = 0.5
    freq = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * freq * t)
    return signal, sample_rate, freq


# ============================================================================
# MIDI File Fixtures
# ============================================================================


@pytest.fixture
def canon_mid_path():
    """Fixture providing path to canon.mid test file."""
    if not os.path.exists(CANON_MID_PATH):
        pytest.skip(f"Test MIDI file not found: {CANON_MID_PATH}")
    return CANON_MID_PATH


@pytest.fixture
def demo_mid_path():
    """Fixture providing path to demo.mid test file."""
    if not os.path.exists(DEMO_MID_PATH):
        pytest.skip(f"Test MIDI file not found: {DEMO_MID_PATH}")
    return DEMO_MID_PATH


@pytest.fixture
def groove_dir():
    """Fixture providing path to groove MIDI directory."""
    if not os.path.exists(GROOVE_DIR):
        pytest.skip(f"Test MIDI directory not found: {GROOVE_DIR}")
    return GROOVE_DIR


@pytest.fixture
def sample_midi_file(tmp_path):
    """Create a sample MIDI file for testing.

    Creates a simple C major scale repeated 4 times.
    """
    from coremusic.midi.utilities import MIDISequence

    midi_path = tmp_path / "test.mid"
    sequence = MIDISequence()
    track = sequence.add_track("Test")

    # C major scale repeated 4 times
    notes = [60, 62, 64, 65, 67, 69, 71, 72] * 4
    for i, note in enumerate(notes):
        track.add_note(i * 0.25, 0.25, note, 100)

    sequence.save(str(midi_path))
    return str(midi_path)


# ============================================================================
# MIDI Event Fixtures
# ============================================================================


@pytest.fixture
def sample_midi_events():
    """Create sample MIDI events for testing.

    Returns a simple ascending C major scale (C4 to C5).
    """
    from coremusic.midi.utilities import MIDIEvent, MIDIStatus

    events = []
    notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale

    for i, note in enumerate(notes):
        # Note on
        events.append(
            MIDIEvent(i * 0.25, MIDIStatus.NOTE_ON, 0, note, 100)
        )
        # Note off
        events.append(
            MIDIEvent((i + 1) * 0.25, MIDIStatus.NOTE_OFF, 0, note, 0)
        )

    return events


# ============================================================================
# Audio Format Fixtures
# ============================================================================


@pytest.fixture
def audio_format():
    """Fixture providing a standard audio format for testing.

    Returns an AudioFormat for 44.1kHz, stereo, 16-bit PCM.
    """
    import coremusic as cm

    return cm.AudioFormat(
        sample_rate=44100.0,
        format_id="lpcm",
        format_flags=12,  # kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked
        bytes_per_packet=4,
        frames_per_packet=1,
        bytes_per_frame=4,
        channels_per_frame=2,
        bits_per_channel=16,
    )


@pytest.fixture
def source_format():
    """Fixture providing source audio format (44.1kHz, stereo, 16-bit) as dict.

    Used by functional capi tests that expect dict format.
    """
    import coremusic.capi as capi

    return {
        "sample_rate": 44100.0,
        "format_id": capi.get_audio_format_linear_pcm(),
        "format_flags": 12,
        "bytes_per_packet": 4,
        "frames_per_packet": 1,
        "bytes_per_frame": 4,
        "channels_per_frame": 2,
        "bits_per_channel": 16,
        "reserved": 0,
    }


@pytest.fixture
def dest_format_mono():
    """Fixture providing destination format (44.1kHz, mono, 16-bit) as dict.

    Used by functional capi tests that expect dict format.
    """
    import coremusic.capi as capi

    return {
        "sample_rate": 44100.0,
        "format_id": capi.get_audio_format_linear_pcm(),
        "format_flags": 12,
        "bytes_per_packet": 2,
        "frames_per_packet": 1,
        "bytes_per_frame": 2,
        "channels_per_frame": 1,
        "bits_per_channel": 16,
        "reserved": 0,
    }


@pytest.fixture
def dest_format_48k():
    """Fixture providing destination format (48kHz, stereo, 16-bit) as dict.

    Used by functional capi tests that expect dict format.
    """
    import coremusic.capi as capi

    return {
        "sample_rate": 48000.0,
        "format_id": capi.get_audio_format_linear_pcm(),
        "format_flags": 12,
        "bytes_per_packet": 4,
        "frames_per_packet": 1,
        "bytes_per_frame": 4,
        "channels_per_frame": 2,
        "bits_per_channel": 16,
        "reserved": 0,
    }


@pytest.fixture
def pcm_format():
    """Fixture providing PCM audio format as AudioFormat object."""
    import coremusic as cm

    return cm.AudioFormat(
        sample_rate=44100.0,
        format_id="lpcm",
        format_flags=12,
        bytes_per_packet=4,
        frames_per_packet=1,
        bytes_per_frame=4,
        channels_per_frame=2,
        bits_per_channel=16,
    )


# ============================================================================
# AudioFormat Object Fixtures (for OO API tests)
# ============================================================================


@pytest.fixture
def source_format_obj():
    """Fixture providing source audio format as AudioFormat object.

    Used by OO API tests that expect AudioFormat objects.
    """
    import coremusic as cm

    return cm.AudioFormat(
        sample_rate=44100.0,
        format_id="lpcm",
        format_flags=12,
        bytes_per_packet=4,
        frames_per_packet=1,
        bytes_per_frame=4,
        channels_per_frame=2,
        bits_per_channel=16,
    )


@pytest.fixture
def dest_format_mono_obj():
    """Fixture providing destination format as AudioFormat object.

    Used by OO API tests that expect AudioFormat objects.
    """
    import coremusic as cm

    return cm.AudioFormat(
        sample_rate=44100.0,
        format_id="lpcm",
        format_flags=12,
        bytes_per_packet=2,
        frames_per_packet=1,
        bytes_per_frame=2,
        channels_per_frame=1,
        bits_per_channel=16,
    )


@pytest.fixture
def dest_format_48k_obj():
    """Fixture providing destination format as AudioFormat object.

    Used by OO API tests that expect AudioFormat objects.
    """
    import coremusic as cm

    return cm.AudioFormat(
        sample_rate=48000.0,
        format_id="lpcm",
        format_flags=12,
        bytes_per_packet=4,
        frames_per_packet=1,
        bytes_per_frame=4,
        channels_per_frame=2,
        bits_per_channel=16,
    )


@pytest.fixture
def pcm_format_dict():
    """Fixture providing PCM audio format as dictionary."""
    import coremusic.capi as capi

    return {
        "sample_rate": 44100.0,
        "format_id": capi.get_audio_format_linear_pcm(),
        "format_flags": 12,
        "bytes_per_packet": 4,
        "frames_per_packet": 1,
        "bytes_per_frame": 4,
        "channels_per_frame": 2,
        "bits_per_channel": 16,
        "reserved": 0,
    }


# ============================================================================
# Temporary File Fixtures
# ============================================================================


@pytest.fixture
def temp_audio_file(tmp_path):
    """Fixture providing temporary audio file path."""
    import tempfile

    path = tmp_path / "test_audio.wav"
    yield str(path)
    # Cleanup happens automatically with tmp_path
