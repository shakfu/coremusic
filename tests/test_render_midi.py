#!/usr/bin/env python3
"""Tests for offline MIDI-to-audio instrument rendering.

The instrument render path was silently broken: instruments output
non-interleaved float32, but the render used a single interleaved buffer and
returned paramErr, producing silent files. These tests assert the render now
produces audible (non-zero) audio.
"""

import pytest

from coremusic.audio import AudioFile
from coremusic.audio.audiounit_host import AudioUnitPlugin, render_midi_file
from coremusic.midi.utilities import MIDISequence

# Apple's DLSMusicDevice ships with macOS; skip if it cannot be found.
try:
    _HAS_DLS = AudioUnitPlugin.from_name("DLSMusicDevice", component_type="aumu") is not None
except Exception:
    _HAS_DLS = False

pytestmark = pytest.mark.skipif(
    not _HAS_DLS, reason="DLSMusicDevice instrument not available"
)


def _make_midi(path):
    seq = MIDISequence()
    track = seq.add_track("melody")
    for i, note in enumerate([60, 64, 67, 72]):
        track.add_note(i * 0.3, note, 100, 0.25, 0)
    seq.save(str(path))
    return seq


def _nonzero_count(path):
    import numpy as np

    with AudioFile(str(path)) as af:
        data = af.read_as_numpy().astype("float64")
    return int(np.count_nonzero(data))


class TestRenderMidiFile:
    def test_produces_audible_output(self, tmp_path):
        pytest.importorskip("numpy")
        midi = tmp_path / "in.mid"
        _make_midi(midi)
        out = tmp_path / "out.wav"
        result = render_midi_file("DLSMusicDevice", str(midi), str(out))
        assert result == str(out)
        assert out.exists()
        # The core regression: output must not be silent.
        assert _nonzero_count(out) > 1000

    def test_output_format_and_duration(self, tmp_path):
        midi = tmp_path / "in.mid"
        seq = _make_midi(midi)
        out = tmp_path / "out.wav"
        render_midi_file(
            "DLSMusicDevice", str(midi), str(out), sample_rate=44100.0, channels=2
        )
        with AudioFile(str(out)) as af:
            assert af.format.sample_rate == 44100
            assert af.format.channels_per_frame == 2
            # Rendered length is MIDI duration plus the default 1s tail.
            assert af.duration == pytest.approx(seq.duration + 1.0, abs=0.1)

    def test_unknown_plugin_raises(self, tmp_path):
        midi = tmp_path / "in.mid"
        _make_midi(midi)
        with pytest.raises(ValueError):
            render_midi_file("NoSuchInstrument12345", str(midi), str(tmp_path / "x.wav"))


class TestPluginRenderMidi:
    def test_render_midi_returns_float32_bytes(self):
        instrument = AudioUnitPlugin.from_name(
            "DLSMusicDevice", component_type="aumu"
        )
        with instrument:
            events = [(0.0, 0x90, 60, 100), (0.5, 0x80, 60, 0)]
            audio = instrument.render_midi(
                events, duration=1.0, sample_rate=44100.0, channels=2
            )
        # 1 second, stereo, float32 == 44100 * 2 * 4 bytes.
        assert len(audio) == 44100 * 2 * 4

    def test_render_midi_rejects_effect_plugin(self):
        # AUDelay is an effect (aufx), not an instrument.
        try:
            effect = AudioUnitPlugin.from_name("AUDelay", component_type="aufx")
        except ValueError:
            pytest.skip("AUDelay effect not available")
        with effect:
            with pytest.raises(ValueError):
                effect.render_midi([(0.0, 0x90, 60, 100)], duration=0.1)


class TestShortcutRenderMidi:
    def test_shortcut(self, tmp_path):
        pytest.importorskip("numpy")
        from coremusic.shortcuts import render_midi

        midi = tmp_path / "in.mid"
        _make_midi(midi)
        out = tmp_path / "out.wav"
        assert render_midi("DLSMusicDevice", str(midi), str(out)) == str(out)
        assert _nonzero_count(out) > 1000
