#!/usr/bin/env python3
"""Tests for audio conversion (library helpers, shortcuts, and CLI command).

These exercise the previously-untested convert path, which had shipped broken:
sample-rate/bit-depth conversion produced an invalid ASBD, and unsupported
output extensions were silently written as WAV bytes under the wrong name.
"""

import argparse
import wave

import pytest

from coremusic.audio import AudioFile
from coremusic.audio.utilities import convert_audio_file, resolve_output_file_type


def _make_wav(path, *, sample_rate=44100, channels=2, seconds=1.0, freq=440.0):
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a dev dependency
        pytest.skip("numpy required")
    n = int(sample_rate * seconds)
    mono = (0.4 * np.sin(2 * np.pi * freq * np.arange(n) / sample_rate) * 32767).astype(
        "<i2"
    )
    frames = np.repeat(mono, channels) if channels > 1 else mono
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(frames.tobytes())
    return path


# ============================================================================
# resolve_output_file_type
# ============================================================================


class TestResolveOutputFileType:
    @pytest.mark.parametrize("ext", [".wav", ".aif", ".aiff", ".caf"])
    def test_supported_extensions(self, ext):
        assert isinstance(resolve_output_file_type(f"out{ext}"), int)

    @pytest.mark.parametrize("ext", [".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac"])
    def test_unsupported_extensions_raise(self, ext):
        with pytest.raises(ValueError):
            resolve_output_file_type(f"out{ext}")

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError):
            resolve_output_file_type("out.xyz")


# ============================================================================
# convert_audio_file (library, PCM containers)
# ============================================================================


class TestConvertAudioFile:
    def test_copy_same_format(self, tmp_path):
        src = _make_wav(tmp_path / "a.wav")
        out = tmp_path / "b.wav"
        # Build an output format matching the source so the fast copy path runs.
        with AudioFile(str(src)) as f:
            fmt = f.format
        convert_audio_file(str(src), str(out), fmt)
        with AudioFile(str(out)) as f:
            assert f.format.channels_per_frame == 2
            assert f.duration == pytest.approx(1.0, abs=0.05)

    def test_reject_unsupported_before_writing(self, tmp_path):
        src = _make_wav(tmp_path / "a.wav")
        with AudioFile(str(src)) as f:
            fmt = f.format
        out = tmp_path / "a.mp3"
        with pytest.raises(ValueError):
            convert_audio_file(str(src), str(out), fmt)
        # Nothing should have been written under the wrong name.
        assert not out.exists()


# ============================================================================
# CLI cmd_convert
# ============================================================================


def _convert_args(inp, out, **kw):
    ns = argparse.Namespace(
        input=str(inp),
        output=str(out),
        output_format=kw.get("output_format"),
        sample_rate=kw.get("sample_rate"),
        channels=kw.get("channels"),
        bit_depth=kw.get("bit_depth"),
        json=False,
    )
    return ns


class TestCmdConvert:
    def _run(self, args):
        from coremusic.cli.convert import cmd_convert

        return cmd_convert(args)

    def test_copy(self, tmp_path):
        src = _make_wav(tmp_path / "a.wav")
        out = tmp_path / "b.wav"
        assert self._run(_convert_args(src, out)) == 0
        with AudioFile(str(out)) as f:
            assert f.format.sample_rate == 44100

    def test_resample(self, tmp_path):
        src = _make_wav(tmp_path / "a.wav")
        out = tmp_path / "c.wav"
        assert self._run(_convert_args(src, out, sample_rate=48000)) == 0
        with AudioFile(str(out)) as f:
            assert f.format.sample_rate == 48000
            assert f.duration == pytest.approx(1.0, abs=0.05)

    def test_to_mono(self, tmp_path):
        src = _make_wav(tmp_path / "a.wav", channels=2)
        out = tmp_path / "m.wav"
        assert self._run(_convert_args(src, out, channels=1)) == 0
        with AudioFile(str(out)) as f:
            assert f.format.channels_per_frame == 1

    def test_bit_depth(self, tmp_path):
        src = _make_wav(tmp_path / "a.wav")
        out = tmp_path / "d.wav"
        assert self._run(_convert_args(src, out, bit_depth=24)) == 0
        with AudioFile(str(out)) as f:
            assert f.format.bits_per_channel == 24

    def test_aiff_container(self, tmp_path):
        src = _make_wav(tmp_path / "a.wav")
        out = tmp_path / "e.aiff"
        assert self._run(_convert_args(src, out)) == 0
        with AudioFile(str(out)) as f:
            assert f.duration == pytest.approx(1.0, abs=0.05)

    def test_caf_container(self, tmp_path):
        src = _make_wav(tmp_path / "a.wav")
        out = tmp_path / "f.caf"
        assert self._run(_convert_args(src, out)) == 0
        with AudioFile(str(out)) as f:
            assert f.duration == pytest.approx(1.0, abs=0.05)

    @pytest.mark.parametrize("ext", [".mp3", ".flac", ".m4a"])
    def test_unsupported_rejected(self, tmp_path, ext):
        from coremusic.cli._utils import CLIError

        src = _make_wav(tmp_path / "a.wav")
        out = tmp_path / f"x{ext}"
        with pytest.raises(CLIError):
            self._run(_convert_args(src, out))
        assert not out.exists()
