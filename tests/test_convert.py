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

    @pytest.mark.parametrize("ext", [".mp3", ".ogg", ".opus"])
    def test_unsupported_extensions_raise(self, ext):
        with pytest.raises(ValueError):
            resolve_output_file_type(f"out{ext}")

    @pytest.mark.parametrize("ext", [".m4a", ".aac", ".flac"])
    def test_compressed_extensions_resolve(self, ext):
        # These now name writable compressed containers.
        assert isinstance(resolve_output_file_type(f"out{ext}"), int)

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
# convert_audio_file (compressed encoding via ExtAudioFile)
# ============================================================================


class TestCompressedEncoding:
    @staticmethod
    def _pcm(sample_rate, channels):
        from coremusic.audio import AudioFormat

        bpf = 2 * channels
        return AudioFormat(
            float(sample_rate), "lpcm", 12, bpf, 1, bpf, channels, 16
        )

    @pytest.mark.parametrize(
        "ext,expected_codec",
        [(".m4a", "aac "), (".aac", "aac "), (".flac", "flac")],
    )
    def test_default_codec_by_extension(self, tmp_path, ext, expected_codec):
        from coremusic.audio import ExtendedAudioFile

        src = _make_wav(tmp_path / "a.wav", seconds=1.0)
        out = tmp_path / f"o{ext}"
        convert_audio_file(str(src), str(out), self._pcm(44100, 2))
        assert out.exists() and out.stat().st_size > 0
        with ExtendedAudioFile(str(out)) as f:
            ff = f.file_format
            assert ff.format_id == expected_codec
            assert ff.channels_per_frame == 2

    def test_explicit_alac_into_m4a(self, tmp_path):
        from coremusic.audio import AudioFormat, ExtendedAudioFile

        src = _make_wav(tmp_path / "a.wav")
        out = tmp_path / "o.m4a"
        convert_audio_file(str(src), str(out), AudioFormat.alac(44100, 2))
        with ExtendedAudioFile(str(out)) as f:
            assert f.file_format.format_id == "alac"

    def test_mono_encode(self, tmp_path):
        from coremusic.audio import AudioFormat, ExtendedAudioFile

        src = _make_wav(tmp_path / "a.wav", channels=2)
        out = tmp_path / "mono.m4a"
        convert_audio_file(str(src), str(out), AudioFormat.aac(44100, 1))
        with ExtendedAudioFile(str(out)) as f:
            assert f.file_format.channels_per_frame == 1

    def test_resample_encode(self, tmp_path):
        from coremusic.audio import AudioFormat, ExtendedAudioFile

        src = _make_wav(tmp_path / "a.wav", sample_rate=44100)
        out = tmp_path / "hi.flac"
        convert_audio_file(str(src), str(out), AudioFormat.flac(48000, 2))
        with ExtendedAudioFile(str(out)) as f:
            assert f.file_format.sample_rate == 48000

    def test_duration_preserved(self, tmp_path):
        from coremusic.audio import ExtendedAudioFile

        src = _make_wav(tmp_path / "a.wav", seconds=1.0)
        out = tmp_path / "o.flac"
        convert_audio_file(str(src), str(out), self._pcm(44100, 2))
        with ExtendedAudioFile(str(out)) as f:
            frames = f.frame_count
            assert frames / f.file_format.sample_rate == pytest.approx(1.0, abs=0.05)

    def test_container_codec_mismatch_rejected(self, tmp_path):
        from coremusic.audio import AudioFormat

        src = _make_wav(tmp_path / "a.wav")
        out = tmp_path / "o.flac"
        # ALAC cannot live in a FLAC container.
        with pytest.raises(ValueError):
            convert_audio_file(str(src), str(out), AudioFormat.alac(44100, 2))

    def test_trim_to_compressed_rejected(self, tmp_path):
        from coremusic.audio.utilities import trim_audio

        src = _make_wav(tmp_path / "a.wav")
        out = tmp_path / "o.m4a"
        with pytest.raises(ValueError):
            trim_audio(str(src), str(out), 0.0, 0.5)
        assert not out.exists()

    def test_aac_bitrate_affects_size(self, tmp_path):
        from coremusic.audio import AudioFormat

        src = _make_noise_wav(tmp_path / "n.wav", seconds=2.0)
        lo = tmp_path / "lo.m4a"
        hi = tmp_path / "hi.m4a"
        convert_audio_file(str(src), str(lo), AudioFormat.aac(44100, 2), bitrate=64000)
        convert_audio_file(str(src), str(hi), AudioFormat.aac(44100, 2), bitrate=256000)
        assert lo.stat().st_size < hi.stat().st_size

    def test_bitrate_rejected_for_lossless(self, tmp_path):
        from coremusic.audio import AudioFormat

        src = _make_wav(tmp_path / "a.wav")
        with pytest.raises(ValueError):
            convert_audio_file(
                str(src), str(tmp_path / "o.flac"),
                AudioFormat.flac(44100, 2), bitrate=128000,
            )

    def test_bitrate_rejected_for_pcm(self, tmp_path):
        src = _make_wav(tmp_path / "a.wav")
        with pytest.raises(ValueError):
            convert_audio_file(
                str(src), str(tmp_path / "o.wav"), self._pcm(44100, 2), bitrate=128000
            )

    def test_set_encode_bitrate_validates_positive(self, tmp_path):
        from coremusic.audio import AudioFormat, ExtendedAudioFile
        from coremusic.constants import AudioFileType

        out = tmp_path / "o.m4a"
        f = ExtendedAudioFile.create(
            str(out), int(AudioFileType.M4A), AudioFormat.aac(44100, 2)
        )
        try:
            with pytest.raises(ValueError):
                f.set_encode_bitrate(0)
            with pytest.raises(ValueError):
                f.set_encode_bitrate(-1)
        finally:
            f.close()


# ============================================================================
# CLI cmd_batch (compressed output)
# ============================================================================


def _batch_args(indir, outdir, **kw):
    return argparse.Namespace(
        input_dir=str(indir),
        output_dir=str(outdir),
        pattern=kw.get("pattern", "*.wav"),
        output_format=kw.get("output_format", "wav"),
        sample_rate=kw.get("sample_rate"),
        channels=kw.get("channels"),
        bit_depth=kw.get("bit_depth"),
        recursive=kw.get("recursive", False),
        json=False,
    )


class TestCmdBatch:
    def _run(self, args):
        from coremusic.cli.convert import cmd_batch

        return cmd_batch(args)

    def test_batch_to_flac(self, tmp_path):
        indir = tmp_path / "in"
        indir.mkdir()
        _make_wav(indir / "one.wav")
        _make_wav(indir / "two.wav")
        outdir = tmp_path / "out"
        assert self._run(_batch_args(indir, outdir, output_format="flac")) == 0
        assert (outdir / "one.flac").exists()
        assert (outdir / "two.flac").exists()

    def test_batch_alac_is_not_aac(self, tmp_path):
        from coremusic.audio import ExtendedAudioFile

        indir = tmp_path / "in"
        indir.mkdir()
        _make_wav(indir / "one.wav")
        outdir = tmp_path / "out"
        assert self._run(_batch_args(indir, outdir, output_format="alac")) == 0
        with ExtendedAudioFile(str(outdir / "one.m4a")) as f:
            assert f.file_format.format_id == "alac"

    def test_batch_rejects_mp3(self, tmp_path):
        from coremusic.cli._utils import CLIError

        indir = tmp_path / "in"
        indir.mkdir()
        _make_wav(indir / "one.wav")
        outdir = tmp_path / "out"
        with pytest.raises(CLIError):
            self._run(_batch_args(indir, outdir, output_format="mp3"))


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
        bitrate=kw.get("bitrate"),
        json=False,
    )
    return ns


def _make_noise_wav(path, *, sample_rate=44100, channels=2, seconds=1.0):
    """A complex (noise) source so the AAC encoder actually spends the bitrate
    budget, making bitrate-driven size differences reliable."""
    import numpy as np

    rng = np.random.default_rng(1234)
    n = int(sample_rate * seconds)
    data = (rng.uniform(-0.5, 0.5, n * channels) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(data.tobytes())
    return path


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

    @pytest.mark.parametrize("ext", [".mp3", ".ogg", ".opus"])
    def test_unsupported_rejected(self, tmp_path, ext):
        from coremusic.cli._utils import CLIError

        src = _make_wav(tmp_path / "a.wav")
        out = tmp_path / f"x{ext}"
        with pytest.raises(CLIError):
            self._run(_convert_args(src, out))
        assert not out.exists()

    @pytest.mark.parametrize("ext", [".m4a", ".aac", ".flac"])
    def test_encode_compressed(self, tmp_path, ext):
        src = _make_wav(tmp_path / "a.wav")
        out = tmp_path / f"x{ext}"
        assert self._run(_convert_args(src, out)) == 0
        assert out.exists() and out.stat().st_size > 0

    def test_encode_alac_override(self, tmp_path):
        from coremusic.audio import ExtendedAudioFile

        src = _make_wav(tmp_path / "a.wav")
        out = tmp_path / "x.m4a"
        assert self._run(_convert_args(src, out, output_format="alac")) == 0
        with ExtendedAudioFile(str(out)) as f:
            assert f.file_format.format_id == "alac"

    def test_unsupported_codec_override_rejected(self, tmp_path):
        from coremusic.cli._utils import CLIError

        src = _make_wav(tmp_path / "a.wav")
        out = tmp_path / "x.m4a"
        with pytest.raises(CLIError):
            self._run(_convert_args(src, out, output_format="mp3"))

    def test_bitrate_kbps_affects_size(self, tmp_path):
        src = _make_noise_wav(tmp_path / "n.wav", seconds=2.0)
        lo = tmp_path / "lo.m4a"
        hi = tmp_path / "hi.m4a"
        # --bitrate is in kbps at the CLI.
        assert self._run(_convert_args(src, lo, bitrate=64)) == 0
        assert self._run(_convert_args(src, hi, bitrate=256)) == 0
        assert lo.stat().st_size < hi.stat().st_size

    def test_bitrate_on_flac_rejected(self, tmp_path):
        from coremusic.cli._utils import CLIError

        src = _make_wav(tmp_path / "a.wav")
        with pytest.raises(CLIError):
            self._run(_convert_args(src, tmp_path / "x.flac", bitrate=128))
