#!/usr/bin/env python3
"""Tests for the generator-driven real-time audio output stream.

The output stream was previously a stub that raised NotImplementedError; these
tests exercise the Cython render-callback backend.
"""

import struct
import time

import pytest
from conftest import has_audio_output

from coremusic import capi
from coremusic.audio.streaming import AudioOutputStream


def _silence_generator(counter, channels=2):
    def gen(num_frames):
        counter["calls"] += 1
        counter["frames"] += num_frames
        return struct.pack("<%df" % (num_frames * channels), *([0.0] * num_frames * channels))

    return gen


def _wait_for_calls(counter, timeout=5.0):
    """Poll until the render callback has fired at least once.

    The CoreAudio output unit can take longer than a fixed sleep to prime on a
    loaded/headless CI runner, so wait for the first callback rather than racing
    a hardcoded duration. Returns as soon as a call is observed.
    """
    deadline = time.monotonic() + timeout
    while counter["calls"] == 0 and time.monotonic() < deadline:
        time.sleep(0.02)


@has_audio_output
class TestAudioOutputStreamImpl:
    def test_lifecycle_and_generator_called(self):
        counter = {"calls": 0, "frames": 0}
        impl = capi.AudioOutputStreamImpl()
        impl.setup(_silence_generator(counter), 44100.0, 2)
        assert impl.is_active is False
        impl.start()
        assert impl.is_active is True
        _wait_for_calls(counter)
        impl.stop()
        assert impl.is_active is False
        impl.close()
        assert counter["calls"] > 0
        assert counter["frames"] > 0
        assert impl.had_error is False

    def test_setup_requires_generator(self):
        impl = capi.AudioOutputStreamImpl()
        with pytest.raises(ValueError):
            impl.setup(None, 44100.0, 2)


@has_audio_output
class TestAudioOutputStream:
    def test_context_manager_bytes_generator(self):
        counter = {"calls": 0, "frames": 0}
        stream = AudioOutputStream(channels=2, sample_rate=44100.0, buffer_size=512)
        stream.set_generator(_silence_generator(counter))
        with stream:
            assert stream.is_active is True
            _wait_for_calls(counter)
        assert stream.is_active is False
        assert counter["calls"] > 0

    def test_numpy_mono_generator_broadcasts(self):
        np = pytest.importorskip("numpy")
        counter = {"calls": 0}

        def gen(num_frames):
            counter["calls"] += 1
            return np.zeros(num_frames, dtype=np.float32)  # mono -> broadcast

        stream = AudioOutputStream(channels=2, sample_rate=44100.0, buffer_size=512)
        stream.set_generator(gen)
        with stream:
            _wait_for_calls(counter)
        assert counter["calls"] > 0

    def test_start_without_generator_raises(self):
        stream = AudioOutputStream(channels=2, sample_rate=44100.0)
        with pytest.raises(RuntimeError):
            stream.start()

    def test_generator_exception_does_not_crash(self):
        def bad_gen(num_frames):
            raise ValueError("boom")

        stream = AudioOutputStream(channels=2, sample_rate=44100.0, buffer_size=512)
        stream.set_generator(bad_gen)
        # A raising generator must not crash the process; output is silence.
        with stream:
            time.sleep(0.1)
        assert stream.is_active is False
