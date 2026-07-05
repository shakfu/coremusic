#!/usr/bin/env python3
"""Tests for the real-time audio input (capture) stream and loopback.

Input capture was previously a stub that raised NotImplementedError. Real
capture requires macOS microphone (TCC) permission; tests that need live audio
are gated on a runtime probe and skip cleanly when permission is absent
(`AudioUnitRender` returns kAudioUnitErr_CannotDoInCurrentContext, -10863).
"""

import time

import pytest

from coremusic import capi
from coremusic.audio.streaming import AudioInputStream, AudioProcessor, create_loopback

_PERMISSION_DENIED = -10863


def _capture_permitted() -> bool:
    """Probe whether the process can actually capture microphone audio."""
    try:
        device_id = capi.audio_hardware_get_default_input_device()
        if device_id <= 0:
            return False
        delivered = []
        impl = capi.AudioInputStreamImpl()
        impl.setup(lambda data, fc: delivered.append(fc), device_id, 44100.0, 2)
        impl.start()
        for _ in range(15):
            time.sleep(0.02)
            if delivered:
                break
            if impl.had_error and impl.last_status == _PERMISSION_DENIED:
                break
        impl.stop()
        impl.close()
        return bool(delivered) and not (
            impl.had_error and impl.last_status == _PERMISSION_DENIED
        )
    except Exception:
        return False


capture_permitted = pytest.mark.skipif(
    not _capture_permitted(), reason="microphone capture not permitted in this environment"
)


class TestInputStreamValidation:
    """Validation that does not require microphone permission."""

    def test_setup_requires_callback(self):
        impl = capi.AudioInputStreamImpl()
        with pytest.raises(ValueError):
            impl.setup(None, 1, 44100.0, 2)

    def test_setup_rejects_invalid_device(self):
        impl = capi.AudioInputStreamImpl()
        with pytest.raises(ValueError):
            impl.setup(lambda d, fc: None, 0, 44100.0, 2)

    def test_start_is_not_a_stub(self):
        """start() must no longer raise the old NotImplementedError stub."""
        stream = AudioInputStream()
        try:
            stream.start()
        except RuntimeError as e:
            assert "Cython" not in str(e)
            assert not stream.is_active
        else:
            assert stream.is_active
            stream.stop()


@capture_permitted
class TestInputCapture:
    def test_capture_delivers_audio(self):
        stream = AudioInputStream(channels=2, sample_rate=44100.0, buffer_size=512)
        counter = {"calls": 0, "frames": 0}

        def on_audio(data, frame_count):
            counter["calls"] += 1
            counter["frames"] += frame_count

        stream.add_callback(on_audio)
        with stream:
            assert stream.is_active
            time.sleep(0.3)
        assert not stream.is_active
        assert counter["calls"] > 0
        assert counter["frames"] > 0

    def test_numpy_payload_shape(self):
        pytest.importorskip("numpy")
        stream = AudioInputStream(channels=2, sample_rate=44100.0, buffer_size=512)
        shapes = []
        stream.add_callback(lambda data, fc: shapes.append(getattr(data, "shape", None)))
        with stream:
            time.sleep(0.2)
        assert shapes
        # NumPy payloads are delivered as (frames, channels).
        assert shapes[0] is not None and shapes[0][1] == 2


@capture_permitted
class TestLoopback:
    def test_loopback_runs(self):
        loopback = create_loopback(channels=2, sample_rate=44100.0, buffer_size=256)
        with loopback:
            assert loopback.is_active
            time.sleep(0.3)
        assert not loopback.is_active


@capture_permitted
class TestProcessorTwoRing:
    def test_processor_runs_with_effect(self):
        """AudioProcessor runs an effect end to end via the two-ring worker."""
        np = pytest.importorskip("numpy")
        processor = AudioProcessor(
            lambda audio: audio * 0.5, channels=2, sample_rate=44100.0, buffer_size=256
        )
        with processor:
            assert processor.is_active
            time.sleep(0.3)
        assert not processor.is_active
        # numpy import kept meaningful (effect uses array math)
        assert np is not None
