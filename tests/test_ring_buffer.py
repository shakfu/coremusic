#!/usr/bin/env python3
"""Deterministic tests for the lock-free SPSC ring buffer (no audio hardware)."""

import threading

import pytest

from coremusic import capi

np = pytest.importorskip("numpy")


def _f32(values):
    return np.array(values, dtype=np.float32)


class TestRingBufferBasics:
    def test_capacity_is_power_of_two(self):
        # frames*channels = 8 -> already a power of two
        assert capi.AudioRingBuffer(4, 2).capacity == 8
        # frames*channels = 6 -> next power of two is 8
        assert capi.AudioRingBuffer(6, 1).capacity == 8
        # 10 -> 16
        assert capi.AudioRingBuffer(5, 2).capacity == 16

    def test_rejects_bad_dimensions(self):
        with pytest.raises(ValueError):
            capi.AudioRingBuffer(0, 2)
        with pytest.raises(ValueError):
            capi.AudioRingBuffer(4, 0)

    def test_push_pop_roundtrip(self):
        r = capi.AudioRingBuffer(8, 1)
        assert r.push_floats(_f32([1, 2, 3, 4])) == 4
        assert r.available() == 4
        out = np.zeros(4, dtype=np.float32)
        assert r.pop_into(out) == 4
        assert out.tolist() == [1, 2, 3, 4]
        assert r.available() == 0

    def test_wraparound(self):
        r = capi.AudioRingBuffer(4, 2)  # capacity 8
        r.push_floats(_f32(range(6)))
        out = np.zeros(4, dtype=np.float32)
        r.pop_into(out)  # consume 0..3, read cursor now at 4
        r.push_floats(_f32([10, 11, 12, 13]))  # writes wrap past the end
        out6 = np.zeros(6, dtype=np.float32)
        assert r.pop_into(out6) == 6
        assert out6.tolist() == [4, 5, 10, 11, 12, 13]

    def test_overrun_drops_and_counts(self):
        r = capi.AudioRingBuffer(4, 1)  # capacity 4
        assert r.push_floats(_f32([0, 1, 2, 3])) == 4  # full
        assert r.push_floats(_f32([9, 9])) == 0  # no room
        assert r.overruns == 2
        assert r.underruns == 0

    def test_underrun_counts(self):
        r = capi.AudioRingBuffer(8, 1)
        r.push_floats(_f32([1, 2, 3]))
        out = np.zeros(8, dtype=np.float32)
        assert r.pop_into(out) == 3
        assert r.underruns == 5
        assert r.overruns == 0

    def test_pop_empty_returns_zero(self):
        r = capi.AudioRingBuffer(8, 1)
        out = np.zeros(4, dtype=np.float32)
        assert r.pop_into(out) == 0


class TestRingBufferConcurrent:
    def test_producer_consumer_fifo_integrity(self):
        """A concurrent producer/consumer must preserve order with no loss/dup."""
        ring = capi.AudioRingBuffer(1024, 1)  # capacity 1024
        total = 50_000
        produced = np.arange(total, dtype=np.float32)
        consumed: list[float] = []

        def producer():
            i = 0
            while i < total:
                # Only push what currently fits, so a full ring is a wait rather
                # than a (counted) shortfall. This models a disciplined feeder.
                free = ring.capacity - ring.available()
                if free == 0:
                    continue
                chunk = produced[i : i + min(128, free)]
                i += ring.push_floats(chunk)

        def consumer():
            buf = np.zeros(256, dtype=np.float32)
            while len(consumed) < total:
                got = ring.pop_into(buf)
                if got:
                    consumed.extend(buf[:got].tolist())

        tp = threading.Thread(target=producer)
        tc = threading.Thread(target=consumer)
        tc.start()
        tp.start()
        tp.join(timeout=10)
        tc.join(timeout=10)

        assert len(consumed) == total
        # Exact contiguous ramp -> no reordering, loss, or duplication.
        assert consumed == list(range(total))
        # The producer retried on full, so nothing was dropped.
        assert ring.overruns == 0
