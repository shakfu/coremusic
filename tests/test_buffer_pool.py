"""Tests for buffer pooling functionality."""

import pytest
import threading
from pathlib import Path

from coremusic.audio.buffer_pool import (
    BufferPool,
    PooledBuffer,
    BufferPoolStats,
    get_global_pool,
    reset_global_pool,
)


class TestBufferPool:
    """Test buffer pooling operations."""

    def test_acquire_release(self):
        """Test basic acquire and release."""
        pool = BufferPool()
        size = 1024

        # Acquire buffer
        buffer = pool.acquire(size)
        assert isinstance(buffer, bytearray)
        assert len(buffer) == size

        # Should be cache miss (first allocation)
        assert pool.cache_misses == 1
        assert pool.cache_hits == 0

        # Release buffer
        pool.release(buffer)

        # Acquire again - should be cache hit
        buffer2 = pool.acquire(size)
        assert pool.cache_hits == 1
        assert len(buffer2) == size

    def test_buffer_reuse(self):
        """Test that buffers are reused from pool."""
        pool = BufferPool()
        size = 1024

        # Acquire and release
        buffer1 = pool.acquire(size)
        buffer_id1 = id(buffer1)
        pool.release(buffer1)

        # Acquire again - should be same buffer (reused)
        buffer2 = pool.acquire(size)
        buffer_id2 = id(buffer2)

        # Should be the same object (reused from pool)
        assert buffer_id1 == buffer_id2
        assert pool.cache_hits == 1

    def test_multiple_sizes(self):
        """Test pooling with multiple buffer sizes."""
        pool = BufferPool()

        # Acquire different sizes
        buffer1 = pool.acquire(1024)
        buffer2 = pool.acquire(2048)

        assert len(buffer1) == 1024
        assert len(buffer2) == 2048

        # Release all
        pool.release(buffer1)
        pool.release(buffer2)

        # Each size should have its own pool
        stats = pool.stats
        assert stats['pool_sizes'] == 2
        assert stats['cached_buffers'] == 2

    def test_max_buffers_per_size(self):
        """Test maximum buffers per size limit."""
        max_buffers = 3
        pool = BufferPool(max_buffers_per_size=max_buffers)
        size = 1024

        # Allocate and release more than max
        buffers = [pool.acquire(size) for _ in range(max_buffers + 2)]
        for buffer in buffers:
            pool.release(buffer)

        # Should only cache up to max_buffers
        stats = pool.stats
        assert stats['cached_buffers'] <= max_buffers

    def test_clear(self):
        """Test clearing all cached buffers."""
        pool = BufferPool()

        # Acquire and release multiple buffers
        for size in [1024, 2048]:
            buffer = pool.acquire(size)
            pool.release(buffer)

        # Verify buffers are cached
        assert pool.stats['cached_buffers'] > 0

        # Clear pool
        pool.clear()

        # Should be empty
        assert pool.stats['cached_buffers'] == 0
        assert pool.stats['pool_sizes'] == 0

    def test_clear_size(self):
        """Test clearing buffers of specific size."""
        pool = BufferPool()

        # Acquire and release different sizes
        buffer1 = pool.acquire(1024)
        buffer2 = pool.acquire(2048)
        pool.release(buffer1)
        pool.release(buffer2)

        # Clear only 1024 size
        pool.clear_size(1024)

        # 2048 should still be cached
        stats = pool.stats
        assert stats['pool_sizes'] == 1

        # Acquiring 1024 should be cache miss
        old_misses = pool.cache_misses
        pool.acquire(1024)
        assert pool.cache_misses == old_misses + 1

    def test_statistics_tracking(self):
        """Test statistics tracking."""
        pool = BufferPool()
        size = 1024

        # First acquire - cache miss
        buffer1 = pool.acquire(size)
        assert pool.total_allocated == 1
        assert pool.cache_misses == 1
        assert pool.cache_hits == 0
        assert pool.hit_rate == 0.0

        pool.release(buffer1)

        # Second acquire - cache hit
        buffer2 = pool.acquire(size)
        assert pool.total_allocated == 1
        assert pool.cache_misses == 1
        assert pool.cache_hits == 1
        assert pool.hit_rate == 0.5

        pool.release(buffer2)

        # Third acquire - cache hit
        buffer3 = pool.acquire(size)
        assert pool.cache_hits == 2
        assert pool.hit_rate == 2.0 / 3.0

    def test_stats_property(self):
        """Test stats property returns complete information."""
        pool = BufferPool()

        # Acquire and release
        buffer = pool.acquire(1024)
        pool.release(buffer)
        pool.acquire(1024)

        stats = pool.stats
        assert 'total_allocated' in stats
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats
        assert 'total_requests' in stats
        assert 'hit_rate' in stats
        assert 'cached_buffers' in stats
        assert 'pool_sizes' in stats

        assert stats['total_requests'] == stats['cache_hits'] + stats['cache_misses']

    def test_hit_rate_zero_requests(self):
        """Test hit rate with no requests."""
        pool = BufferPool()
        assert pool.hit_rate == 0.0

    def test_repr(self):
        """Test string representation."""
        pool = BufferPool()
        repr_str = repr(pool)
        assert 'BufferPool' in repr_str
        assert 'allocated=' in repr_str
        assert 'cached=' in repr_str
        assert 'hit_rate=' in repr_str


class TestThreadSafety:
    """Test thread safety of buffer pool."""

    def test_concurrent_acquire_release(self):
        """Test concurrent acquire and release operations."""
        pool = BufferPool()
        size = 1024
        num_threads = 2
        operations_per_thread = 5

        def worker():
            for _ in range(operations_per_thread):
                buffer = pool.acquire(size)
                # Simulate some work
                buffer[:10] = b'\xFF' * 10
                pool.release(buffer)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All operations should have completed
        total_ops = num_threads * operations_per_thread
        stats = pool.stats
        assert stats['total_requests'] == total_ops


class TestPooledBuffer:
    """Test PooledBuffer context manager."""

    def test_context_manager(self):
        """Test context manager acquisition and release."""
        pool = BufferPool()
        size = 1024

        with PooledBuffer(size, pool=pool) as buffer:
            assert isinstance(buffer, bytearray)
            assert len(buffer) == size
            # Buffer should be acquired
            assert pool.cache_misses == 1

        # Buffer should be released after context
        assert pool.stats['cached_buffers'] == 1

    def test_automatic_release_on_exception(self):
        """Test buffer is released even if exception occurs."""
        pool = BufferPool()
        size = 1024

        with pytest.raises(ValueError):
            with PooledBuffer(size, pool=pool) as buffer:
                # Simulate error during processing
                raise ValueError("Processing error")

        # Buffer should still be released
        assert pool.stats['cached_buffers'] == 1

    def test_default_global_pool(self):
        """Test using default global pool."""
        reset_global_pool()  # Start fresh

        size = 1024
        with PooledBuffer(size) as buffer:
            assert isinstance(buffer, bytearray)
            assert len(buffer) == size

        # Should have used global pool
        global_pool = get_global_pool()
        assert global_pool.stats['cached_buffers'] == 1

    def test_buffer_usage(self):
        """Test actual buffer usage within context."""
        size = 1024

        with PooledBuffer(size) as buffer:
            # Write data
            buffer[:100] = b'\xAB' * 100

            # Verify data
            assert buffer[:100] == b'\xAB' * 100


class TestGlobalPool:
    """Test global buffer pool instance."""

    def test_get_global_pool(self):
        """Test getting global pool instance."""
        reset_global_pool()  # Start fresh

        pool1 = get_global_pool()
        pool2 = get_global_pool()

        # Should be same instance
        assert pool1 is pool2

    def test_global_pool_persistence(self):
        """Test global pool persists across calls."""
        reset_global_pool()

        pool = get_global_pool()
        buffer = pool.acquire(1024)
        pool.release(buffer)

        # Get pool again
        pool2 = get_global_pool()
        assert pool2.stats['cached_buffers'] == 1

    def test_reset_global_pool(self):
        """Test resetting global pool."""
        reset_global_pool()

        pool1 = get_global_pool()
        buffer = pool1.acquire(1024)
        pool1.release(buffer)

        assert pool1.stats['cached_buffers'] == 1

        # Reset
        reset_global_pool()

        # New instance
        pool2 = get_global_pool()
        assert pool2.stats['cached_buffers'] == 0


class TestBufferPoolStats:
    """Test BufferPoolStats functionality."""

    def test_track_acquisition(self):
        """Test tracking acquisitions."""
        stats = BufferPoolStats()

        stats.track_acquisition(1024)
        stats.track_acquisition(2048)
        stats.track_acquisition(1024)

        assert stats.acquisitions == 3

    def test_track_release(self):
        """Test tracking releases."""
        stats = BufferPoolStats()

        stats.track_release(1024)
        stats.track_release(2048)

        assert stats.releases == 2

    def test_outstanding_buffers(self):
        """Test outstanding buffer count."""
        stats = BufferPoolStats()

        stats.track_acquisition(1024)
        stats.track_acquisition(2048)
        assert stats.outstanding == 2

        stats.track_release(1024)
        assert stats.outstanding == 1

        stats.track_release(2048)
        assert stats.outstanding == 0

    def test_summary(self):
        """Test statistics summary."""
        stats = BufferPoolStats()

        stats.track_acquisition(1024)
        stats.track_acquisition(1024)
        stats.track_acquisition(2048)
        stats.track_release(1024)

        summary = stats.summary()
        assert 'Acquisitions:' in summary
        assert 'Releases:' in summary
        assert 'Outstanding:' in summary
        assert 'Size Distribution:' in summary
        assert '1024' in summary
        assert '2048' in summary

    def test_reset(self):
        """Test resetting statistics."""
        stats = BufferPoolStats()

        stats.track_acquisition(1024)
        stats.track_acquisition(2048)
        stats.track_release(1024)

        stats.reset()

        assert stats.acquisitions == 0
        assert stats.releases == 0
        assert stats.outstanding == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
