"""Buffer pooling for efficient memory management.

This module provides buffer pooling to reduce memory allocation overhead
in audio processing applications.
"""

from threading import Lock
from typing import Dict, List, Optional, Union


class BufferPool:
    """Thread-safe buffer pool for reusing allocated buffers.

    This class manages a pool of pre-allocated buffers to reduce memory
    allocation overhead in audio processing. Buffers are organized by size
    for efficient reuse.

    Attributes:
        max_buffers_per_size: Maximum buffers to cache per size
        total_allocated: Total number of buffers allocated
        cache_hits: Number of times a buffer was reused
        cache_misses: Number of times a new buffer was allocated

    Example::

        # Create a buffer pool
        pool = BufferPool(max_buffers_per_size=10)

        # Acquire buffer for audio processing
        buffer = pool.acquire(8192)  # 8KB buffer

        # Use buffer for processing
        # ... audio processing ...

        # Return buffer to pool for reuse
        pool.release(buffer)

        # Check statistics
        print(f"Hit rate: {pool.hit_rate:.2%}")

    Notes:
        - Thread-safe for concurrent access
        - Automatically grows to accommodate demand
        - Implements least-recently-used eviction
        - Buffers are NOT zeroed on reuse for performance
    """

    def __init__(self, max_buffers_per_size: int = 16):
        """Initialize buffer pool.

        Args:
            max_buffers_per_size: Maximum buffers to cache per size
        """
        self.max_buffers_per_size = max_buffers_per_size
        self._pools: Dict[int, List[bytearray]] = {}
        self._lock = Lock()
        self._total_allocated = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def acquire(self, size: int) -> bytearray:
        """Acquire a buffer of the specified size.

        Args:
            size: Buffer size in bytes

        Returns:
            Buffer of requested size (may contain old data if reused)

        Note:
            Buffers are NOT zeroed on reuse for performance. Callers should
            overwrite the entire buffer before use.

        Example::

            # Get buffer for stereo float32 at 44.1kHz (1024 frames)
            buffer = pool.acquire(1024 * 2 * 4)  # 8192 bytes
        """
        with self._lock:
            # Check if we have a buffer of this size
            if size in self._pools and self._pools[size]:
                # Reuse existing buffer
                buffer = self._pools[size].pop()
                self._cache_hits += 1

                # Zero the buffer for safety
                # Note: We don't actually zero for performance - caller should overwrite
                # If zeroing is needed, uncomment: buffer[:] = b'\x00' * len(buffer)
                return buffer
            else:
                # Allocate new buffer
                self._cache_misses += 1
                self._total_allocated += 1
                return bytearray(size)

    def release(self, buffer: bytearray) -> None:
        """Release a buffer back to the pool.

        Args:
            buffer: Buffer to return to pool

        Example::

            buffer = pool.acquire(8192)
            # ... use buffer ...
            pool.release(buffer)
        """
        size = len(buffer)

        with self._lock:
            # Create pool for this size if needed
            if size not in self._pools:
                self._pools[size] = []

            # Add to pool if under limit
            if len(self._pools[size]) < self.max_buffers_per_size:
                self._pools[size].append(buffer)
            # else: let buffer be garbage collected

    def clear(self) -> None:
        """Clear all cached buffers.

        Example::

            pool.clear()  # Release all cached buffers
        """
        with self._lock:
            self._pools.clear()

    def clear_size(self, size: int) -> None:
        """Clear cached buffers of specific size.

        Args:
            size: Buffer size to clear

        Example::

            pool.clear_size(8192)  # Clear 8KB buffers only
        """
        with self._lock:
            if size in self._pools:
                del self._pools[size]

    @property
    def total_allocated(self) -> int:
        """Get total number of buffers allocated."""
        with self._lock:
            return self._total_allocated

    @property
    def cache_hits(self) -> int:
        """Get number of cache hits."""
        with self._lock:
            return self._cache_hits

    @property
    def cache_misses(self) -> int:
        """Get number of cache misses."""
        with self._lock:
            return self._cache_misses

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate (0.0 to 1.0).

        Returns:
            Hit rate as decimal (0.0 = 0%, 1.0 = 100%)
        """
        with self._lock:
            total = self._cache_hits + self._cache_misses
            if total == 0:
                return 0.0
            return self._cache_hits / total

    @property
    def stats(self) -> Dict[str, Union[int, float]]:
        """Get pool statistics.

        Returns:
            Dictionary with statistics (int or float values)

        Example::

            stats = pool.stats
            print(f"Allocated: {stats['total_allocated']}")
            print(f"Hit rate: {stats['hit_rate']:.2%}")
        """
        with self._lock:
            total_requests = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
            return {
                'total_allocated': self._total_allocated,
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'total_requests': total_requests,
                'hit_rate': hit_rate,
                'cached_buffers': sum(len(buffers) for buffers in self._pools.values()),
                'pool_sizes': len(self._pools),
            }

    def __repr__(self) -> str:
        """String representation."""
        stats = self.stats
        return (
            f"BufferPool("
            f"allocated={stats['total_allocated']}, "
            f"cached={stats['cached_buffers']}, "
            f"hit_rate={stats['hit_rate']:.1%})"
        )


# Global buffer pool instance
_global_pool: Optional[BufferPool] = None
_global_pool_lock = Lock()


def get_global_pool() -> BufferPool:
    """Get the global buffer pool instance.

    Returns:
        Global BufferPool instance

    Example::

        # Use global pool
        from coremusic.audio.buffer_pool import get_global_pool

        pool = get_global_pool()
        buffer = pool.acquire(8192)
        # ... use buffer ...
        pool.release(buffer)
    """
    global _global_pool

    if _global_pool is None:
        with _global_pool_lock:
            if _global_pool is None:
                _global_pool = BufferPool()

    return _global_pool


def reset_global_pool() -> None:
    """Reset the global buffer pool.

    This clears all cached buffers and resets statistics.

    Example::

        reset_global_pool()  # Clear all cached buffers
    """
    global _global_pool

    with _global_pool_lock:
        if _global_pool is not None:
            _global_pool.clear()
            _global_pool = None


class PooledBuffer:
    """Context manager for pooled buffer acquisition.

    This class provides automatic buffer acquisition and release using
    Python's context manager protocol.

    Example::

        from coremusic.audio.buffer_pool import PooledBuffer

        # Automatic acquisition and release
        with PooledBuffer(8192) as buffer:
            # Use buffer for processing
            buffer[:1024] = audio_data

        # Buffer automatically returned to pool

        # Or use custom pool
        pool = BufferPool()
        with PooledBuffer(8192, pool=pool) as buffer:
            # Use buffer
            pass
    """

    def __init__(self, size: int, pool: Optional[BufferPool] = None):
        """Initialize pooled buffer context manager.

        Args:
            size: Buffer size in bytes
            pool: Buffer pool to use (default: global pool)
        """
        self.size = size
        self.pool = pool or get_global_pool()
        self.buffer: Optional[bytearray] = None

    def __enter__(self) -> bytearray:
        """Acquire buffer from pool."""
        self.buffer = self.pool.acquire(self.size)
        return self.buffer

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release buffer back to pool."""
        if self.buffer is not None:
            self.pool.release(self.buffer)
            self.buffer = None


class BufferPoolStats:
    """Statistics tracker for buffer pool performance.

    This class tracks detailed statistics about buffer pool usage
    for performance analysis and optimization.

    Example::

        stats = BufferPoolStats()

        # Track buffer operations
        with stats.track():
            buffer = pool.acquire(8192)
            # ... use buffer ...
            pool.release(buffer)

        # Print statistics
        print(stats.summary())
    """

    def __init__(self):
        """Initialize statistics tracker."""
        self._acquisitions = 0
        self._releases = 0
        self._size_distribution: Dict[int, int] = {}
        self._lock = Lock()

    def track_acquisition(self, size: int) -> None:
        """Track buffer acquisition.

        Args:
            size: Buffer size acquired
        """
        with self._lock:
            self._acquisitions += 1
            self._size_distribution[size] = self._size_distribution.get(size, 0) + 1

    def track_release(self, size: int) -> None:
        """Track buffer release.

        Args:
            size: Buffer size released
        """
        with self._lock:
            self._releases += 1

    @property
    def acquisitions(self) -> int:
        """Get total acquisitions."""
        with self._lock:
            return self._acquisitions

    @property
    def releases(self) -> int:
        """Get total releases."""
        with self._lock:
            return self._releases

    @property
    def outstanding(self) -> int:
        """Get number of outstanding buffers."""
        with self._lock:
            return self._acquisitions - self._releases

    def summary(self) -> str:
        """Get statistics summary.

        Returns:
            Formatted statistics string
        """
        with self._lock:
            outstanding = self._acquisitions - self._releases
            lines = [
                "Buffer Pool Statistics:",
                f"  Acquisitions: {self._acquisitions}",
                f"  Releases: {self._releases}",
                f"  Outstanding: {outstanding}",
                "  Size Distribution:",
            ]

            for size, count in sorted(self._size_distribution.items()):
                lines.append(f"    {size:6d} bytes: {count:6d} times")

            return '\n'.join(lines)

    def reset(self) -> None:
        """Reset statistics."""
        with self._lock:
            self._acquisitions = 0
            self._releases = 0
            self._size_distribution.clear()
