"""Performance benchmarking suite for CoreMusic.

This module provides comprehensive benchmarks for measuring performance
of various CoreMusic operations.
"""

import time
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import statistics

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import coremusic as cm
from coremusic.audio.mmap_file import MMapAudioFile
from coremusic.audio.buffer_pool import BufferPool, get_global_pool


class Benchmark:
    """Base class for benchmarks."""

    def __init__(self, name: str, iterations: int = 10):
        """Initialize benchmark.

        Args:
            name: Benchmark name
            iterations: Number of iterations to run
        """
        self.name = name
        self.iterations = iterations
        self.times: List[float] = []

    def setup(self) -> None:
        """Setup before benchmark runs."""
        pass

    def teardown(self) -> None:
        """Cleanup after benchmark runs."""
        pass

    def run_iteration(self) -> None:
        """Run single benchmark iteration (must be implemented)."""
        raise NotImplementedError

    def run(self) -> Dict[str, float]:
        """Run benchmark and collect statistics.

        Returns:
            Dictionary with timing statistics
        """
        self.setup()

        # Warmup
        self.run_iteration()

        # Timed runs
        self.times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            self.run_iteration()
            end = time.perf_counter()
            self.times.append(end - start)

        self.teardown()

        # Calculate statistics
        return {
            'name': self.name,
            'iterations': self.iterations,
            'mean': statistics.mean(self.times),
            'median': statistics.median(self.times),
            'stdev': statistics.stdev(self.times) if len(self.times) > 1 else 0.0,
            'min': min(self.times),
            'max': max(self.times),
        }


class FileReadBenchmark(Benchmark):
    """Benchmark audio file reading performance."""

    def __init__(self, file_path: str, method: str = "standard"):
        """Initialize file read benchmark.

        Args:
            file_path: Path to test audio file
            method: Read method ("standard", "mmap", "chunks")
        """
        super().__init__(f"File Read ({method})")
        self.file_path = file_path
        self.method = method

    def run_iteration(self) -> None:
        """Run file read iteration."""
        if self.method == "standard":
            with cm.AudioFile(self.file_path) as audio:
                # Get total frame count from format
                duration = audio.duration
                frame_count = int(duration * audio.format.sample_rate)
                data, count = audio.read_packets(0, frame_count)

        elif self.method == "mmap":
            with MMapAudioFile(self.file_path) as audio:
                data = audio.read_as_numpy()

        elif self.method == "chunks":
            with cm.AudioFile(self.file_path) as audio:
                chunk_size = 4096
                duration = audio.duration
                frame_count = int(duration * audio.format.sample_rate)
                for i in range(0, frame_count, chunk_size):
                    data, count = audio.read_packets(i, min(chunk_size, frame_count - i))


class BufferPoolBenchmark(Benchmark):
    """Benchmark buffer pooling performance."""

    def __init__(self, with_pool: bool = True):
        """Initialize buffer pool benchmark.

        Args:
            with_pool: Whether to use buffer pooling
        """
        name = "Buffer Pool" if with_pool else "No Pool"
        super().__init__(name, iterations=1000)
        self.with_pool = with_pool
        self.pool = BufferPool() if with_pool else None

    def run_iteration(self) -> None:
        """Run buffer allocation iteration."""
        size = 8192  # 8KB buffer

        if self.with_pool:
            buffer = self.pool.acquire(size)
            # Simulate processing
            buffer[:100] = b'\x00' * 100
            self.pool.release(buffer)
        else:
            buffer = bytearray(size)
            # Simulate processing
            buffer[:100] = b'\x00' * 100
            # Let buffer be garbage collected


class FormatConversionBenchmark(Benchmark):
    """Benchmark audio format conversion."""

    def __init__(self, file_path: str):
        """Initialize format conversion benchmark.

        Args:
            file_path: Path to test audio file
        """
        super().__init__("Format Conversion")
        self.file_path = file_path

    def run_iteration(self) -> None:
        """Run format conversion iteration."""
        # Read and convert format
        with cm.ExtendedAudioFile(self.file_path) as audio:
            in_format = audio.file_format

            # Create output format (different sample rate)
            out_format = cm.AudioFormat(
                sample_rate=48000.0 if in_format.sample_rate != 48000.0 else 44100.0,
                format_id=in_format.format_id,
                format_flags=in_format.format_flags,
                channels_per_frame=in_format.channels_per_frame,
                bits_per_channel=in_format.bits_per_channel
            )

            # Set client format for conversion
            audio.client_format = out_format

            # Read with automatic conversion (1 second = 44100 frames)
            chunk_size = 4096
            total_read = 0
            target_frames = 44100  # 1 second
            while total_read < target_frames:
                data, count = audio.read(min(chunk_size, target_frames - total_read))
                if count == 0:
                    break
                total_read += count


class NumPyIntegrationBenchmark(Benchmark):
    """Benchmark NumPy integration performance."""

    def __init__(self, file_path: str):
        """Initialize NumPy benchmark.

        Args:
            file_path: Path to test audio file
        """
        super().__init__("NumPy Integration")
        self.file_path = file_path

    def run_iteration(self) -> None:
        """Run NumPy integration iteration."""
        import numpy as np

        with cm.AudioFile(self.file_path) as audio:
            # Read as NumPy array (first second)
            samples = audio.read_as_numpy(0, 44100)

            # Perform NumPy operation
            normalized = samples / (np.max(np.abs(samples)) + 1e-8)

            # Convert back to bytes
            output = normalized.tobytes()


class MMapRandomAccessBenchmark(Benchmark):
    """Benchmark memory-mapped random access."""

    def __init__(self, file_path: str):
        """Initialize mmap random access benchmark.

        Args:
            file_path: Path to test audio file
        """
        super().__init__("MMap Random Access")
        self.file_path = file_path

    def run_iteration(self) -> None:
        """Run random access iteration."""
        with MMapAudioFile(self.file_path) as audio:
            # Random access patterns
            _ = audio[1000:2000]  # Slice
            _ = audio[5000]       # Single frame
            _ = audio[10000:20000]  # Another slice
            _ = audio[::100]       # Strided access


def run_benchmark_suite(test_file: str) -> List[Dict[str, float]]:
    """Run complete benchmark suite.

    Args:
        test_file: Path to test audio file

    Returns:
        List of benchmark results
    """
    benchmarks = [
        FileReadBenchmark(test_file, "standard"),
        FileReadBenchmark(test_file, "mmap"),
        FileReadBenchmark(test_file, "chunks"),
        BufferPoolBenchmark(with_pool=False),
        BufferPoolBenchmark(with_pool=True),
        # FormatConversionBenchmark(test_file),  # Skip - format API issue
        NumPyIntegrationBenchmark(test_file),
        MMapRandomAccessBenchmark(test_file),
    ]

    results = []
    for benchmark in benchmarks:
        print(f"\nRunning: {benchmark.name}...")
        result = benchmark.run()
        results.append(result)
        print(f"  Mean: {result['mean']*1000:.2f}ms (±{result['stdev']*1000:.2f}ms)")

    return results


def print_results(results: List[Dict[str, float]]) -> None:
    """Print benchmark results in formatted table.

    Args:
        results: List of benchmark results
    """
    print("\n" + "="*80)
    print("CoreMusic Performance Benchmark Results")
    print("="*80)
    print(f"{'Benchmark':<30} {'Mean (ms)':>12} {'Median (ms)':>12} {'StdDev (ms)':>12}")
    print("-"*80)

    for result in results:
        print(f"{result['name']:<30} "
              f"{result['mean']*1000:>12.2f} "
              f"{result['median']*1000:>12.2f} "
              f"{result['stdev']*1000:>12.2f}")

    print("="*80)

    # Calculate speedups
    print("\nPerformance Comparisons:")
    print("-"*80)

    # Find standard vs mmap read
    standard_read = next((r for r in results if "standard" in r['name'].lower()), None)
    mmap_read = next((r for r in results if "mmap" in r['name'].lower()), None)

    if standard_read and mmap_read:
        speedup = standard_read['mean'] / mmap_read['mean']
        print(f"Memory-mapped read speedup: {speedup:.2f}x")

    # Find buffer pool comparison
    no_pool = next((r for r in results if "no pool" in r['name'].lower()), None)
    with_pool = next((r for r in results if "buffer pool" in r['name'].lower()), None)

    if no_pool and with_pool:
        speedup = no_pool['mean'] / with_pool['mean']
        improvement = ((no_pool['mean'] - with_pool['mean']) / no_pool['mean']) * 100
        print(f"Buffer pooling speedup: {speedup:.2f}x ({improvement:.1f}% faster)")

    print("="*80)


def print_buffer_pool_stats() -> None:
    """Print buffer pool statistics."""
    pool = get_global_pool()
    stats = pool.stats

    print("\nBuffer Pool Statistics:")
    print("-"*80)
    print(f"Total allocated: {stats['total_allocated']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache misses: {stats['cache_misses']}")
    print(f"Hit rate: {stats['hit_rate']:.1%}")
    print(f"Cached buffers: {stats['cached_buffers']}")
    print("="*80)


def main():
    """Run benchmarking suite."""
    # Check for test file
    test_file = Path(__file__).parent.parent / "tests" / "amen.wav"

    if not test_file.exists():
        print(f"Error: Test file not found: {test_file}")
        print("Please ensure tests/amen.wav exists")
        return 1

    print("CoreMusic Performance Benchmarking Suite")
    print(f"Test file: {test_file}")
    print(f"File size: {test_file.stat().st_size / 1024:.1f} KB")

    # Run benchmarks
    results = run_benchmark_suite(str(test_file))

    # Print results
    print_results(results)

    # Print buffer pool stats
    print_buffer_pool_stats()

    return 0


if __name__ == '__main__':
    sys.exit(main())
