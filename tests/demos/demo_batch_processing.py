#!/usr/bin/env python3
"""Demo: Batch Parallel Processing

Demonstrates the batch processing capabilities of CoreMusic including:
- Parallel audio file processing
- Progress tracking with callbacks
- Error handling and retry logic
- Result aggregation and reporting
- Performance comparisons

This shows how to efficiently process large collections of audio files
using multiple CPU cores with real-time progress monitoring.
"""

import sys
import time
from pathlib import Path

# Add src to path for demo purposes
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import coremusic as cm
from coremusic.utils import batch


def demo_simple_parallel():
    """Demo 1: Simple parallel processing."""
    print("\n" + "=" * 70)
    print("DEMO 1: Simple Parallel Processing")
    print("=" * 70)

    # Process numbers in parallel
    def square(x):
        time.sleep(0.1)  # Simulate work
        return x**2

    print("\nProcessing 10 numbers (squaring them)...")
    result = batch.batch_process_parallel(
        items=list(range(10)),
        func=square,
        max_workers=4,
    )

    print(f"\n{result}")
    print(f"Results: {result.successful_results}")
    print(f"Processing rate: {result.total / result.total_duration:.1f} items/second")


def demo_progress_tracking():
    """Demo 2: Progress tracking with callbacks."""
    print("\n" + "=" * 70)
    print("DEMO 2: Progress Tracking")
    print("=" * 70)

    # Progress callback with visual bar
    def show_progress(progress: batch.BatchProgress):
        bar_length = 40
        filled = int(bar_length * progress.percent / 100)
        bar = "█" * filled + "░" * (bar_length - filled)

        eta_str = f", ETA: {progress.estimated_remaining:.1f}s" if progress.estimated_remaining else ""
        print(f"\r[{bar}] {progress.percent:.1f}% ({progress.completed}/{progress.total}{eta_str})", end="", flush=True)

    def slow_operation(x):
        time.sleep(0.15)
        return x * 10

    print("\nProcessing 20 items with progress bar...")
    result = batch.batch_process_parallel(
        items=list(range(20)),
        func=slow_operation,
        max_workers=4,
        progress_callback=show_progress,
    )
    print()  # New line after progress bar
    print(f"\n{result}")


def demo_error_handling():
    """Demo 3: Error handling and retry logic."""
    print("\n" + "=" * 70)
    print("DEMO 3: Error Handling and Retry")
    print("=" * 70)

    # Flaky function that fails sometimes
    failure_counts = {}

    def flaky_operation(x):
        if x not in failure_counts:
            failure_counts[x] = 0

        failure_counts[x] += 1

        # Fail first 2 attempts for even numbers
        if x % 2 == 0 and failure_counts[x] < 3:
            raise RuntimeError(f"Temporary failure for {x} (attempt {failure_counts[x]})")

        return x * 100

    print("\nProcessing with retry logic (some items will fail initially)...")

    options = batch.BatchOptions(
        retry_policy=batch.RetryPolicy.IMMEDIATE,
        max_retries=5,
    )

    result = batch.batch_process_parallel(
        items=list(range(10)),
        func=flaky_operation,
        options=options,
    )

    print(f"\n{result}")
    print(f"\nRetry statistics:")
    for item, count in sorted(failure_counts.items()):
        print(f"  Item {item}: {count} attempts")


def demo_audio_file_batch():
    """Demo 4: Batch processing of audio files."""
    print("\n" + "=" * 70)
    print("DEMO 4: Audio File Batch Processing")
    print("=" * 70)

    # Find test audio file
    test_file = Path(__file__).parent.parent / "amen.wav"

    if not test_file.exists():
        print(f"\nSkipping: Test audio file not found at {test_file}")
        return

    # Create temporary copies for testing
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create 10 copies of the test file
        print(f"\nCreating 10 test audio files...")
        test_files = []
        for i in range(10):
            dest = tmp_path / f"test_audio_{i}.wav"
            shutil.copy(test_file, dest)
            test_files.append(dest)

        # Process files to extract metadata
        def analyze_audio(path: Path) -> dict:
            with cm.AudioFile(str(path)) as audio:
                return {
                    "file": path.name,
                    "duration": audio.duration,
                    "sample_rate": audio.format.sample_rate,
                    "channels": audio.format.channels_per_frame,
                    "format": audio.format.format_id,
                }

        print(f"Processing {len(test_files)} audio files in parallel...")

        progress_log = []

        def log_progress(progress: batch.BatchProgress):
            progress_log.append(progress.percent)
            print(f"\rProcessing: {progress.completed}/{progress.total} files", end="", flush=True)

        result = batch.batch_process_files(
            file_paths=test_files,
            func=analyze_audio,
            max_workers=4,
            progress_callback=log_progress,
        )

        print()  # New line
        print(f"\n{result}")

        # Display results
        print(f"\nAnalysis Results:")
        print(f"{'File':<20} {'Duration':>10} {'Sample Rate':>12} {'Channels':>8}")
        print("-" * 54)

        for analysis in result.successful_results[:5]:  # Show first 5
            print(
                f"{analysis['file']:<20} "
                f"{analysis['duration']:>10.2f}s "
                f"{analysis['sample_rate']:>12.0f}Hz "
                f"{analysis['channels']:>8}"
            )

        if len(result.successful_results) > 5:
            print(f"... and {len(result.successful_results) - 5} more files")


def demo_performance_comparison():
    """Demo 5: Performance comparison - sequential vs parallel."""
    print("\n" + "=" * 70)
    print("DEMO 5: Performance Comparison")
    print("=" * 70)

    def cpu_intensive_task(x):
        # Simulate CPU-intensive work
        result = 0
        for i in range(1000000):
            result += i % (x + 1)
        return result

    items = list(range(20))

    # Sequential processing
    print("\nSequential processing...")
    start = time.time()
    result_seq = batch.batch_process_parallel(
        items=items,
        func=cpu_intensive_task,
        options=batch.BatchOptions(mode=batch.ProcessingMode.SEQUENTIAL),
    )
    seq_time = time.time() - start
    print(f"Sequential time: {seq_time:.2f}s ({result_seq.total / seq_time:.1f} items/sec)")

    # Parallel processing with 2 workers
    print("\nParallel processing (2 workers)...")
    start = time.time()
    result_par2 = batch.batch_process_parallel(
        items=items,
        func=cpu_intensive_task,
        options=batch.BatchOptions(mode=batch.ProcessingMode.PROCESSES, max_workers=2),
    )
    par2_time = time.time() - start
    print(f"Parallel (2) time: {par2_time:.2f}s ({result_par2.total / par2_time:.1f} items/sec)")
    print(f"Speedup: {seq_time / par2_time:.2f}x")

    # Parallel processing with 4 workers
    print("\nParallel processing (4 workers)...")
    start = time.time()
    result_par4 = batch.batch_process_parallel(
        items=items,
        func=cpu_intensive_task,
        options=batch.BatchOptions(mode=batch.ProcessingMode.PROCESSES, max_workers=4),
    )
    par4_time = time.time() - start
    print(f"Parallel (4) time: {par4_time:.2f}s ({result_par4.total / par4_time:.1f} items/sec)")
    print(f"Speedup: {seq_time / par4_time:.2f}x")


def demo_advanced_options():
    """Demo 6: Advanced options and configuration."""
    print("\n" + "=" * 70)
    print("DEMO 6: Advanced Options")
    print("=" * 70)

    def risky_operation(x):
        import random

        if random.random() < 0.2:  # 20% failure rate
            raise ValueError(f"Random failure for {x}")
        time.sleep(0.05)
        return x**2

    print("\nProcessing with advanced options:")
    print("- Exponential backoff retry")
    print("- 3 maximum retries")
    print("- Timeout of 1 second")
    print("- Ordered results")

    options = batch.BatchOptions(
        retry_policy=batch.RetryPolicy.EXPONENTIAL_BACKOFF,
        max_retries=3,
        timeout=1.0,
        ordered=True,
        max_workers=4,
    )

    result = batch.batch_process_parallel(
        items=list(range(15)),
        func=risky_operation,
        options=options,
    )

    print(f"\n{result}")

    if result.failed > 0:
        print(f"\nFailed items:")
        for item, error in result.errors:
            print(f"  Item {item}: {error}")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("COREMUSIC BATCH PROCESSING DEMO")
    print("=" * 70)
    print("\nThis demo showcases parallel batch processing capabilities:")
    print("- Parallel execution using multiprocessing")
    print("- Real-time progress tracking")
    print("- Error handling with retry logic")
    print("- Audio file batch processing")
    print("- Performance comparisons")

    try:
        demo_simple_parallel()
        demo_progress_tracking()
        demo_error_handling()
        demo_audio_file_batch()
        demo_performance_comparison()
        demo_advanced_options()

        print("\n" + "=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
