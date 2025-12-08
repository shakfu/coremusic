#!/usr/bin/env python3
"""Batch parallel processing utilities for CoreMusic.

This module provides high-performance batch processing capabilities with:
- Parallel processing using multiprocessing
- Progress tracking and callbacks
- Error handling with retry logic
- Result aggregation and reporting
- Cancellation support

Features:
- Process multiple audio files in parallel
- Custom processing functions with any signature
- Real-time progress monitoring
- Automatic error recovery
- Memory-efficient chunk processing
- Graceful shutdown and cleanup

Example:
    ```python
    from coremusic.utils import batch

    # Simple parallel processing
    def process_file(path):
        with cm.AudioFile(path) as audio:
            return audio.duration

    results = batch.batch_process_parallel(
        items=audio_files,
        func=process_file,
        max_workers=4,
        progress_callback=lambda p: print(f"Progress: {p.percent:.1f}%")
    )
    ```
"""

import concurrent.futures
import logging
import multiprocessing as mp
import time
import traceback
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (Any, Callable, Generic, Iterable, List, Optional, Tuple,
                    TypeVar, Union)

logger = logging.getLogger(__name__)

# Type variables for generic processing
T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type

__all__ = [
    "batch_process_parallel",
    "batch_process_files",
    "BatchResult",
    "BatchProgress",
    "BatchOptions",
    "ProcessingMode",
    "RetryPolicy",
]


# ============================================================================
# Configuration and Options
# ============================================================================


class ProcessingMode(Enum):
    """Processing execution mode."""

    PROCESSES = "processes"  # Use multiprocessing (CPU-bound tasks)
    THREADS = "threads"  # Use threading (I/O-bound tasks)
    SEQUENTIAL = "sequential"  # No parallelization (debugging)


class RetryPolicy(Enum):
    """Retry policy for failed items."""

    NONE = "none"  # No retries
    IMMEDIATE = "immediate"  # Retry immediately
    EXPONENTIAL_BACKOFF = "exponential"  # Retry with exponential backoff


@dataclass
class BatchOptions:
    """Configuration options for batch processing.

    Attributes:
        max_workers: Maximum number of parallel workers (None = CPU count)
        mode: Processing mode (processes, threads, or sequential)
        retry_policy: How to handle retries for failed items
        max_retries: Maximum number of retry attempts per item
        timeout: Maximum time (seconds) per item (None = no timeout)
        chunk_size: Number of items to process per chunk (for chunked mode)
        fail_fast: Stop processing on first error (default: False)
        ordered: Maintain input order in results (may reduce performance)
    """

    max_workers: Optional[int] = None
    mode: ProcessingMode = ProcessingMode.PROCESSES
    retry_policy: RetryPolicy = RetryPolicy.NONE
    max_retries: int = 3
    timeout: Optional[float] = None
    chunk_size: Optional[int] = None
    fail_fast: bool = False
    ordered: bool = False


# ============================================================================
# Progress Tracking
# ============================================================================


@dataclass
class BatchProgress:
    """Progress information for batch processing.

    Attributes:
        total: Total number of items to process
        completed: Number of successfully completed items
        failed: Number of failed items
        in_progress: Number of items currently being processed
        percent: Completion percentage (0-100)
        elapsed_time: Elapsed time in seconds
        estimated_remaining: Estimated remaining time in seconds (None if unknown)
        current_item: Currently processing item (if available)
        items_per_second: Processing rate
    """

    total: int
    completed: int = 0
    failed: int = 0
    in_progress: int = 0
    elapsed_time: float = 0.0
    current_item: Optional[Any] = None

    @property
    def percent(self) -> float:
        """Completion percentage (0-100)."""
        if self.total == 0:
            return 100.0
        return (self.completed + self.failed) / self.total * 100.0

    @property
    def items_per_second(self) -> float:
        """Processing rate in items per second."""
        if self.elapsed_time == 0:
            return 0.0
        return (self.completed + self.failed) / self.elapsed_time

    @property
    def estimated_remaining(self) -> Optional[float]:
        """Estimated remaining time in seconds."""
        if self.items_per_second == 0:
            return None
        remaining_items = self.total - self.completed - self.failed
        return remaining_items / self.items_per_second

    def __str__(self) -> str:
        """Human-readable progress string."""
        eta = f", ETA: {self.estimated_remaining:.1f}s" if self.estimated_remaining else ""
        return (
            f"Progress: {self.percent:.1f}% "
            f"({self.completed}/{self.total} completed, "
            f"{self.failed} failed{eta})"
        )


# ============================================================================
# Result Types
# ============================================================================


@dataclass
class ItemResult(Generic[T, R]):
    """Result for a single processed item.

    Attributes:
        item: The input item that was processed
        result: The processing result (None if failed)
        success: Whether processing succeeded
        error: Error message if failed (None if succeeded)
        error_type: Exception type if failed
        traceback: Full traceback if failed
        attempts: Number of attempts made
        duration: Processing time in seconds
    """

    item: T
    result: Optional[R] = None
    success: bool = True
    error: Optional[str] = None
    error_type: Optional[str] = None
    traceback_str: Optional[str] = None
    attempts: int = 1
    duration: float = 0.0


@dataclass
class BatchResult(Generic[T, R]):
    """Aggregated results from batch processing.

    Attributes:
        results: List of individual item results
        total: Total number of items processed
        successful: Number of successful items
        failed: Number of failed items
        total_duration: Total processing time in seconds
        options: Processing options used
    """

    results: List[ItemResult[T, R]] = field(default_factory=list)
    total: int = 0
    successful: int = 0
    failed: int = 0
    total_duration: float = 0.0
    options: Optional[BatchOptions] = None

    @property
    def success_rate(self) -> float:
        """Success rate as percentage (0-100)."""
        if self.total == 0:
            return 100.0
        return self.successful / self.total * 100.0

    @property
    def successful_results(self) -> List[Optional[R]]:
        """List of successful results only (includes None values)."""
        return [r.result for r in self.results if r.success]

    @property
    def failed_items(self) -> List[T]:
        """List of items that failed processing."""
        return [r.item for r in self.results if not r.success]

    @property
    def errors(self) -> List[Tuple[T, str]]:
        """List of (item, error_message) tuples for failed items."""
        return [(r.item, r.error or "Unknown error") for r in self.results if not r.success]

    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"BatchResult: {self.successful}/{self.total} successful "
            f"({self.success_rate:.1f}%), "
            f"{self.failed} failed, "
            f"duration: {self.total_duration:.2f}s"
        )


# ============================================================================
# Core Processing Functions
# ============================================================================


def _process_item_with_retry(
    item: T,
    func: Callable[[T], R],
    options: BatchOptions,
) -> ItemResult[T, R]:
    """Process a single item with retry logic.

    Args:
        item: Item to process
        func: Processing function
        options: Batch processing options

    Returns:
        ItemResult containing the result or error information
    """
    start_time = time.time()
    attempts = 0
    last_error: Optional[Exception] = None

    max_attempts = 1 if options.retry_policy == RetryPolicy.NONE else options.max_retries + 1

    for attempt in range(max_attempts):
        attempts = attempt + 1
        try:
            # Execute the processing function
            result = func(item)
            duration = time.time() - start_time

            return ItemResult(
                item=item,
                result=result,
                success=True,
                attempts=attempts,
                duration=duration,
            )

        except Exception as e:
            last_error = e

            # On last attempt, record the failure
            if attempt >= max_attempts - 1:
                duration = time.time() - start_time
                return ItemResult(
                    item=item,
                    result=None,
                    success=False,
                    error=str(e),
                    error_type=type(e).__name__,
                    traceback_str=traceback.format_exc(),
                    attempts=attempts,
                    duration=duration,
                )

            # Apply backoff if using exponential backoff
            if options.retry_policy == RetryPolicy.EXPONENTIAL_BACKOFF:
                backoff_time = 2**attempt * 0.1  # 0.1s, 0.2s, 0.4s, ...
                time.sleep(backoff_time)

    # Should never reach here, but satisfy type checker
    duration = time.time() - start_time
    return ItemResult(
        item=item,
        result=None,
        success=False,
        error=str(last_error) if last_error else "Unknown error",
        attempts=attempts,
        duration=duration,
    )


def batch_process_parallel(
    items: Iterable[T],
    func: Callable[[T], R],
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[BatchProgress], None]] = None,
    options: Optional[BatchOptions] = None,
) -> BatchResult[T, R]:
    """Process items in parallel with progress tracking.

    This is the main entry point for parallel batch processing. It handles:
    - Parallel execution using processes or threads
    - Progress tracking with callbacks
    - Error handling and retries
    - Result aggregation

    Args:
        items: Iterable of items to process
        func: Function to apply to each item (must be picklable for multiprocessing)
        max_workers: Maximum number of parallel workers (None = CPU count)
        progress_callback: Optional callback for progress updates
        options: Batch processing options (None = defaults)

    Returns:
        BatchResult containing all results and statistics

    Example:
        ```python
        # Process audio files with progress tracking
        def get_duration(path):
            with cm.AudioFile(path) as audio:
                return audio.duration

        def show_progress(progress):
            print(f"\\r{progress}", end="", flush=True)

        results = batch_process_parallel(
            items=audio_files,
            func=get_duration,
            max_workers=4,
            progress_callback=show_progress
        )

        print(f"\\nProcessed {results.successful} files")
        print(f"Total duration: {sum(results.successful_results)}s")
        ```

    Raises:
        ValueError: If func is not callable
    """
    # Convert items to list for counting
    items_list = list(items)

    if not callable(func):
        raise ValueError("func must be callable")

    # Handle empty items list gracefully
    if not items_list:
        return BatchResult(
            results=[],
            total=0,
            successful=0,
            failed=0,
            total_duration=0.0,
            options=options or BatchOptions(max_workers=max_workers),
        )

    # Use provided options or defaults
    if options is None:
        options = BatchOptions(max_workers=max_workers)
    elif max_workers is not None:
        options.max_workers = max_workers

    # Determine number of workers
    if options.max_workers is None:
        if options.mode == ProcessingMode.PROCESSES:
            options.max_workers = mp.cpu_count()
        elif options.mode == ProcessingMode.THREADS:
            options.max_workers = min(32, (mp.cpu_count() or 1) * 5)
        else:  # SEQUENTIAL
            options.max_workers = 1

    # Initialize progress tracking
    progress = BatchProgress(total=len(items_list))
    start_time = time.time()

    # Choose processing mode
    if options.mode == ProcessingMode.SEQUENTIAL:
        return _process_sequential(items_list, func, options, progress, progress_callback, start_time)
    elif options.mode == ProcessingMode.THREADS:
        return _process_parallel(
            items_list, func, options, progress, progress_callback, start_time, use_threads=True
        )
    else:  # PROCESSES
        return _process_parallel(
            items_list, func, options, progress, progress_callback, start_time, use_threads=False
        )


def _process_sequential(
    items: List[T],
    func: Callable[[T], R],
    options: BatchOptions,
    progress: BatchProgress,
    progress_callback: Optional[Callable[[BatchProgress], None]],
    start_time: float,
) -> BatchResult[T, R]:
    """Process items sequentially (for debugging or testing)."""
    results: List[ItemResult[T, R]] = []

    for item in items:
        progress.current_item = item
        progress.in_progress = 1
        progress.elapsed_time = time.time() - start_time

        if progress_callback:
            progress_callback(progress)

        # Process item
        result = _process_item_with_retry(item, func, options)
        results.append(result)

        # Update progress
        progress.in_progress = 0
        if result.success:
            progress.completed += 1
        else:
            progress.failed += 1
            if options.fail_fast:
                break

        progress.elapsed_time = time.time() - start_time
        if progress_callback:
            progress_callback(progress)

    return BatchResult(
        results=results,
        total=len(items),
        successful=progress.completed,
        failed=progress.failed,
        total_duration=time.time() - start_time,
        options=options,
    )


def _process_parallel(
    items: List[T],
    func: Callable[[T], R],
    options: BatchOptions,
    progress: BatchProgress,
    progress_callback: Optional[Callable[[BatchProgress], None]],
    start_time: float,
    use_threads: bool = False,
) -> BatchResult[T, R]:
    """Process items in parallel using processes or threads."""
    results: List[ItemResult[T, R]] = []

    # Choose executor
    ExecutorClass = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    try:
        with ExecutorClass(max_workers=options.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(_process_item_with_retry, item, func, options): item
                for item in items
            }

            # Process completed futures
            for future in as_completed(future_to_item):
                try:
                    result = future.result(timeout=options.timeout)
                    results.append(result)

                    # Update progress
                    if result.success:
                        progress.completed += 1
                    else:
                        progress.failed += 1
                        if options.fail_fast:
                            # Cancel remaining tasks
                            for f in future_to_item:
                                f.cancel()
                            break

                    progress.elapsed_time = time.time() - start_time
                    if progress_callback:
                        progress_callback(progress)

                except concurrent.futures.TimeoutError:
                    item = future_to_item[future]
                    results.append(
                        ItemResult(
                            item=item,
                            result=None,
                            success=False,
                            error=f"Timeout after {options.timeout}s",
                            error_type="TimeoutError",
                        )
                    )
                    progress.failed += 1

                except Exception as e:
                    item = future_to_item[future]
                    results.append(
                        ItemResult(
                            item=item,
                            result=None,
                            success=False,
                            error=str(e),
                            error_type=type(e).__name__,
                            traceback_str=traceback.format_exc(),
                        )
                    )
                    progress.failed += 1

    except KeyboardInterrupt:
        logger.warning("Batch processing interrupted by user")
        raise

    # Sort results if ordered output requested
    if options.ordered:
        item_to_index = {item: i for i, item in enumerate(items)}
        results.sort(key=lambda r: item_to_index.get(r.item, float("inf")))

    return BatchResult(
        results=results,
        total=len(items),
        successful=progress.completed,
        failed=progress.failed,
        total_duration=time.time() - start_time,
        options=options,
    )


# ============================================================================
# Convenience Functions
# ============================================================================


def batch_process_files(
    file_paths: List[Union[str, Path]],
    func: Callable[[Path], R],
    pattern: Optional[str] = None,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[BatchProgress], None]] = None,
    **kwargs: Any,
) -> BatchResult[Path, R]:
    """Batch process audio files in parallel.

    Convenience function for processing multiple audio files. Handles:
    - Path conversion and validation
    - Glob pattern expansion
    - Progress tracking
    - Error handling per file

    Args:
        file_paths: List of file paths or glob patterns
        func: Function to apply to each file path
        pattern: Optional glob pattern to filter files (e.g., "*.wav")
        max_workers: Maximum number of parallel workers
        progress_callback: Optional progress callback
        **kwargs: Additional arguments passed to BatchOptions

    Returns:
        BatchResult with results for each file

    Example:
        ```python
        def analyze_file(path):
            with cm.AudioFile(path) as audio:
                return {
                    'duration': audio.duration,
                    'sample_rate': audio.format.sample_rate,
                }

        results = batch_process_files(
            file_paths=['audio/*.wav'],
            func=analyze_file,
            max_workers=4
        )

        for result in results.successful_results:
            print(f"Duration: {result['duration']}s")
        ```
    """
    # Convert to Path objects
    paths: List[Path] = []
    for path in file_paths:
        p = Path(path)
        if "*" in str(path) or "?" in str(path):
            # Expand glob pattern
            import glob

            paths.extend(Path(f) for f in glob.glob(str(path)))
        else:
            paths.append(p)

    # Filter by pattern if provided
    if pattern:
        paths = [p for p in paths if p.match(pattern)]

    # Validate paths exist
    valid_paths = [p for p in paths if p.exists()]
    if len(valid_paths) < len(paths):
        missing = len(paths) - len(valid_paths)
        logger.warning(f"{missing} file(s) not found, processing {len(valid_paths)} files")

    # Build options from kwargs
    options = BatchOptions(**kwargs) if kwargs else None

    return batch_process_parallel(
        items=valid_paths,
        func=func,
        max_workers=max_workers,
        progress_callback=progress_callback,
        options=options,
    )
