#!/usr/bin/env python3
"""Tests for batch parallel processing utilities."""

import os
import time
import tempfile
from pathlib import Path
import pytest
import coremusic as cm
from coremusic.utils.batch import (
    batch_process_parallel,
    batch_process_files,
    BatchResult,
    BatchProgress,
    BatchOptions,
    ProcessingMode,
    RetryPolicy,
)


# ============================================================================
# Module-level functions (needed for multiprocessing pickling)
# ============================================================================


def square(x):
    """Square a number."""
    return x * x


def double(x):
    """Double a number."""
    return x * 2


def add_one(x):
    """Add one to a number."""
    return x + 1


def multiply_by_three(x):
    """Multiply by three."""
    return x * 3


def slow_func(x):
    """Slow function for testing."""
    time.sleep(0.01)
    return x


def simple_calc(x):
    """Simple calculation."""
    return x**2 + 2 * x + 1


def failing_func(x):
    """Fail on even numbers."""
    if x % 2 == 0:
        raise ValueError(f"Even number: {x}")
    return x


def always_fail(x):
    """Always fail."""
    raise ValueError("Nope")


def return_none_for_even(x):
    """Return None for even numbers."""
    return None if x % 2 == 0 else x


def identity(x):
    """Return the input unchanged."""
    return x


def slow_func_with_timeout(x):
    """Slow function that sleeps for 2 seconds on x==2."""
    if x == 2:
        time.sleep(2.0)
    return x


def process_audio_file_info(path):
    """Extract audio file information."""
    with cm.AudioFile(str(path)) as audio:
        return {
            "path": str(path),
            "duration": audio.duration,
            "sample_rate": audio.format.sample_rate,
        }


def get_duration(path):
    """Get audio file duration."""
    with cm.AudioFile(str(path)) as audio:
        return audio.duration


def count_channels(path):
    """Count channels in audio file."""
    with cm.AudioFile(str(path)) as audio:
        return audio.format.channels_per_frame


def get_sample_rate(path):
    """Get sample rate from audio file."""
    with cm.AudioFile(str(path)) as audio:
        return audio.format.sample_rate


def dummy_func(path):
    """Return the file name."""
    return path.name


def detailed_error(x):
    """Raise detailed error for x==3."""
    if x == 3:
        raise RuntimeError("Specific error for item 3")
    return x


def reverse_index(x):
    """Reverse processing time - slower for earlier items."""
    time.sleep(0.01 * (10 - x))
    return x * 10


def analyze_audio(path):
    """Analyze audio file and return metadata."""
    with cm.AudioFile(str(path)) as audio:
        return {
            "path": path.name,
            "duration": audio.duration,
            "sample_rate": audio.format.sample_rate,
            "channels": audio.format.channels_per_frame,
        }


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_audio_file():
    """Path to sample audio file for testing."""
    return Path(__file__).parent / "data" / "wav" / "amen.wav"


@pytest.fixture
def temp_audio_files(tmp_path, sample_audio_file):
    """Create temporary copies of audio files for testing."""
    if not sample_audio_file.exists():
        pytest.skip(f"Sample audio file not found: {sample_audio_file}")

    # Create 5 copies
    files = []
    for i in range(5):
        dest = tmp_path / f"test_audio_{i}.wav"
        import shutil

        shutil.copy(sample_audio_file, dest)
        files.append(dest)

    return files


# ============================================================================
# Basic Functionality Tests
# ============================================================================


class TestBatchProcessBasics:
    """Test basic batch processing functionality."""

    def test_simple_parallel_processing(self):
        """Test basic parallel processing of numbers."""
        items = list(range(10))
        result = batch_process_parallel(items, square, max_workers=2)

        assert result.total == 10
        assert result.successful == 10
        assert result.failed == 0
        assert result.success_rate == 100.0
        # Results may be unordered, so sort for comparison
        assert sorted(result.successful_results) == [x * x for x in items]

    def test_empty_items_returns_empty_result(self):
        """Test that empty items list returns empty result."""
        result = batch_process_parallel([], identity)
        assert result.total == 0
        assert result.successful == 0
        assert result.failed == 0
        assert len(result.results) == 0

    def test_non_callable_func_raises_error(self):
        """Test that non-callable func raises ValueError."""
        with pytest.raises(ValueError, match="func must be callable"):
            batch_process_parallel([1, 2, 3], "not a function")  # type: ignore

    def test_sequential_mode(self):
        """Test sequential processing mode."""
        options = BatchOptions(mode=ProcessingMode.SEQUENTIAL)
        result = batch_process_parallel(
            items=list(range(5)), func=double, options=options
        )

        assert result.successful == 5
        assert result.successful_results == [0, 2, 4, 6, 8]

    def test_thread_mode(self):
        """Test thread-based processing mode."""
        options = BatchOptions(mode=ProcessingMode.THREADS, max_workers=2)
        result = batch_process_parallel(
            items=list(range(5)), func=add_one, options=options
        )

        assert result.successful == 5
        assert sorted(result.successful_results) == [1, 2, 3, 4, 5]

    def test_process_mode(self):
        """Test process-based processing mode (default)."""
        options = BatchOptions(mode=ProcessingMode.PROCESSES, max_workers=2)
        result = batch_process_parallel(
            items=list(range(5)), func=multiply_by_three, options=options
        )

        assert result.successful == 5
        assert sorted(result.successful_results) == [0, 3, 6, 9, 12]


# ============================================================================
# Progress Tracking Tests
# ============================================================================


class TestProgressTracking:
    """Test progress tracking and callbacks."""

    def test_progress_callback_called(self):
        """Test that progress callback is called during processing."""
        progress_updates = []

        def record_progress(progress: BatchProgress):
            progress_updates.append(progress.percent)

        batch_process_parallel(
            items=list(range(5)),
            func=slow_func,
            max_workers=2,
            progress_callback=record_progress,
        )

        # Progress should have been updated multiple times
        assert len(progress_updates) > 0
        # Final progress should be 100%
        assert progress_updates[-1] == 100.0

    def test_progress_attributes(self):
        """Test BatchProgress attributes."""
        progress = BatchProgress(total=100, completed=75, failed=5)

        assert progress.percent == 80.0  # (75 + 5) / 100
        assert progress.total == 100
        assert progress.completed == 75
        assert progress.failed == 5

    def test_progress_estimated_remaining(self):
        """Test estimated remaining time calculation."""
        progress = BatchProgress(total=100, completed=50, failed=0)
        progress.elapsed_time = 10.0  # 10 seconds elapsed

        # 50 items done in 10s = 5 items/s
        # 50 items remaining / 5 items/s = 10s estimated
        assert progress.items_per_second == 5.0
        assert progress.estimated_remaining == 10.0

    def test_progress_string_representation(self):
        """Test progress string representation."""
        progress = BatchProgress(total=100, completed=75, failed=5)
        progress_str = str(progress)

        assert "80.0%" in progress_str
        assert "75/100" in progress_str
        assert "5 failed" in progress_str


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling and retry logic."""

    def test_handle_exceptions_gracefully(self):
        """Test that exceptions are caught and recorded."""
        result = batch_process_parallel(
            items=list(range(10)), func=failing_func, max_workers=2
        )

        assert result.total == 10
        assert result.successful == 5  # Odd numbers
        assert result.failed == 5  # Even numbers
        assert len(result.failed_items) == 5
        assert len(result.errors) == 5

    def test_retry_policy_none(self):
        """Test no retry policy."""
        attempts_count = []

        def track_attempts(x):
            attempts_count.append(x)
            raise ValueError("Always fails")

        # Use thread mode so attempts_count can be shared
        options = BatchOptions(
            retry_policy=RetryPolicy.NONE, mode=ProcessingMode.THREADS
        )
        result = batch_process_parallel(
            items=[1, 2, 3], func=track_attempts, options=options
        )

        assert result.failed == 3
        # Each item attempted exactly once
        assert len(attempts_count) == 3

    def test_retry_policy_immediate(self):
        """Test immediate retry policy."""
        call_count = {1: 0, 2: 0}

        def fail_twice(x):
            call_count[x] += 1
            if call_count[x] <= 2:
                raise ValueError("Fail first two times")
            return x * 10

        # Use thread mode so call_count can be shared
        options = BatchOptions(
            retry_policy=RetryPolicy.IMMEDIATE,
            max_retries=3,
            mode=ProcessingMode.THREADS,
        )
        result = batch_process_parallel(items=[1, 2], func=fail_twice, options=options)

        assert result.successful == 2
        assert call_count[1] == 3  # Failed 2 times, succeeded on 3rd
        assert call_count[2] == 3

    def test_fail_fast_mode(self):
        """Test fail-fast mode stops on first error."""

        def fail_on_five(x):
            if x == 5:
                raise ValueError("Hit 5!")
            return x

        options = BatchOptions(fail_fast=True, mode=ProcessingMode.SEQUENTIAL)
        result = batch_process_parallel(
            items=list(range(10)), func=fail_on_five, options=options
        )

        # Should stop after hitting 5
        assert result.failed == 1
        # May have completed some before the failure
        assert result.successful <= 5

    @pytest.mark.skip(
        reason="Timeout feature needs reimplementation - as_completed() yields "
        "already-completed futures, so timeout on future.result() has no effect. "
        "Proper per-item timeout requires checking futures periodically and "
        "canceling those that exceed the time limit."
    )
    def test_timeout_handling(self):
        """Test timeout handling for slow operations.

        Note: This test is skipped because the current timeout implementation
        doesn't work correctly. as_completed() yields futures that are already
        complete, so calling result(timeout=X) on them returns immediately.

        To implement proper timeout:
        1. Check futures periodically while they're running
        2. Cancel futures that exceed the timeout
        3. Or wrap the processing function with timeout enforcement
        """
        options = BatchOptions(timeout=0.5, mode=ProcessingMode.PROCESSES)
        result = batch_process_parallel(
            items=[1, 2, 3], func=slow_func_with_timeout, options=options
        )

        # Item 2 should timeout
        assert result.failed >= 1
        # Check that one failure was due to timeout
        timeout_errors = [e for e in result.errors if "Timeout" in e[1]]
        assert len(timeout_errors) >= 1

    def test_error_details_captured(self):
        """Test that error details are properly captured."""
        result = batch_process_parallel(items=[1, 2, 3, 4], func=detailed_error)

        failed_result = [r for r in result.results if not r.success][0]
        assert failed_result.item == 3
        assert "Specific error for item 3" in failed_result.error
        assert failed_result.error_type == "RuntimeError"
        assert failed_result.traceback_str is not None


# ============================================================================
# Audio File Processing Tests
# ============================================================================


class TestAudioFileProcessing:
    """Test batch processing of audio files."""

    def test_batch_process_audio_files(self, temp_audio_files):
        """Test processing multiple audio files."""
        result = batch_process_files(
            file_paths=temp_audio_files, func=get_duration, max_workers=2
        )

        assert result.successful == len(temp_audio_files)
        assert result.failed == 0
        # All files should have same duration (they're copies)
        durations = result.successful_results
        assert all(abs(d - durations[0]) < 0.01 for d in durations)

    def test_batch_process_with_glob_pattern(self, tmp_path, sample_audio_file):
        """Test batch processing with glob patterns."""
        if not sample_audio_file.exists():
            pytest.skip(f"Sample audio file not found: {sample_audio_file}")

        # Create test files
        import shutil

        for i in range(3):
            shutil.copy(sample_audio_file, tmp_path / f"test_{i}.wav")

        pattern = str(tmp_path / "*.wav")
        result = batch_process_files(file_paths=[pattern], func=count_channels)

        assert result.successful == 3

    def test_batch_process_nonexistent_files(self):
        """Test handling of non-existent files."""
        result = batch_process_files(
            file_paths=[Path("/nonexistent1.wav"), Path("/nonexistent2.wav")],
            func=dummy_func,
        )

        # Files don't exist, so nothing should be processed
        assert result.total == 0

    def test_batch_process_mixed_files(self, tmp_path, sample_audio_file):
        """Test processing mix of valid and invalid files."""
        if not sample_audio_file.exists():
            pytest.skip(f"Sample audio file not found: {sample_audio_file}")

        # Create one valid file
        import shutil

        valid_file = tmp_path / "valid.wav"
        shutil.copy(sample_audio_file, valid_file)

        # Create one invalid file (empty)
        invalid_file = tmp_path / "invalid.wav"
        invalid_file.write_bytes(b"not a wav file")

        result = batch_process_files(
            file_paths=[valid_file, invalid_file], func=get_sample_rate
        )

        assert result.successful >= 1  # At least the valid file
        # Invalid file should fail
        if result.failed > 0:
            assert any("invalid.wav" in str(item) for item in result.failed_items)


# ============================================================================
# Result Aggregation Tests
# ============================================================================


class TestResultAggregation:
    """Test result aggregation and reporting."""

    def test_batch_result_properties(self):
        """Test BatchResult computed properties."""
        from coremusic.utils.batch import ItemResult

        results = [
            ItemResult(item=1, result=10, success=True),
            ItemResult(item=2, result=20, success=True),
            ItemResult(item=3, result=None, success=False, error="Failed"),
        ]

        batch_result = BatchResult(
            results=results, total=3, successful=2, failed=1, total_duration=1.5
        )

        assert batch_result.success_rate == pytest.approx(66.67, rel=0.01)
        assert batch_result.successful_results == [10, 20]
        assert batch_result.failed_items == [3]
        assert len(batch_result.errors) == 1
        assert batch_result.errors[0] == (3, "Failed")

    def test_batch_result_string(self):
        """Test BatchResult string representation."""
        result = BatchResult(total=100, successful=95, failed=5, total_duration=10.0)
        result_str = str(result)

        assert "95/100" in result_str
        assert "95.0%" in result_str
        assert "5 failed" in result_str
        assert "10.00s" in result_str

    def test_ordered_results(self):
        """Test that ordered option maintains input order."""
        options = BatchOptions(ordered=True, mode=ProcessingMode.THREADS)
        result = batch_process_parallel(
            items=list(range(10)), func=reverse_index, options=options
        )

        # Results should be in order despite processing times
        assert [r.item for r in result.results] == list(range(10))
        assert [r.result for r in result.results] == [x * 10 for x in range(10)]


# ============================================================================
# Performance and Concurrency Tests
# ============================================================================


class TestPerformance:
    """Test performance and concurrency aspects."""

    def test_parallel_faster_than_sequential(self):
        """Test that parallel processing is actually faster."""

        def slow_func(x):
            time.sleep(0.05)
            return x

        items = list(range(20))

        # Sequential
        start = time.time()
        result_seq = batch_process_parallel(
            items, slow_func, options=BatchOptions(mode=ProcessingMode.SEQUENTIAL)
        )
        seq_time = time.time() - start

        # Parallel with 4 workers
        start = time.time()
        result_par = batch_process_parallel(
            items,
            slow_func,
            options=BatchOptions(mode=ProcessingMode.THREADS, max_workers=4),
        )
        par_time = time.time() - start

        # Parallel should be significantly faster
        assert par_time < seq_time * 0.6  # At least 40% faster
        assert result_seq.successful == result_par.successful

    def test_max_workers_respected(self):
        """Test that max_workers setting is respected."""
        # This is hard to test directly, but we can verify the option is set
        options = BatchOptions(max_workers=3)
        result = batch_process_parallel(
            items=[1, 2, 3], func=lambda x: x, options=options
        )

        assert result.options.max_workers == 3

    def test_large_batch_processing(self):
        """Test processing a large batch of items."""
        items = list(range(1000))
        result = batch_process_parallel(items, simple_calc, max_workers=4)

        assert result.total == 1000
        assert result.successful == 1000
        assert result.failed == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_audio_analysis_pipeline(self, temp_audio_files):
        """Test complete audio analysis pipeline."""
        if len(temp_audio_files) == 0:
            pytest.skip("No test audio files available")

        results_list = []

        def track_progress(progress: BatchProgress):
            results_list.append(progress.percent)

        result = batch_process_files(
            file_paths=temp_audio_files,
            func=analyze_audio,
            max_workers=2,
            progress_callback=track_progress,
        )

        assert result.successful == len(temp_audio_files)
        assert len(result.successful_results) == len(temp_audio_files)
        # Progress was tracked
        assert len(results_list) > 0

        # All results have expected keys
        for analysis in result.successful_results:
            assert "path" in analysis
            assert "duration" in analysis
            assert "sample_rate" in analysis
            assert "channels" in analysis

    def test_error_recovery_with_retry(self):
        """Test error recovery with retry logic."""
        failure_counts = {}

        def flaky_func(x):
            if x not in failure_counts:
                failure_counts[x] = 0

            failure_counts[x] += 1

            # Fail first 2 attempts, succeed on 3rd
            if failure_counts[x] < 3:
                raise RuntimeError(f"Attempt {failure_counts[x]} failed")

            return x * 100

        # Use thread mode so failure_counts can be shared
        options = BatchOptions(
            retry_policy=RetryPolicy.IMMEDIATE,
            max_retries=5,  # Allow enough retries
            mode=ProcessingMode.THREADS,
        )

        result = batch_process_parallel(
            items=[1, 2, 3, 4, 5], func=flaky_func, options=options
        )

        # All should eventually succeed
        assert result.successful == 5
        assert result.failed == 0
        # Each item should have been attempted 3 times
        for item in [1, 2, 3, 4, 5]:
            assert failure_counts[item] == 3


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_item(self):
        """Test processing a single item."""
        result = batch_process_parallel(items=[42], func=double)

        assert result.total == 1
        assert result.successful == 1
        assert result.successful_results == [84]

    def test_all_items_fail(self):
        """Test when all items fail."""
        result = batch_process_parallel(items=[1, 2, 3], func=always_fail)

        assert result.total == 3
        assert result.successful == 0
        assert result.failed == 3
        assert result.success_rate == 0.0

    def test_none_results(self):
        """Test handling of None results."""
        result = batch_process_parallel(items=[1, 2, 3, 4, 5], func=return_none_for_even)

        # All should succeed, some with None results
        assert result.successful == 5
        assert None in result.successful_results

    def test_custom_worker_count(self):
        """Test custom worker count."""
        for workers in [1, 2, 4, 8]:
            result = batch_process_parallel(
                items=list(range(10)), func=identity, max_workers=workers
            )
            assert result.successful == 10
