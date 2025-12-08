#!/usr/bin/env python3
"""CoreMusic utilities package.

This package contains various utility modules for CoreMusic:
- batch: Parallel batch processing with progress tracking
- scipy: SciPy-based audio signal processing utilities (optional)
- fourcc: FourCC (four-character code) conversion utilities
"""

# Import utilities modules
from . import batch, fourcc, scipy
# Import commonly used batch processing functions
from .batch import (BatchOptions, BatchProgress, BatchResult, ProcessingMode,
                    RetryPolicy, batch_process_files, batch_process_parallel)

__all__ = [
    # Submodules
    "batch",
    "scipy",
    "fourcc",
    # Batch processing
    "batch_process_parallel",
    "batch_process_files",
    "BatchResult",
    "BatchProgress",
    "BatchOptions",
    "ProcessingMode",
    "RetryPolicy",
]
