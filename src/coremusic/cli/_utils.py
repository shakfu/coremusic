"""Shared CLI utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NoReturn

# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_FILE_NOT_FOUND = 2
EXIT_DEVICE_NOT_FOUND = 3
EXIT_INVALID_FORMAT = 4


class CLIError(Exception):
    """Base CLI error with exit code."""

    exit_code = EXIT_ERROR

    def __init__(self, message: str, exit_code: int | None = None):
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class CLIFileNotFoundError(CLIError):
    """File not found error."""

    exit_code = EXIT_FILE_NOT_FOUND


class DeviceNotFoundError(CLIError):
    """Device not found error."""

    exit_code = EXIT_DEVICE_NOT_FOUND


def error(message: str, exit_code: int = EXIT_ERROR) -> NoReturn:
    """Print error message to stderr and exit."""
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(exit_code)


def warning(message: str) -> None:
    """Print warning message to stderr."""
    print(f"Warning: {message}", file=sys.stderr)


def require_file(path: str) -> Path:
    """Validate file exists, return Path object."""
    p = Path(path)
    if not p.exists():
        raise CLIFileNotFoundError(f"File not found: {path}")
    return p


def require_numpy() -> None:
    """Raise error if NumPy is not available."""
    try:
        import numpy  # noqa: F401
    except ImportError:
        raise CLIError("This command requires NumPy. Install with: pip install numpy")


def require_scipy() -> None:
    """Raise error if SciPy is not available."""
    try:
        import scipy  # noqa: F401
    except ImportError:
        raise CLIError("This command requires SciPy. Install with: pip install scipy")
