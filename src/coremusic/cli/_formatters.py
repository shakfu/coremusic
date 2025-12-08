"""Output formatters for CLI commands."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any


def format_duration(seconds: float) -> str:
    """Format duration as MM:SS.mmm."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string."""
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def format_sample_rate(rate: float) -> str:
    """Format sample rate with units."""
    if rate >= 1000:
        return f"{rate / 1000:.1f} kHz"
    return f"{rate:.0f} Hz"


def format_db(linear: float) -> str:
    """Format linear amplitude as dB."""
    import math

    if linear <= 0:
        return "-inf dB"
    db = 20 * math.log10(linear)
    return f"{db:.1f} dB"


def output_json(data: Any) -> None:
    """Output data as JSON to stdout."""
    if is_dataclass(data) and not isinstance(data, type):
        data = asdict(data)
    print(json.dumps(data, indent=2, default=str))


def output_table(headers: list[str], rows: list[list[str]]) -> None:
    """Output data as aligned table."""
    if not rows:
        return

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Print header
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for row in rows:
        line = "  ".join(str(c).ljust(w) for c, w in zip(row, widths))
        print(line)
