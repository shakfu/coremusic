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


def output_table(
    headers: list[str],
    rows: list[list[str]],
    max_widths: list[int] | None = None,
) -> None:
    """Output data as aligned table.

    Args:
        headers: Column header names.
        rows: List of rows, each row is a list of cell values.
        max_widths: Optional maximum width for each column. Use 0 for no limit.
    """
    if not rows:
        return

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Apply max_widths if provided
    if max_widths:
        for i, max_w in enumerate(max_widths):
            if max_w > 0 and i < len(widths):
                widths[i] = min(widths[i], max_w)

    def truncate(text: str, width: int) -> str:
        """Truncate text to width, adding ellipsis if needed."""
        text = str(text)
        if len(text) <= width:
            return text.ljust(width)
        return text[: width - 1] + "~"

    # Print header (last column not padded)
    parts = [truncate(h, w) for h, w in zip(headers[:-1], widths[:-1])]
    parts.append(headers[-1])
    header_line = "  ".join(parts)
    print(header_line)
    print("-" * len(header_line))

    # Print rows (last column not padded)
    for row in rows:
        parts = [truncate(c, w) for c, w in zip(row[:-1], widths[:-1])]
        parts.append(str(row[-1]) if len(row) > len(widths) - 1 else str(row[-1]))
        print("  ".join(parts))
