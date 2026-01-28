"""Pytest configuration for tutorial doctests."""
from __future__ import annotations

import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "doctest: mark test as a doctest"
    )


@pytest.fixture(autouse=True)
def add_doctest_namespace(doctest_namespace):
    """Add common imports to doctest namespace."""
    import coremusic as cm
    doctest_namespace["cm"] = cm
