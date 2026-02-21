"""Pytest configuration for tutorial doctests."""

from __future__ import annotations


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "doctest: mark test as a doctest")
