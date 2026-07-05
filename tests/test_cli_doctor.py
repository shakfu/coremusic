#!/usr/bin/env python3
"""Tests for the `coremusic doctor` diagnostics command."""

import argparse

from coremusic.cli import doctor


def test_gather_report_structure():
    report = doctor._gather_report()
    assert set(report) >= {
        "coremusic_version",
        "python_version",
        "platform",
        "is_macos",
        "optional_dependencies",
        "audio",
        "plugins",
        "midi",
    }
    assert isinstance(report["optional_dependencies"], dict)
    assert set(report["optional_dependencies"]) == {"numpy", "scipy", "matplotlib"}


def test_check_sections_report_accessible_flag():
    for section in (doctor._check_audio(), doctor._check_plugins(), doctor._check_midi()):
        assert "accessible" in section


def test_cmd_doctor_text(capsys):
    args = argparse.Namespace(json=False)
    assert doctor.cmd_doctor(args) == 0
    out = capsys.readouterr().out
    assert "coremusic doctor" in out
    assert "Optional dependencies:" in out


def test_cmd_doctor_json(capsys):
    import json

    args = argparse.Namespace(json=True)
    assert doctor.cmd_doctor(args) == 0
    data = json.loads(capsys.readouterr().out)
    assert "coremusic_version" in data
    assert "audio" in data
