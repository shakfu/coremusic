#!/usr/bin/env python3
"""Tests for the device CLI commands."""

import argparse
import types

import pytest

from coremusic.audio import AudioDeviceManager
from coremusic.cli import devices as cli_devices
from coremusic.exceptions import AudioDeviceError


def _set_default_args(**kw):
    return argparse.Namespace(
        device=kw.get("device", "Fake Device"),
        input_device=kw.get("input_device", False),
        output_device=kw.get("output_device", False),
        json=kw.get("json", False),
    )


@pytest.fixture
def fake_device(monkeypatch):
    device = types.SimpleNamespace(name="Fake Device", uid="fake-uid")
    monkeypatch.setattr(cli_devices, "_find_device", lambda name: device)
    return device


class TestCmdSetDefaultExitStatus:
    """The exit status must reflect whether the device actually changed."""

    def test_success_exits_zero(self, fake_device, monkeypatch):
        monkeypatch.setattr(
            AudioDeviceManager, "set_default_output_device", staticmethod(lambda d: None)
        )

        assert cli_devices.cmd_set_default(_set_default_args()) == 0

    def test_failure_exits_nonzero(self, fake_device, monkeypatch):
        def refuse(_device):
            raise AudioDeviceError("device is in use")

        monkeypatch.setattr(
            AudioDeviceManager, "set_default_output_device", staticmethod(refuse)
        )

        assert cli_devices.cmd_set_default(_set_default_args()) != 0

    def test_partial_failure_exits_nonzero(self, fake_device, monkeypatch):
        def refuse(_device):
            raise AudioDeviceError("no input path")

        monkeypatch.setattr(
            AudioDeviceManager, "set_default_output_device", staticmethod(lambda d: None)
        )
        monkeypatch.setattr(
            AudioDeviceManager, "set_default_input_device", staticmethod(refuse)
        )

        args = _set_default_args(output_device=True, input_device=True)
        assert cli_devices.cmd_set_default(args) != 0
