#!/usr/bin/env python3
"""Tests for the MIDI CLI commands."""

import argparse
import time

import pytest

from coremusic.cli import midi as cli_midi
from coremusic.cli._utils import CLIError


def _send_args(**kw):
    return argparse.Namespace(
        dest=kw.get("dest", 0),
        test=kw.get("test", False),
        note=kw.get("note"),
        velocity=kw.get("velocity", 100),
        channel=kw.get("channel", 0),
        cc=kw.get("cc"),
        program=kw.get("program"),
        duration=kw.get("duration", 0.0),
        json=False,
    )


@pytest.fixture
def fake_midi(monkeypatch):
    """Stand in for CoreMIDI, recording every message sent."""
    sent = []

    monkeypatch.setattr(cli_midi.capi, "midi_get_number_of_destinations", lambda: 1)
    monkeypatch.setattr(cli_midi.capi, "midi_get_destination", lambda i: 100)
    monkeypatch.setattr(cli_midi, "_get_endpoint_name", lambda dest_id: "Fake Synth")
    monkeypatch.setattr(cli_midi.capi, "midi_client_create", lambda name: 1)
    monkeypatch.setattr(cli_midi.capi, "midi_output_port_create", lambda c, n: 2)
    monkeypatch.setattr(cli_midi.capi, "midi_client_dispose", lambda c: None)
    monkeypatch.setattr(
        cli_midi.capi,
        "midi_send_data",
        lambda port, dest, data, ts: sent.append(bytes(data)),
    )
    return sent


class TestCmdSendNoteRelease:
    """A Note On must always be followed by its Note Off."""

    def test_note_off_follows_note_on(self, fake_midi):
        cli_midi.cmd_send(_send_args(note=60))

        assert [data[0] & 0xF0 for data in fake_midi] == [0x90, 0x80]

    def test_interrupt_during_the_wait_still_releases(self, fake_midi, monkeypatch):
        def interrupted(_duration):
            raise KeyboardInterrupt

        monkeypatch.setattr(time, "sleep", interrupted)

        with pytest.raises(KeyboardInterrupt):
            cli_midi.cmd_send(_send_args(note=60, duration=5.0))

        assert [data[0] & 0xF0 for data in fake_midi] == [0x90, 0x80]
        assert fake_midi[1][1] == 60

    def test_negative_duration_sends_nothing(self, fake_midi):
        """Rejected before the Note On, so no note can be left sounding."""
        with pytest.raises(CLIError, match="Duration"):
            cli_midi.cmd_send(_send_args(note=60, duration=-1.0))

        assert fake_midi == []
