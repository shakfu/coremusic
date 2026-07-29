#!/usr/bin/env python3
"""Guards on which exceptions the library swallows.

A handler that catches `Exception` and returns a fallback value cannot tell a
CoreAudio refusal from a bug in this library, so it converts `AttributeError`,
`TypeError` and `NameError` into a plausible-looking answer the caller cannot
question. That is how `AudioFileStream.ready_to_produce_packets` reported "not
ready" for every stream in every state while calling a function that did not
exist.

101 such handlers were narrowed to `FRAMEWORK_ERRORS`. The 13 that remain broad
are legitimate and enumerated in `ALLOWED_BROAD` below; these tests fail if that
set grows, or if the meaning of `FRAMEWORK_ERRORS` drifts.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import coremusic
from coremusic.exceptions import FRAMEWORK_ERRORS, CoreAudioError

SRC = pathlib.Path(coremusic.__file__).parent


# ============================================================================
# What FRAMEWORK_ERRORS means
# ============================================================================


def test_framework_errors_covers_what_the_layers_below_raise():
    """capi signals a non-zero OSStatus with RuntimeError; wrappers use CoreAudioError."""
    assert issubclass(CoreAudioError, Exception)
    for exc in (CoreAudioError("x"), RuntimeError("x"), OSError("x")):
        assert isinstance(exc, FRAMEWORK_ERRORS), f"{type(exc).__name__} not covered"

    # Subclasses of CoreAudioError must be covered too.
    from coremusic.exceptions import AudioFileError, MIDIError

    assert isinstance(AudioFileError("x"), FRAMEWORK_ERRORS)
    assert isinstance(MIDIError("x"), FRAMEWORK_ERRORS)


def test_framework_errors_excludes_bug_signalling_types():
    """The whole point: a defect must not be absorbed as a refused operation.

    ValueError is excluded deliberately -- capi raises it for an invalid
    argument, which means the caller got it wrong.
    """
    for exc in (
        AttributeError("x"),
        TypeError("x"),
        NameError("x"),
        IndexError("x"),
        KeyError("x"),
        ValueError("x"),
        MemoryError(),
    ):
        assert not isinstance(exc, FRAMEWORK_ERRORS), (
            f"{type(exc).__name__} is swallowed by FRAMEWORK_ERRORS; a bug of "
            "this kind would be silently converted into a fallback value"
        )


# ============================================================================
# Behaviour at a real call site
# ============================================================================


def test_programming_error_propagates_through_a_guarded_call(
    amen_wav_path, monkeypatch
):
    """A bug behind a guarded call must reach the caller, not become a default.

    `AudioFile.metadata` returns None when CoreAudio has no info dictionary.
    Before narrowing, an AttributeError from the capi layer produced that same
    None and was indistinguishable from "this file has no metadata".
    """
    from coremusic import capi
    from coremusic.audio import AudioFile

    def boom(*args, **kwargs):
        raise AttributeError("simulated typo in the capi call")

    monkeypatch.setattr(capi, "audio_file_read_info_dictionary", boom)

    with AudioFile(amen_wav_path) as audio_file, pytest.raises(AttributeError):
        _ = audio_file.metadata


def test_framework_error_is_still_absorbed(amen_wav_path, monkeypatch):
    """The complement: a genuine CoreAudio refusal still yields the fallback."""
    from coremusic import capi
    from coremusic.audio import AudioFile

    def refuse(*args, **kwargs):
        raise RuntimeError("AudioFileGetProperty failed: kAudioFileUnsupportedProperty")

    monkeypatch.setattr(capi, "audio_file_read_info_dictionary", refuse)

    with AudioFile(amen_wav_path) as audio_file:
        assert audio_file.metadata is None


# ============================================================================
# The allowlist
# ============================================================================

# (module path relative to the package, enclosing function) -> why it stays broad.
# Keyed on the enclosing function rather than a line number so ordinary edits
# do not churn this table.
ALLOWED_BROAD = {
    (
        "audio/streaming.py",
        "_drain_loop",
    ): "invokes caller-supplied subscriber callbacks",
    ("audio/streaming.py", "_enqueue_output"): "invokes caller-supplied process_func",
    ("audio/streaming.py", "_generate_output"): "invokes caller-supplied process_func",
    ("audio/streaming.py", "process"): "invokes caller-supplied processor",
    (
        "utils/batch.py",
        "_process_item_with_retry",
    ): "capturing per-item failure is the contract",
    (
        "utils/batch.py",
        "_process_parallel",
    ): "caller work function surfacing through the executor",
    ("midi/link.py", "_clock_thread"): "worker-thread loop must not die",
    ("midi/link.py", "_sequencer_thread"): "worker-thread loop must not die",
    ("cli/doctor.py", "_check_audio"): "doctor reports any failure",
    ("cli/doctor.py", "_check_plugins"): "doctor reports any failure",
    ("cli/doctor.py", "_check_midi"): "doctor reports any failure",
}


def _broad_swallowing_handlers() -> set[tuple[str, str]]:
    """Every `except Exception` (or bare except) that does not re-raise."""
    found = set()
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        # map each node to its enclosing function
        enclosing: dict[ast.AST, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                for child in ast.walk(node):
                    enclosing.setdefault(child, node.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            exc = node.type
            # A tuple counts as broad when Exception is one of its members:
            # `except (AudioDeviceError, Exception)` catches everything, and
            # reads as if it does not. Two such handlers hid from an earlier
            # version of this check that only looked at the bare-name form.
            broad = (
                exc is None
                or (
                    isinstance(exc, ast.Name)
                    and exc.id in ("Exception", "BaseException")
                )
                or (
                    isinstance(exc, ast.Tuple)
                    and any(
                        isinstance(e, ast.Name)
                        and e.id in ("Exception", "BaseException")
                        for e in exc.elts
                    )
                )
            )
            if not broad:
                continue
            if any(isinstance(n, ast.Raise) for n in ast.walk(node)):
                continue  # re-raises: the error still reaches the caller
            rel = path.relative_to(SRC).as_posix()
            found.add((rel, enclosing.get(node, "<module>")))
    return found


def test_no_undocumented_broad_swallowing_handlers():
    """Every broad handler that swallows must be listed, with a reason, above."""
    actual = _broad_swallowing_handlers()
    undocumented = sorted(actual - set(ALLOWED_BROAD))
    assert not undocumented, (
        "broad `except Exception` that swallows, outside the allowlist:\n"
        + "\n".join(f"  {f}::{fn}" for f, fn in undocumented)
        + "\n\nCatch coremusic.exceptions.FRAMEWORK_ERRORS instead, so a bug "
        "propagates. If the broad catch is genuinely right, add it to "
        "ALLOWED_BROAD with a reason."
    )


def test_allowlist_has_no_stale_entries():
    """A listed exemption that no longer exists should be removed, not left to rot."""
    actual = _broad_swallowing_handlers()
    stale = sorted(set(ALLOWED_BROAD) - actual)
    assert not stale, f"ALLOWED_BROAD lists handlers that no longer exist: {stale}"
