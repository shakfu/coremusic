#!/usr/bin/env python3
"""Tests that CoreAudio handles are released when objects are collected.

`CoreAudioObject` documents automatic resource management. That promise was
previously false: `__dealloc__` could only reach the `cdef` `_dispose_internal`,
which a Python subclass cannot override, so every subclass `dispose()` was
skipped at collection time and each handle leaked. Release now happens in
`__del__` (tp_finalize), which does normal method lookup.

The subclass-dispatch test is the precise regression guard. The descriptor-count
test is the end-to-end evidence that it translates into handles actually being
returned to the OS.
"""

from __future__ import annotations

import gc
import os
import subprocess

import pytest

from coremusic import capi
from coremusic.audio import AudioFile


def _open_descriptor_count() -> int:
    """Count lsof rows for this process.

    Absolute value includes non-descriptor rows; only differences are meaningful.
    """
    result = subprocess.run(
        ["lsof", "-p", str(os.getpid())],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return len(result.stdout.splitlines())


def test_subclass_dispose_runs_on_collection():
    """A Python subclass's dispose() override must run when the object is collected.

    This is the exact defect: __dealloc__ calls a cdef method that Python code
    cannot override, so this callback never fired and subclasses never released
    their handles.
    """
    calls = []

    class Tracked(capi.CoreAudioObject):
        def dispose(self) -> None:
            calls.append("disposed")
            super().dispose()

    obj = Tracked()
    del obj
    gc.collect()

    assert calls == ["disposed"], (
        "subclass dispose() was not called on collection; CoreAudio handles "
        "held by subclasses will leak"
    )


def test_explicit_dispose_is_not_repeated_on_collection():
    """Disposing explicitly must not cause a second dispose at finalization."""
    calls = []

    class Tracked(capi.CoreAudioObject):
        def dispose(self) -> None:
            if not self.is_disposed:
                calls.append("disposed")
            super().dispose()

    obj = Tracked()
    obj.dispose()
    assert calls == ["disposed"]

    del obj
    gc.collect()
    assert calls == ["disposed"], "dispose() ran twice"


@pytest.mark.skipif(not os.path.exists("/usr/sbin/lsof"), reason="lsof not available")
def test_unclosed_audio_files_do_not_leak_descriptors(amen_wav_path):
    """Opened-then-dropped AudioFiles must not accumulate descriptors.

    Before the fix this leaked exactly one descriptor per iteration.
    """
    count = 50

    gc.collect()
    before = _open_descriptor_count()

    for _ in range(count):
        audio_file = AudioFile(amen_wav_path)
        audio_file.open()
        del audio_file
    gc.collect()

    after = _open_descriptor_count()

    # A small drift is possible from unrelated runtime activity; a real leak
    # would show up as roughly `count` extra rows.
    assert after - before < count // 2, (
        f"{after - before} descriptors leaked across {count} unclosed AudioFile "
        "opens; expected close to zero"
    )


def test_context_manager_still_releases(amen_wav_path):
    """The explicit path must keep working; it is still the recommended one."""
    with AudioFile(amen_wav_path) as audio_file:
        assert audio_file.object_id != 0
        assert not audio_file.is_disposed
    assert audio_file.is_disposed
