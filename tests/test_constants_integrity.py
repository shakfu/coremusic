#!/usr/bin/env python3
"""Integrity checks for the `coremusic.constants` package.

The enum classes in `constants/` restate values that the C compiler already
knows. Restated facts drift, so these tests reconcile them against two
independent sources of truth:

1. The compiled `coremusic.capi` getters, which resolve the C macros at build
   time. Always available.
2. The macOS SDK headers themselves, via a generated C probe. Skipped when no
   compiler or SDK is present.

A third check verifies that every constant's value round-trips the FourCC in
its trailing comment, which catches transcription typos without needing either
source.

Historical note: 52 constants in this package once disagreed with the SDK, and
one enum class described a CoreMIDI API that does not exist. None of it was
visible to the rest of the suite, because nothing compared the two layers.
"""

from __future__ import annotations

import ast
import enum
import pathlib
import re
import shutil
import subprocess
import tempfile
import tokenize

import pytest

import coremusic
from coremusic import capi, constants

CONSTANTS_DIR = pathlib.Path(constants.__file__).parent


# ============================================================================
# Source parsing
# ============================================================================


def _parse_annotated_members() -> list[tuple[str, str, str, int, str | None]]:
    """Extract (module, class, member, value, c_name) from the constants source.

    Uses tokenize rather than a line regex so that values wrapped across lines
    by the formatter are still matched to their trailing comment.
    """
    found = []
    for path in sorted(CONSTANTS_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        source = path.read_text()
        tree = ast.parse(source)

        # member -> line number of its assignment
        member_lines: dict[tuple[str, str], int] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    if isinstance(target, ast.Name):
                        member_lines[(node.name, target.id)] = (
                            stmt.end_lineno or stmt.lineno
                        )

        # line number -> C constant name mentioned in that line's comment
        comments: dict[int, str] = {}
        with open(path, "rb") as handle:
            for tok in tokenize.tokenize(handle.readline):
                if tok.type == tokenize.COMMENT:
                    match = re.search(r"#\s*(k[A-Za-z0-9_]+)", tok.string)
                    if match:
                        comments[tok.start[0]] = match.group(1)

        for (cls_name, member), lineno in member_lines.items():
            cls = getattr(constants, cls_name, None)
            if cls is None or not (
                isinstance(cls, type) and issubclass(cls, enum.Enum)
            ):
                continue
            value = int(cls[member].value)
            found.append((path.name, cls_name, member, value, comments.get(lineno)))
    return found


ANNOTATED = _parse_annotated_members()


def test_source_parsing_found_members():
    """Guard the parser itself: a silent parse failure would void every test here."""
    assert len(ANNOTATED) > 200, f"only parsed {len(ANNOTATED)} members"
    assert sum(1 for row in ANNOTATED if row[4]) > 180, "C names not being picked up"


# ============================================================================
# Check 1: FourCC comments round-trip (no external dependency)
# ============================================================================


def _as_fourcc(value: int) -> str | None:
    if not 0 < value <= 0xFFFFFFFF:
        return None
    raw = value.to_bytes(4, "big")
    return raw.decode("ascii") if all(32 <= b < 127 for b in raw) else None


def test_fourcc_comments_match_values():
    """Every ``# kName ('abcd')`` comment must agree with the integer beside it."""
    mismatches = []
    for path in sorted(CONSTANTS_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            match = re.search(r"(\d+)\s*#.*?\('(.{4})'\)", line)
            if not match:
                continue
            value, documented = int(match.group(1)), match.group(2)
            if _as_fourcc(value) != documented:
                mismatches.append(
                    f"{path.name}:{lineno} {value} decodes "
                    f"{_as_fourcc(value)!r}, comment says {documented!r}"
                )
    assert not mismatches, "FourCC comment/value disagreement:\n" + "\n".join(
        mismatches
    )


# ============================================================================
# Check 2: cross-check against the compiled capi getters
# ============================================================================


def _getter_values() -> dict[str, int]:
    values = {}
    for name in dir(capi):
        if not name.startswith("get_"):
            continue
        fn = getattr(capi, name)
        if not callable(fn):
            continue
        try:
            result = fn()
        except Exception:  # noqa: BLE001 - getter needs arguments; not a constant
            continue
        if isinstance(result, int):
            values[name] = result
    return values


GETTERS = _getter_values()


def test_enum_values_match_capi_getters():
    """Where an enum member and a capi getter name the same constant, they must agree.

    The getters are resolved by the C compiler at build time, so any
    disagreement means the hand-written enum is wrong.
    """
    mismatches = []
    checked = 0
    for _module, cls_name, member, value, _c_name in ANNOTATED:
        prefix = re.sub(r"(?<!^)(?=[A-Z])", "_", cls_name).lower()
        candidate = f"get_{prefix}_{member.lower()}"
        if candidate not in GETTERS:
            continue
        checked += 1
        if GETTERS[candidate] != value:
            mismatches.append(
                f"{cls_name}.{member} = {value} but {candidate}() = {GETTERS[candidate]}"
            )
    assert checked >= 30, f"name heuristic only matched {checked} pairs; did it break?"
    assert not mismatches, "enum/getter disagreement:\n" + "\n".join(mismatches)


# ============================================================================
# Check 3: full verification against the macOS SDK
# ============================================================================

_CLANG = shutil.which("clang")


def _sdk_path() -> str | None:
    try:
        result = subprocess.run(
            ["xcrun", "--show-sdk-path"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    path = result.stdout.strip()
    return path if result.returncode == 0 and path else None


@pytest.mark.slow
@pytest.mark.skipif(_CLANG is None, reason="clang not available")
def test_every_annotated_constant_matches_the_sdk():
    """Compile a probe that prints each documented C constant, and compare.

    This is the check that would have caught the original 52 wrong values. It
    needs a compiler and the macOS SDK, so it is marked slow and skipped where
    those are absent; checks 1 and 2 still run everywhere.
    """
    if _sdk_path() is None:
        pytest.skip("macOS SDK not available")

    named = [(cls, member, value, c) for _m, cls, member, value, c in ANNOTATED if c]
    assert named, "no constants carry a C name comment"

    c_names = sorted({c for _cls, _m, _v, c in named})
    lines = [
        "#include <AudioToolbox/AudioToolbox.h>",
        "#include <CoreAudio/CoreAudio.h>",
        "#include <CoreAudioTypes/CoreAudioTypes.h>",
        "#include <stdio.h>",
        "",
        "int main(void) {",
    ]
    lines += [f'    printf("{n} %lld\\n", (long long)({n}));' for n in c_names]
    lines += ["    return 0;", "}"]

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        src, exe = tmpdir / "probe.c", tmpdir / "probe"
        src.write_text("\n".join(lines) + "\n")
        compile_result = subprocess.run(
            [
                _CLANG,
                "-o",
                str(exe),
                str(src),
                "-framework",
                "AudioToolbox",
                "-framework",
                "CoreAudio",
                "-framework",
                "CoreFoundation",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert compile_result.returncode == 0, (
            "probe failed to compile -- a constant below names no SDK symbol:\n"
            + compile_result.stderr
        )
        run_result = subprocess.run(
            [str(exe)], capture_output=True, text=True, timeout=60
        )
        assert run_result.returncode == 0, run_result.stderr

    truth = {}
    for line in run_result.stdout.splitlines():
        name, raw = line.split()
        truth[name] = int(raw)

    mismatches = [
        f"{cls}.{member} = {value} but {c_name} = {truth[c_name]}"
        for cls, member, value, c_name in named
        if truth.get(c_name) != value
    ]
    assert not mismatches, (
        f"{len(mismatches)} of {len(named)} constants disagree with the SDK:\n"
        + "\n".join(mismatches)
    )


# ============================================================================
# Check 4: the extension and its type stub declare the same names
# ============================================================================


def test_capi_stub_matches_extension():
    """`capi.pyi` must not promise names the compiled module lacks.

    mypy trusts the stub, so a name declared here but missing from the
    extension is an AttributeError that static checking cannot see.
    """
    # In a wheel install the stub sits beside the .so; in this repo the
    # compiled module is installed but the Python package resolves to src/.
    candidates = [
        pathlib.Path(capi.__file__).parent / "capi.pyi",
        pathlib.Path(coremusic.__file__).parent / "capi.pyi",
    ]
    stub = next((p for p in candidates if p.exists()), None)
    assert stub is not None, f"capi.pyi not found in any of {candidates}"

    text = stub.read_text()
    declared = set(re.findall(r"^def (\w+)", text, re.M))
    declared |= set(re.findall(r"^class (\w+)", text, re.M))
    actual = {n for n in dir(capi) if not n.startswith("__")}

    phantom = sorted(n for n in declared - actual if not n.startswith("_"))
    assert not phantom, (
        "capi.pyi declares names the extension does not export "
        f"(calls to these raise AttributeError at runtime): {phantom}"
    )
