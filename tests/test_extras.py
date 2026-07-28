"""Run the standalone utilities in extras/.

The experimental modules there (`daw/`, `generative/`) have their own test
modules alongside this one. The two command-line utilities had nothing, and
`audio_converter.py` had gone stale unnoticed: it called
`capi.convert_audio_file`, which does not exist - the helper lives in
`coremusic.audio`. Running them is cheap, so there is no reason not to.
"""

import os
import subprocess
import sys

import pytest
from conftest import AMEN_WAV_PATH

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRAS_DIR = os.path.join(REPO_ROOT, "extras")


def run_utility(*args, cwd=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO_ROOT, "src") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd or REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture
def sample_wav():
    if not os.path.exists(AMEN_WAV_PATH):
        pytest.skip(f"Test audio file not found: {AMEN_WAV_PATH}")
    return AMEN_WAV_PATH


def test_audio_inspector(sample_wav):
    result = run_utility(os.path.join(EXTRAS_DIR, "audio_inspector.py"), sample_wav)
    assert result.returncode == 0, result.stderr
    assert "Sample Rate" in result.stdout


def test_audio_inspector_reports_missing_file():
    result = run_utility(os.path.join(EXTRAS_DIR, "audio_inspector.py"), "nope.wav")
    assert result.returncode != 0
    assert "not found" in (result.stdout + result.stderr).lower()


def test_audio_converter(sample_wav, tmp_path):
    out = tmp_path / "converted.wav"
    result = run_utility(
        os.path.join(EXTRAS_DIR, "audio_converter.py"),
        sample_wav,
        str(out),
        "--rate",
        "48000",
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()

    from coremusic.audio import AudioFile

    with AudioFile(str(out)) as f:
        assert f.format.sample_rate == 48000.0
