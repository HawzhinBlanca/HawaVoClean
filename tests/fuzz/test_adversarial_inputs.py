"""Adversarial-input regression gate.

Every generated input shape must produce one of exactly three outcomes:
  - a valid output (right length, finite, under the ceiling, with a report),
  - a CLEAN rejection (non-zero exit, no traceback), for inputs that must
    or may be rejected,
  - never a hang, never an unhandled traceback, never a silent bad output.

This gate was added after fuzzing found three crashes the unit/mutation
suites had not modeled (11025 Hz limiter parity, 1-sample input,
video-first MP4 containers). Run: pytest tests/fuzz -m fuzz
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

HERE = Path(__file__).parent
CEILING = 10 ** (-1.0 / 20.0)
TIMEOUT_S = 300

MUST_REJECT = {
    "rate_96k_must_reject",
    "zero_byte",
    "truncated_header",
    "text_not_audio",
    "video_no_audio",
    "empty_0samples",
    "six_channel",
    "nan_in_file",
    "inf_in_file",
}
MAY_REJECT = {
    "stereo_inverted_polarity",
    "stereo_one_silent",
    "stereo_tiny_level_diff",
    "truncated_data",
    "one_sample",
}
# Slow / large cases excluded from the default gate; run manually.
SKIP_BY_DEFAULT = {"long_8min"}


@pytest.fixture(scope="session")
def fuzz_inputs(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    out = tmp_path_factory.mktemp("fuzz-inputs")
    env = {"FUZZ_OUT_DIR": str(out)}
    import os

    subprocess.run(
        [sys.executable, str(HERE / "gen_inputs.py")],
        check=True,
        env={**os.environ, **env},
        capture_output=True,
        timeout=600,
    )
    return sorted(p for p in out.iterdir() if p.stem not in SKIP_BY_DEFAULT)


def _case_ids() -> list[str]:
    # Discover names without generating audio (cheap parse of gen_inputs.py)
    src = (HERE / "gen_inputs.py").read_text()
    import re

    names = set(re.findall(r'cases\["([^"]+)"\]', src))
    names |= {
        "mp3_lossy",
        "aac_m4a",
        "pcm_24bit",
        "pcm_8bit_u8",
        "flac",
        "zero_byte",
        "truncated_header",
        "truncated_data",
        "text_not_audio",
        "name with spaces & quote's",
        "video_with_audio",
        "video_no_audio",
    }
    names |= {f"rate_{r}" for r in (8000, 11025, 16000, 22050, 44100)}
    return sorted(n for n in names if n not in SKIP_BY_DEFAULT)


@pytest.mark.fuzz
@pytest.mark.parametrize("profile", ["production"])
@pytest.mark.parametrize("case", _case_ids())
def test_adversarial_input(
    case: str, profile: str, fuzz_inputs: list[Path], tmp_path: Path
) -> None:
    matches = [p for p in fuzz_inputs if p.stem == case]
    if not matches:
        pytest.skip(f"{case}: encoder unavailable on this host")
    src = matches[0]
    dest = tmp_path / f"{case}.wav"
    cli = shutil.which("hawavoclean") or str(Path(sys.executable).with_name("hawavoclean"))

    try:
        proc = subprocess.run(
            [cli, "process", str(src), "-o", str(dest), "--profile", profile, "--overwrite"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{case}: HANG (> {TIMEOUT_S}s)")

    err = proc.stderr + proc.stdout
    assert "Traceback" not in err, f"{case}: unhandled traceback:\n{err[-800:]}"

    if proc.returncode != 0:
        assert case in MUST_REJECT or case in MAY_REJECT, (
            f"{case}: rejected (exit {proc.returncode}) but should have processed:\n{err[-400:]}"
        )
        return

    assert case not in MUST_REJECT, f"{case}: accepted input that must be rejected"
    y, sr_y = sf.read(str(dest), dtype="float32", always_2d=True)
    assert np.all(np.isfinite(y)), f"{case}: NaN/Inf in output"
    assert float(np.max(np.abs(y))) <= CEILING + 1e-4, f"{case}: output over the -1 dBTP ceiling"
    assert (tmp_path / f"{case}.hawavoclean.json").exists(), f"{case}: no report"
    try:
        info = sf.info(str(src))
        if info.subtype not in ("", None) and src.suffix.lower() in (".wav", ".flac"):
            assert info.frames == y.shape[0], f"{case}: length {y.shape[0]} != input {info.frames}"
            assert info.samplerate == sr_y, f"{case}: sample rate changed"
    except RuntimeError:
        pass  # lossy/odd containers: soundfile cannot read them; pipeline re-syncs
