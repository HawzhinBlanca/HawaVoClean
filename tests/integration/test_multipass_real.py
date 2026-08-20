"""Multi-pass evidence on the real lab files (gitignored; skip when absent).

The measured recipe this feature ships on (production profile, teat1vo
``src.mp3``): pass 1 clears the guard only at strength 0.50 and its tonal
restoration lifts the presence band; that restored output lets pass 2 run at
strength 1.00, deepening speech/floor separation to ~23.6 dB. A third pass
converges (separation slightly down), so auto mode stops at 2 or 3 with the
regressing pass discarded. On an already-good source (Flute 09) the second
pass changes little and auto ships the approved single-pass result.

Synthetic equivalents of every logic branch live in
``tests/unit/test_multipass.py`` so CI does not depend on these files.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from hawavoclean.hashing import hash_file
from hawavoclean.report.writer import load_json_report
from tests.support.wavbytes import masked_wav_bytes

REPO = Path(__file__).resolve().parents[2]
TEAT1VO = REPO / "test_output" / "teat1vo-lab" / "src.mp3"
FLUTE = REPO / "test_output" / "ui-smoke" / "Flute 09.m4a.mp4"

pytestmark = pytest.mark.integration


def _run_cli(*argv: str, timeout: float = 900.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hawavoclean.cli", *argv],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=REPO,
    )


@pytest.mark.skipif(not TEAT1VO.exists(), reason="gitignored lab file not present")
def test_teat1vo_two_passes_reproduce_measured_separation(tmp_path: Path) -> None:
    out = tmp_path / "two.wav"
    proc = _run_cli("process", str(TEAT1VO), "-o", str(out), "--passes", "2")
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = load_json_report(out.parent / f"{out.stem}.hawavoclean.json")

    assert [p.pass_index for p in report.passes] == [1, 2]
    assert not any(p.discarded for p in report.passes)
    # The measured recipe: pass 1 clears the guard only at 0.50, and its
    # restored output lets pass 2 run at full strength.
    assert report.passes[0].chosen_strengths == [0.5]
    assert report.passes[1].chosen_strengths == [1.0]
    assert report.passes[1].separation_db == pytest.approx(23.6, abs=0.5)
    assert report.passes[1].separation_db > report.passes[0].separation_db + 0.5
    # Provenance chain: source -> pass 1 -> pass 2 -> shipped output.
    assert report.passes[0].input_sha256 == report.input.sha256
    assert report.passes[1].input_sha256 == report.passes[0].output_sha256
    assert report.passes[1].output_sha256 == report.output.sha256 == hash_file(out)
    # The published pair verifies like any single-pass output.
    v = _run_cli("verify", str(out), "-r", str(out.parent / f"{out.stem}.hawavoclean.json"))
    assert v.returncode == 0, v.stderr


@pytest.mark.skipif(not TEAT1VO.exists(), reason="gitignored lab file not present")
def test_teat1vo_auto_stops_and_records_any_discard(tmp_path: Path) -> None:
    out = tmp_path / "auto.wav"
    proc = _run_cli("process", str(TEAT1VO), "-o", str(out), "--passes", "auto")
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = load_json_report(out.parent / f"{out.stem}.hawavoclean.json")

    kept = [p for p in report.passes if not p.discarded]
    assert 2 <= len(kept) <= 3, f"auto should settle at 2-3 passes here, got {len(kept)}"
    assert report.output.sha256 == kept[-1].output_sha256 == hash_file(out)
    if report.passes[-1].discarded:
        # The regressing pass is on record, with its reason, and did not ship.
        assert report.passes[-1].discard_reason
        assert report.passes[-1].output_sha256 != report.output.sha256
    # Every kept pass improved on its predecessor by the auto criteria.
    for prev, new in zip(kept, kept[1:], strict=False):
        assert new.enhanced >= prev.enhanced
        assert new.separation_db >= prev.separation_db + 0.5


@pytest.mark.skipif(not TEAT1VO.exists(), reason="gitignored lab file not present")
def test_teat1vo_single_pass_is_byte_identical_to_plain_run(tmp_path: Path) -> None:
    plain = tmp_path / "plain.wav"
    single = tmp_path / "single.wav"
    p1 = _run_cli("process", str(TEAT1VO), "-o", str(plain))
    assert p1.returncode == 0, p1.stderr[-2000:]
    p2 = _run_cli("process", str(TEAT1VO), "-o", str(single), "--passes", "1")
    assert p2.returncode == 0, p2.stderr[-2000:]
    # Byte-identical modulo libsndfile's PEAK write-time second (see
    # tests/support/wavbytes.py).
    assert masked_wav_bytes(plain.read_bytes()) == masked_wav_bytes(single.read_bytes())
    report = load_json_report(single.parent / f"{single.stem}.hawavoclean.json")
    assert report.passes == []


@pytest.mark.skipif(not FLUTE.exists(), reason="gitignored lab file not present")
def test_flute_auto_terminates_and_ships_approved_result(tmp_path: Path) -> None:
    out = tmp_path / "flute.wav"
    # Termination is part of the assertion: auto on an already-good source
    # must stand itself down early, not grind to the 4-pass cap. (Measured
    # 2026-08-20: pass 2 clears both criteria here, pass 3 is discarded for
    # guard regression — 4/5 enhanced vs pass 2's 6 — so auto stops at 3
    # records, shipping pass 2.)
    proc = _run_cli("process", str(FLUTE), "-o", str(out), "--passes", "auto", timeout=900.0)
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = load_json_report(out.parent / f"{out.stem}.hawavoclean.json")
    assert len(report.passes) <= 3, "an already-good source must stand auto down early"

    # A discarded pass can only ever be the last record, and never ships.
    discarded = [p for p in report.passes if p.discarded]
    assert len(discarded) <= 1
    if discarded:
        assert report.passes[-1].discarded and report.passes[-1].discard_reason
    kept = [p for p in report.passes if not p.discarded]
    assert report.output.sha256 == kept[-1].output_sha256 == hash_file(out)

    if len(kept) == 1:
        # Pass 2 was discarded: the shipped audio must be the approved
        # single-pass result, byte for byte.
        single = tmp_path / "single.wav"
        p1 = _run_cli("process", str(FLUTE), "-o", str(single))
        assert p1.returncode == 0, p1.stderr[-2000:]
        assert masked_wav_bytes(single.read_bytes()) == masked_wav_bytes(out.read_bytes())
