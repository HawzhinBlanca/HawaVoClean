"""Bugs found by adversarial review of pipeline/worker/job/CLI (round 4).
Each test is the hunter's minimal repro, kept as a permanent regression."""

import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from hawavoclean.errors import PublicationError
from hawavoclean.pipeline import run_pipeline

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures" / "sample_noisy_hum.wav"
CLI = str(Path(sys.executable).with_name("hawavoclean"))


# 3. output == input must be refused BEFORE any processing ------------------------
def test_output_equal_to_input_is_refused_and_source_untouched(tmp_path: Path) -> None:
    src = tmp_path / "src.wav"
    shutil.copy(FIX, src)
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    with pytest.raises(PublicationError):
        run_pipeline(src, src, overwrite=True)
    after = hashlib.sha256(src.read_bytes()).hexdigest()
    assert before == after, "SOURCE FILE WAS MODIFIED"


def test_output_siblings_colliding_with_input_are_refused(tmp_path: Path) -> None:
    """`-o x.hawavoclean.json`-shaped collisions: the report sidecars must not
    overwrite the input either."""
    src = tmp_path / "take.hawavoclean.json"  # an input that happens to have this name
    shutil.copy(FIX, src)
    before = src.read_bytes()
    with pytest.raises(PublicationError):
        run_pipeline(src, tmp_path / "take.wav", overwrite=True)
    assert src.read_bytes() == before


# 5. workspace must not leak on user-error paths ------------------------------------
def test_no_workspace_leak_on_destination_exists_refusal(tmp_path: Path, monkeypatch: Any) -> None:
    work = tmp_path / "work"
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(work))
    dest = tmp_path / "out.wav"
    dest.write_bytes(b"x")
    for _ in range(3):
        with pytest.raises(PublicationError):
            run_pipeline(FIX, dest, overwrite=False)
    leftovers = list(work.iterdir()) if work.exists() else []
    assert not leftovers, f"workspaces leaked on a user-error path: {leftovers}"


# 11. unwritable destination: clean error at preflight, no traceback, no leak ------
def test_unwritable_destination_fails_at_preflight(tmp_path: Path, monkeypatch: Any) -> None:
    work = tmp_path / "work"
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(work))
    ro = tmp_path / "ro"
    ro.mkdir()
    os.chmod(ro, 0o500)
    try:
        t0 = time.perf_counter()
        with pytest.raises(PublicationError):
            run_pipeline(FIX, ro / "o.wav", overwrite=True)
        assert time.perf_counter() - t0 < 2.0, "refusal came after processing, not at preflight"
    finally:
        os.chmod(ro, 0o700)
    leftovers = list(work.iterdir()) if work.exists() else []
    assert not leftovers


# 7. tampered calibration must be refused by the PIPELINE too ------------------------
def test_pipeline_refuses_tampered_calibration(tmp_path: Path, monkeypatch: Any) -> None:
    from hawavoclean.errors import CalibrationError
    from hawavoclean.paths import models_dir

    real = models_dir()
    override = tmp_path / "models"
    shutil.copytree(real, override)
    calib = override / "guard-calibration.json"
    calib.write_text(
        calib.read_text().replace('"max_timing_drift_ms": 75.0', '"max_timing_drift_ms": 900.0')
    )
    monkeypatch.setenv("HAWAVOCLEAN_MODEL_DIR", str(override))
    with pytest.raises(CalibrationError):
        run_pipeline(FIX, tmp_path / "o.wav", overwrite=True)


# 6. verify must use the CONFIGURED ceiling ------------------------------------------
def test_verify_honors_configured_ceiling(tmp_path: Path) -> None:
    from hawavoclean.paths import profile_config_path

    cfg = tmp_path / "hot.toml"
    base = profile_config_path("production").read_text()
    cfg.write_text(base.replace("true_peak_ceiling_dbtp = -1.0", "true_peak_ceiling_dbtp = -0.2"))
    out = tmp_path / "hot.wav"
    env = {**os.environ, "HAWAVOCLEAN_WORK_DIR": str(tmp_path / "w")}
    r = subprocess.run(
        [CLI, "process", str(FIX), "-o", str(out), "-c", str(cfg), "--overwrite"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, r.stderr[-400:]
    v = subprocess.run(
        [CLI, "verify", str(out), "-r", str(tmp_path / "hot.hawavoclean.json")],
        capture_output=True,
        text=True,
        env=env,
    )
    assert v.returncode == 0, (
        f"verify rejected a master that met its own configured ceiling:\n{v.stderr}"
    )


# 9. empty manifest must not pass the release gate ------------------------------------
def test_eval_gate_fails_on_empty_manifest(tmp_path: Path) -> None:
    from hawavoclean.eval.acceptance import evaluate_acceptance_gates

    m = tmp_path / "empty.jsonl"
    m.write_text("")
    res = evaluate_acceptance_gates(m, output_dir=tmp_path / "o")
    assert res["release_gate_status"] == "FAILED"


# 10. eval/benchmark/calibrate must exit with documented codes, no traceback ---------
@pytest.mark.parametrize("sub", ["eval", "benchmark", "calibrate"])
def test_eval_family_missing_manifest_exits_4_no_traceback(sub: str, tmp_path: Path) -> None:
    r = subprocess.run(
        [CLI, sub, "-m", str(tmp_path / "nope.json"), "-o", str(tmp_path / "x")],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 4, f"{sub}: exit {r.returncode}\n{r.stderr[-300:]}"
    assert "Traceback" not in r.stderr + r.stdout


# 4. batch stem collisions must be refused up front ------------------------------------
def test_batch_refuses_colliding_stems(tmp_path: Path) -> None:
    a_wav = tmp_path / "a.wav"
    a_flac = tmp_path / "a.flac"
    shutil.copy(FIX, a_wav)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(a_wav), str(a_flac)], check=True)
    r = subprocess.run(
        [CLI, "batch", str(a_wav), str(a_flac), "-o", str(tmp_path / "out"), "--overwrite"],
        capture_output=True,
        text=True,
        env={**os.environ, "HAWAVOCLEAN_WORK_DIR": str(tmp_path / "w")},
    )
    assert r.returncode == 4, f"colliding stems were not refused (exit {r.returncode})"
    assert "a_clean.wav" in (r.stderr + r.stdout)


# 8. isolated worker must honor phase_coherent; misconfig is a clean config error -----------
def test_studio_with_phase_coherent_true_is_a_config_error(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from hawavoclean.paths import profile_config_path

    cfg = tmp_path / "bad.toml"
    cfg.write_text(
        profile_config_path("studio")
        .read_text()
        .replace("phase_coherent = false", "phase_coherent = true")
    )
    r = subprocess.run(
        [CLI, "process", str(FIX), "-o", str(tmp_path / "o.wav"), "-c", str(cfg), "--overwrite"],
        capture_output=True,
        text=True,
        env={**os.environ, "HAWAVOCLEAN_WORK_DIR": str(tmp_path / "w")},
    )
    assert r.returncode == 2, f"expected preflight (2), got {r.returncode}\n{r.stderr[-300:]}"
    assert "Traceback" not in r.stderr


# 1 + 2. worker: no hang at exit after child death; fast crash detection ----------------------
def test_worker_child_death_is_detected_fast_and_process_exits() -> None:
    script = r"""
import os, sys, time, numpy as np
from hawavoclean.enhancement import worker as W
from hawavoclean.enhancement.protocol import EnhancementResult, EnhancerMetadata
from hawavoclean.errors import WorkerError
class Dies:
    def __init__(self, _core_id="x", sample_rate=48000, **_): self._m = EnhancerMetadata("d","0","t",sample_rate,True)
    @property
    def metadata(self): return self._m
    def warmup(self): pass
    def enhance(self, w, sr): os._exit(0)   # die instantly on the FIRST request
if __name__ == "__main__":
    wk = W.IsolatedEnhancementWorker(timeout_s=60.0, enhancer_class=Dies)
    t0 = time.perf_counter()
    try:
        wk.enhance(np.zeros(48000*16, np.float32), 48000)   # 3 MB payload (> pipe buffer)
        print("UNEXPECTED RETURN")
    except WorkerError as e:
        print(f"detected in {time.perf_counter()-t0:.1f}s: {type(e).__name__}")
    wk.close()
    print("closing interpreter")
"""
    # spawn-mode children re-import __main__ by path, so the script must be a
    # real file, not a -c string.
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix="_worker_death.py", delete=False) as f:
        f.write(script)
        script_path = f.name
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            [sys.executable, script_path], capture_output=True, text=True, timeout=40
        )
    except subprocess.TimeoutExpired:
        pytest.fail("interpreter HUNG at exit after worker child death")
    finally:
        os.unlink(script_path)
    elapsed = time.perf_counter() - t0
    assert "detected in" in r.stdout, r.stdout + r.stderr[-400:]
    secs = float(r.stdout.split("detected in ")[1].split("s")[0])
    assert secs < 10.0, f"dead child noticed only after {secs:.1f}s (waited for the full timeout)"
    assert elapsed < 30.0
