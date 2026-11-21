from __future__ import annotations

import queue
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hawavoclean.config import EnhancementConfig, HawaVoCleanConfig
from hawavoclean.enhancement.worker import IsolatedEnhancementWorker
from hawavoclean.errors import (
    ConfigError,
    PreflightError,
    WorkerCrashError,
    WorkerTimeoutError,
)
from hawavoclean.multipass import run_multipass
from hawavoclean.natural_contract import (
    _probe_optional_runtime_contract,
    _runtime_import_search_path,
    load_core_lock,
    load_natural_route_contract,
)
from hawavoclean.report.schema import (
    HawaVoCleanReport,
    MediaStats,
    UnitSummary,
    current_release_metadata,
)
from tests.support.report_provenance import build, core, environment, guard

# --- 1. Enhancement Worker Error Branches ---


def test_isolated_worker_error_message() -> None:
    with patch.object(IsolatedEnhancementWorker, "_start_worker"):
        worker = IsolatedEnhancementWorker()
        worker.process = MagicMock()
        worker.process.is_alive.return_value = True
        worker.req_queue = MagicMock()
        worker.resp_queue = MagicMock()

        worker.resp_queue.get.return_value = {"type": "ERROR", "error": "simulated GPU error"}
        with pytest.raises(WorkerCrashError, match="simulated GPU error"):
            worker.enhance(np.zeros(1600, dtype=np.float32), 16000)


def test_isolated_worker_unknown_message() -> None:
    with patch.object(IsolatedEnhancementWorker, "_start_worker"):
        worker = IsolatedEnhancementWorker()
        worker.process = MagicMock()
        worker.process.is_alive.return_value = True
        worker.req_queue = MagicMock()
        worker.resp_queue = MagicMock()

        worker.resp_queue.get.return_value = {"type": "MYSTERY_TYPE"}
        with pytest.raises(WorkerCrashError, match="Unknown message type"):
            worker.enhance(np.zeros(1600, dtype=np.float32), 16000)


def test_isolated_worker_timeout() -> None:
    with patch.object(IsolatedEnhancementWorker, "_start_worker"):
        worker = IsolatedEnhancementWorker()
        worker.process = MagicMock()
        worker.process.is_alive.return_value = True
        worker.req_queue = MagicMock()
        worker.resp_queue = MagicMock()

        worker.resp_queue.get.side_effect = queue.Empty()
        worker.timeout_s = 0.01
        with pytest.raises(WorkerTimeoutError, match="Worker timed out"):
            worker.enhance(np.zeros(1600, dtype=np.float32), 16000)


def test_isolated_worker_kill_escalation() -> None:
    with patch.object(IsolatedEnhancementWorker, "_start_worker"):
        worker = IsolatedEnhancementWorker()
        fake_proc = MagicMock()
        fake_proc.is_alive.side_effect = [True, True, False]
        worker.process = fake_proc
        worker.req_queue = MagicMock()
        worker.resp_queue = MagicMock()

        worker._kill_worker()
        assert fake_proc.terminate.called
        assert fake_proc.kill.called


def test_isolated_worker_graceful_close() -> None:
    with patch.object(IsolatedEnhancementWorker, "_start_worker"):
        worker = IsolatedEnhancementWorker()
        fake_proc = MagicMock()
        fake_proc.is_alive.side_effect = [True, False]
        fake_req = MagicMock()
        worker.process = fake_proc
        worker.req_queue = fake_req
        worker.resp_queue = MagicMock()

        worker.close(grace_s=1.0)
        assert fake_req.put.called
        assert fake_proc.join.called


# --- 2. Natural Contract Branches ---


def test_runtime_import_search_path_non_string() -> None:
    with patch("sys.path", [123, "/custom/valid/path"]):
        paths = _runtime_import_search_path()
        assert "/custom/valid/path" in paths
        assert "123" not in paths


def test_probe_optional_runtime_contract_timeout() -> None:
    with (
        patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["python"], timeout=1.0)),
        pytest.raises(PreflightError, match="optional runtime import/contract probe timed out"),
    ):
        _probe_optional_runtime_contract(
            core_id="test-core",
            probe_reference="mod.func",
            required_modules=("test_mod",),
            search_path=("/fake",),
        )


def test_probe_optional_runtime_contract_oserror() -> None:
    with (
        patch("subprocess.run", side_effect=OSError("Exec format error")),
        pytest.raises(
            PreflightError, match="optional runtime import/contract probe could not start"
        ),
    ):
        _probe_optional_runtime_contract(
            core_id="test-core",
            probe_reference="mod.func",
            required_modules=("test_mod",),
            search_path=("/fake",),
        )


def test_probe_optional_runtime_contract_failed_exit() -> None:
    fake_completed = MagicMock()
    fake_completed.returncode = 1
    fake_completed.stdout = (
        '@@HAWAVOCLEAN_DEPENDENCY_PROBE@@{"error": "MissingDep", "detail": "not installed"}\n'
    )
    with (
        patch("subprocess.run", return_value=fake_completed),
        pytest.raises(PreflightError, match="optional runtime import/contract failed"),
    ):
        _probe_optional_runtime_contract(
            core_id="test-core-failed",
            probe_reference="mod.func",
            required_modules=("test_mod",),
            search_path=("/fake",),
        )


def test_load_core_lock_table_validations(tmp_path: Path) -> None:
    # Invalid params / weights format (TOML format with string params)
    bad_lock = tmp_path / "bad.lock"
    bad_lock.write_text(
        'core_id = "test-core"\nparams_hash = "abc"\nparams = "not_a_dict"\n', encoding="utf-8"
    )

    fake_reg = MagicMock()
    fake_reg.lock_filename = "bad.lock"
    fake_reg.implementation_params_hash.return_value = "abc"

    with (
        patch("hawavoclean.natural_contract._probe_optional_runtime_contract"),
        patch("hawavoclean.natural_contract.models_dir", return_value=tmp_path),
        patch("hawavoclean.natural_contract.resolve_core", return_value=fake_reg),
        pytest.raises(PreflightError, match="Core lockfile params/weight tables are invalid"),
    ):
        load_core_lock("test-core")


def test_natural_route_contract_config_mismatches(tmp_path: Path) -> None:
    config = HawaVoCleanConfig(
        enhancement=EnhancementConfig(
            core_id="wiener-dd-48k-v1",
            phase_coherent=False,  # Mismatch: wiener core is phase_coherent=True
        )
    )

    fake_calib = tmp_path / "calib.json"

    with (
        patch("hawavoclean.natural_contract.resolve_calibration_file", return_value=fake_calib),
        patch(
            "hawavoclean.natural_contract.load_calibration_artifact",
            return_value={
                "thresholds": {},
                "calibration_id": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
            },
        ),
        patch(
            "hawavoclean.natural_contract.apply_calibrated_thresholds", return_value=config.guard
        ),
        patch(
            "hawavoclean.natural_contract.load_core_lock",
            return_value=({"phase_coherent": True, "expected_sample_rates": [48000]}, "hash123"),
        ),
        patch("hawavoclean.natural_contract.resolve_core"),
        patch("hawavoclean.natural_contract.hash_file", return_value="sha123"),
        pytest.raises(
            ConfigError,
            match="enhancement.phase_coherent = False but core 'wiener-dd-48k-v1' is phase-coherent",
        ),
    ):
        load_natural_route_contract("natural-clarity", config=config)


def test_natural_route_contract_sample_rate_mismatch(tmp_path: Path) -> None:
    config = HawaVoCleanConfig(
        enhancement=EnhancementConfig(
            core_id="wiener-dd-48k-v1",
            phase_coherent=True,
            model_sample_rate=16000,  # Mismatch: core expects 48000
        )
    )

    fake_calib = tmp_path / "calib.json"

    with (
        patch("hawavoclean.natural_contract.resolve_calibration_file", return_value=fake_calib),
        patch(
            "hawavoclean.natural_contract.load_calibration_artifact",
            return_value={
                "thresholds": {},
                "calibration_id": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
            },
        ),
        patch(
            "hawavoclean.natural_contract.apply_calibrated_thresholds", return_value=config.guard
        ),
        patch(
            "hawavoclean.natural_contract.load_core_lock",
            return_value=({"phase_coherent": True, "expected_sample_rates": [48000]}, "hash123"),
        ),
        patch("hawavoclean.natural_contract.resolve_core"),
        patch("hawavoclean.natural_contract.hash_file", return_value="sha123"),
        pytest.raises(
            ConfigError,
            match=r"enhancement.model_sample_rate = 16000 but core 'wiener-dd-48k-v1' runs at \[48000\]",
        ),
    ):
        load_natural_route_contract("natural-clarity", config=config)


# --- 3. Multipass Early Breaks Coverage ---


def _dummy_report() -> HawaVoCleanReport:
    stats = MediaStats(
        path="in.wav",
        sha256="485bab70a13e482016421d53d752001a85fce260deca06bf96d48176fbf1102a",
        sample_rate=48000,
        channels=1,
        samples=48000,
        duration_s=1.0,
        integrated_lufs=-19.0,
        true_peak_dbtp=-1.5,
    )
    return HawaVoCleanReport(
        release=current_release_metadata(),
        build=build(),
        job_id="job",
        config_hash="c" * 64,
        input=stats,
        output=stats,
        core=core("wiener-dd-48k-v1", "wiener-dd", "e" * 64),
        guard=guard("g", "f" * 64, "cal"),
        environment=environment(platform="p", os_version="v"),
        summary=UnitSummary(units_total=1, enhanced=1),
        units=[],
        passes=[],
    )


def test_multipass_pass_1_pristine_early_break(tmp_path: Path) -> None:
    input_wav = tmp_path / "in.wav"
    input_wav.write_bytes(b"RIFF" + b"\x00" * 40)
    output_wav = tmp_path / "out.wav"

    dummy_rep = _dummy_report()
    fake_buf = MagicMock()
    fake_buf.data = np.zeros((1, 48000), dtype=np.float32)

    def fake_run_pipeline(*_args: Any, **kwargs: Any) -> HawaVoCleanReport:
        p = Path(kwargs["output_path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"RIFF" + b"\x00" * 40)
        return dummy_rep

    # Pass 1 produces pristine separation >= 50 dB
    with (
        patch("hawavoclean.multipass.probe_audio", return_value=dummy_rep.input),
        patch("hawavoclean.multipass.decode_audio", return_value=fake_buf),
        patch("hawavoclean.multipass.run_pipeline", side_effect=fake_run_pipeline),
        patch("hawavoclean.multipass.measure_separation_db", return_value=52.0),
    ):
        res = run_multipass(input_wav, output_wav, passes="auto")
        assert len(res.passes) == 1
        assert res.passes[0].separation_db == 52.0


def test_multipass_pass_2_drift_exceeded_early_break(tmp_path: Path) -> None:
    input_wav = tmp_path / "in.wav"
    input_wav.write_bytes(b"RIFF" + b"\x00" * 40)
    output_wav = tmp_path / "out.wav"

    dummy_rep = _dummy_report()
    fake_buf = MagicMock()
    fake_buf.data = np.zeros((1, 48000), dtype=np.float32)

    def fake_run_pipeline(*_args: Any, **kwargs: Any) -> HawaVoCleanReport:
        p = Path(kwargs["output_path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"RIFF" + b"\x00" * 40)
        return dummy_rep

    with (
        patch("hawavoclean.multipass.probe_audio", return_value=dummy_rep.input),
        patch("hawavoclean.multipass.decode_audio", return_value=fake_buf),
        patch("hawavoclean.multipass.run_pipeline", side_effect=fake_run_pipeline),
        patch("hawavoclean.multipass.measure_separation_db", side_effect=[15.0, 18.0]),
        patch(
            "hawavoclean.multipass.cumulative_spectral_drift", side_effect=[5.5]
        ),  # 5.5 exceeds MAX_CUMULATIVE_DRIFT_DB (3.0)
    ):
        res = run_multipass(input_wav, output_wav, passes="auto")
        assert len(res.passes) == 2
        assert res.passes[1].discarded is True
        assert "cumulative spectral drift" in (res.passes[1].discard_reason or "")
