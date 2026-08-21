#!/usr/bin/env python3
"""Run deterministic real-audio regressions against frozen profile references."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from hawavoclean.report.schema import HawaVoCleanReport

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "evidence" / "release" / "audio-regressions.json"
HEX64 = set("0123456789abcdef")
CASE_FIELDS = {
    "id",
    "profile",
    "input",
    "input_sha256",
    "reference_source_commit",
    "reference_audio",
    "audio_sha256",
    "candidate_audio_sha256",
    "reference_report",
    "report_sha256",
    "core_id",
}


class RegressionError(RuntimeError):
    """A frozen artifact, output hash or semantic report comparison failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_file(relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RegressionError("artifact paths must be non-empty strings")
    candidate = (ROOT / relative).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RegressionError(f"artifact escapes repository: {relative}") from exc
    if not candidate.is_file():
        raise RegressionError(f"required local artifact is unavailable: {relative}")
    return candidate


def _expect_hash(path: Path, expected: object, label: str) -> None:
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(char not in HEX64 for char in expected)
    ):
        raise RegressionError(f"{label} is not a lowercase SHA-256")
    actual = _sha256(path)
    if actual != expected:
        raise RegressionError(f"{label} mismatch: expected {expected}, got {actual}")


def _manifest(path: Path) -> dict[str, Any]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "deterministic_device",
        "audio_drift_contract",
        "allowed_report_drift",
        "cases",
    }:
        raise RegressionError("regression manifest has an unexpected shape")
    if value["schema_version"] != 1 or value["deterministic_device"] != "cpu":
        raise RegressionError("only schema 1 on the deterministic CPU path is supported")
    audio_contract = value["audio_drift_contract"]
    if (
        not isinstance(audio_contract, dict)
        or set(audio_contract) != {"reason", "max_absolute_lsb"}
        or not isinstance(audio_contract["reason"], str)
        or not audio_contract["reason"]
        or not isinstance(audio_contract["max_absolute_lsb"], (int, float))
        or isinstance(audio_contract["max_absolute_lsb"], bool)
        or audio_contract["max_absolute_lsb"] <= 0
    ):
        raise RegressionError("audio_drift_contract is invalid")
    limits = value["allowed_report_drift"]
    if not isinstance(limits, list) or not all(isinstance(item, str) and item for item in limits):
        raise RegressionError("allowed_report_drift must be a non-empty string list")
    cases = value["cases"]
    if not isinstance(cases, list) or not cases:
        raise RegressionError("regression manifest must contain cases")
    ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != CASE_FIELDS:
            raise RegressionError("regression case fields differ from the schema")
        if case["profile"] not in {"production", "studio", "lowband"}:
            raise RegressionError(f"invalid profile in case {case.get('id')}")
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise RegressionError("regression case IDs must be unique non-empty strings")
        ids.add(case_id)
        commit = case["reference_source_commit"]
        if not isinstance(commit, str) or len(commit) != 40 or any(c not in HEX64 for c in commit):
            raise RegressionError(f"invalid reference commit in case {case_id}")
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        _expect_hash(_repo_file(case["input"]), case["input_sha256"], f"{case_id} input")
        _expect_hash(
            _repo_file(case["reference_audio"]),
            case["audio_sha256"],
            f"{case_id} reference audio",
        )
        candidate_hash = case["candidate_audio_sha256"]
        if (
            not isinstance(candidate_hash, str)
            or len(candidate_hash) != 64
            or any(char not in HEX64 for char in candidate_hash)
        ):
            raise RegressionError(f"{case_id} candidate audio is not a lowercase SHA-256")
        _expect_hash(
            _repo_file(case["reference_report"]),
            case["report_sha256"],
            f"{case_id} reference report",
        )
    return value


def _semantic_report(report: HawaVoCleanReport) -> dict[str, Any]:
    """Keep audio/decision truth; remove only the declared run/build drift."""
    value: dict[str, Any] = copy.deepcopy(report.model_dump())
    value.pop("schema_version")
    value.pop("release")
    value.pop("build")
    value.pop("job_id")
    value.pop("environment")
    value["core"].pop("lock_sha256")
    value["core"].pop("weight_sha256")
    value["guard"].pop("calibration_sha256")
    value["input"].pop("path")
    value["output"].pop("path")
    value["output"].pop("sha256")
    for unit in value["units"]:
        unit.pop("runtime_ms")
    return value


def _semantic_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _measure_dither(reference: Path, candidate: Path, max_absolute_lsb: float) -> dict[str, Any]:
    old, old_rate = sf.read(reference, dtype="float64", always_2d=True)
    new, new_rate = sf.read(candidate, dtype="float64", always_2d=True)
    if old_rate != new_rate or old.shape != new.shape:
        raise RegressionError("candidate changed sample rate, count or channel layout")
    difference = np.asarray(new - old, dtype=np.float64)
    lsb = 1.0 / float(2**23)
    max_lsb = float(np.max(np.abs(difference)) / lsb) if difference.size else 0.0
    rms_lsb = (
        float(np.sqrt(np.mean(np.square(difference), dtype=np.float64)) / lsb)
        if difference.size
        else 0.0
    )
    if max_lsb > max_absolute_lsb:
        raise RegressionError(
            f"audio drift exceeds dither contract: {max_lsb:.6f} > {max_absolute_lsb:.6f} LSB"
        )
    return {
        "max_absolute_lsb": max_lsb,
        "rms_lsb": rms_lsb,
        "changed_samples": int(np.count_nonzero(difference)),
        "total_samples": int(difference.size),
    }


def _run_case(
    case: dict[str, Any], runs: int, root: Path, max_absolute_lsb: float
) -> dict[str, Any]:
    case_id = str(case["id"])
    reference = HawaVoCleanReport.model_validate_json(
        _repo_file(case["reference_report"]).read_text(encoding="utf-8")
    )
    expected_semantic = _semantic_report(reference)
    output_hashes: list[str] = []
    report_semantics: list[dict[str, Any]] = []
    job_ids: list[str] = []
    for run_index in range(1, runs + 1):
        run_dir = root / f"run-{run_index}"
        run_dir.mkdir(parents=True, exist_ok=True)
        output = run_dir / f"{case_id}.wav"
        env = os.environ.copy()
        env["HAWAVOCLEAN_DEVICE"] = "cpu"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "hawavoclean.cli",
                "process",
                str(_repo_file(case["input"])),
                "--output",
                str(output),
                "--profile",
                str(case["profile"]),
                "--overwrite",
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RegressionError(
                f"{case_id} run {run_index} failed ({completed.returncode}): "
                f"{completed.stderr[-2000:]}"
            )
        actual_hash = _sha256(output)
        if actual_hash != case["candidate_audio_sha256"]:
            raise RegressionError(
                f"{case_id} run {run_index} audio drift: "
                f"expected frozen v3.3 candidate {case['candidate_audio_sha256']}, "
                f"got {actual_hash}"
            )
        report_path = output.with_suffix(".hawavoclean.json")
        current = HawaVoCleanReport.model_validate_json(report_path.read_text(encoding="utf-8"))
        if current.output.sha256 != actual_hash or current.core.id != case["core_id"]:
            raise RegressionError(f"{case_id} report does not describe its emitted master/core")
        current_semantic = _semantic_report(current)
        if current_semantic != expected_semantic:
            expected_hash = _semantic_digest(expected_semantic)
            actual_semantic_hash = _semantic_digest(current_semantic)
            raise RegressionError(
                f"{case_id} has unexplained semantic report drift: "
                f"expected {expected_hash}, got {actual_semantic_hash}"
            )
        output_hashes.append(actual_hash)
        report_semantics.append(current_semantic)
        job_ids.append(current.job_id)
        if run_index == 1:
            dither_measurement = _measure_dither(
                _repo_file(case["reference_audio"]), output, max_absolute_lsb
            )
    if len(set(output_hashes)) != 1 or len(set(job_ids)) != 1:
        raise RegressionError(f"{case_id} is not deterministic across {runs} runs")
    first_semantic = report_semantics[0]
    if any(value != first_semantic for value in report_semantics[1:]):
        raise RegressionError(f"{case_id} report semantics are not deterministic")
    return {
        "id": case_id,
        "profile": case["profile"],
        "runs": runs,
        "audio_sha256": output_hashes[0],
        "reference_audio_sha256": case["audio_sha256"],
        "audio_difference": dither_measurement,
        "semantic_report_sha256": _semantic_digest(first_semantic),
        "job_id": job_ids[0],
        "status": "passed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    try:
        if args.runs < 2:
            raise RegressionError("--runs must be at least 2 to prove repetition")
        manifest = _manifest(args.manifest.resolve())
        cases: list[dict[str, Any]] = manifest["cases"]
        if args.case:
            requested = set(args.case)
            cases = [case for case in cases if case["id"] in requested]
            found = {str(case["id"]) for case in cases}
            if found != requested:
                raise RegressionError(f"unknown case IDs: {sorted(requested - found)}")
        with tempfile.TemporaryDirectory(prefix="hawavoclean-regression-") as temp:
            max_lsb = float(manifest["audio_drift_contract"]["max_absolute_lsb"])
            results = [_run_case(case, args.runs, Path(temp), max_lsb) for case in cases]
        payload = {
            "schema_version": 1,
            "manifest_sha256": _sha256(args.manifest.resolve()),
            "deterministic_device": manifest["deterministic_device"],
            "cases": results,
            "status": "passed",
        }
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.output_json is not None:
            output = args.output_json.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
    except (
        OSError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
        RegressionError,
    ) as exc:
        print(f"audio regression gate failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
