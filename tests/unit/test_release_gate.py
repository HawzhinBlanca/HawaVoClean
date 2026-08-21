from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from scripts import release_gate


def _pass(index: int, suffix: str = "") -> dict[str, object]:
    return {
        "index": index,
        "artifacts": {
            name: {"sha256": f"{position:064x}{suffix}"}
            for position, name in enumerate(release_gate.PROMISED_ARTIFACTS, start=1)
        },
    }


def test_compare_runs_requires_every_artifact_and_exact_repetition() -> None:
    result = release_gate._compare_runs([_pass(1), _pass(2)])
    assert result["status"] == "passed"
    assert result["passes"] == 2

    changed = _pass(2)
    artifacts = changed["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts["wheel"] = {"sha256": "f" * 64}
    with pytest.raises(release_gate.GateError, match="non-reproducible artifact wheel"):
        release_gate._compare_runs([_pass(1), changed])


def test_artifact_identity_is_path_independent_and_mode_sensitive(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "file").write_bytes(b"payload")
    (second / "file").write_bytes(b"payload")

    left = release_gate._artifact_identity("bundle", first)
    right = release_gate._artifact_identity("bundle", second)
    assert left == right

    (second / "file").chmod(0o755)
    assert release_gate._artifact_identity("bundle", second)["sha256"] != left["sha256"]


def test_private_regression_copy_rejects_escape_and_verifies_hash(tmp_path: Path) -> None:
    source = tmp_path / "source"
    checkout = tmp_path / "checkout"
    source.mkdir()
    checkout.mkdir()
    private = source / "test_output" / "private.wav"
    private.parent.mkdir()
    private.write_bytes(b"audio")
    digest = release_gate._sha256(private)
    manifest = source / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "input": "test_output/private.wav",
                        "input_sha256": digest,
                        "reference_audio": "test_output/private.wav",
                        "audio_sha256": digest,
                        "reference_report": "test_output/private.wav",
                        "report_sha256": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert release_gate._copy_private_inputs(source, checkout, manifest) == {
        "test_output/private.wav": digest
    }
    assert (checkout / "test_output" / "private.wav").read_bytes() == b"audio"

    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["cases"][0]["input"] = "../escape.wav"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(release_gate.GateError, match="escapes"):
        release_gate._copy_private_inputs(source, checkout, manifest)


def test_written_report_has_a_verifiable_canonical_proof_hash(tmp_path: Path) -> None:
    path = tmp_path / "proof.json"
    report = {"schema_version": 1, "status": "passed", "runs": []}
    release_gate._write_report(path, report)

    stored = json.loads(path.read_text(encoding="utf-8"))
    proof = stored.pop("proof_sha256")
    assert proof == release_gate._canonical_sha256(stored)


def test_runner_scopes_build_environment_to_one_step(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    runner = release_gate.Runner(checkout, tmp_path / "logs", {"BASE": "present"})
    output = tmp_path / "environment.txt"
    probe = [
        os.fspath(Path(sys.executable)),
        "-c",
        (
            "import os,pathlib;"
            f"pathlib.Path({str(output)!r}).write_text("
            "os.environ.get('BASE','')+'|'+os.environ.get('BUILD_ONLY',''))"
        ),
    ]

    runner.run("scoped", probe, extra_environment={"BUILD_ONLY": "yes"})
    assert output.read_text(encoding="utf-8") == "present|yes"
    assert runner.environment == {"BASE": "present"}


def test_verify_command_names_the_exact_report() -> None:
    assert release_gate._verify_command(
        ["hawavoclean"], "output.wav", "output.hawavoclean.json"
    ) == [
        "hawavoclean",
        "verify",
        "output.wav",
        "--report",
        "output.hawavoclean.json",
    ]
