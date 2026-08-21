"""Adversarial coverage for build and installed-distribution provenance."""

import hashlib
import io
import json
import subprocess
from importlib import metadata, resources
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import hawavoclean.provenance as provenance
from hawavoclean.provenance import ProvenanceError
from hawavoclean.release import RELEASE_IDENTITY


def _valid_identity(artifact_type: str = "source-tree") -> dict[str, Any]:
    base: dict[str, Any] = {
        "provenance_schema_version": 1,
        "artifact_type": artifact_type,
        "source_revision": "a" * 40,
        "source_date_epoch": 1,
        "source_dirty": artifact_type == "source-tree",
        "dependency_lock_sha256": "b" * 64,
        "release_identity_sha256": RELEASE_IDENTITY.identity_sha256,
    }
    base["build_id"] = hashlib.sha256(provenance._canonical(base)).hexdigest()
    return base


def _recompute(value: dict[str, Any]) -> dict[str, Any]:
    base = {key: item for key, item in value.items() if key != "build_id"}
    value["build_id"] = hashlib.sha256(provenance._canonical(base)).hexdigest()
    return value


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: value.update(extra=True), "field set"),
        (lambda value: value.update(provenance_schema_version=2), "unsupported"),
        (lambda value: value.update(artifact_type="zip"), "artifact type"),
        (lambda value: value.update(source_revision="short"), "full Git SHA"),
        (lambda value: value.update(source_date_epoch=0), "source date epoch"),
        (lambda value: value.update(source_dirty="no"), "dirty-source flag"),
        (lambda value: value.update(dependency_lock_sha256="short"), "not a SHA-256"),
        (lambda value: value.update(build_id="0" * 64), "does not recompute"),
    ],
)
def test_malformed_build_identity_fails_closed(change: Any, message: str) -> None:
    value = _valid_identity()
    change(value)
    with pytest.raises(ProvenanceError, match=message):
        provenance._validate_build_identity(value)


def test_release_and_dirty_artifact_disagreement_fail_after_recomputation() -> None:
    wrong_release = _valid_identity()
    wrong_release["release_identity_sha256"] = "0" * 64
    with pytest.raises(ProvenanceError, match="different packaged release"):
        provenance._validate_build_identity(_recompute(wrong_release))

    dirty_wheel = _valid_identity("wheel")
    dirty_wheel["source_dirty"] = True
    with pytest.raises(ProvenanceError, match="cannot claim dirty source"):
        provenance._validate_build_identity(_recompute(dirty_wheel))


def test_git_and_non_checkout_failures_are_designed_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("gone"))
    )
    with pytest.raises(ProvenanceError, match="cannot inspect"):
        provenance._git(tmp_path, "status")

    fake_module = tmp_path / "src" / "hawavoclean" / "provenance.py"
    monkeypatch.setattr(provenance, "__file__", str(fake_module))
    with pytest.raises(ProvenanceError, match="not a source checkout"):
        provenance._source_tree_identity()


class _Resource:
    def __init__(self, raw: bytes | None) -> None:
        self.raw = raw

    def joinpath(self, _name: str) -> "_Resource":
        return self

    def open(self, _mode: str) -> io.BytesIO:
        if self.raw is None:
            raise FileNotFoundError
        return io.BytesIO(self.raw)


def test_packaged_identity_fallback_invalid_json_and_valid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _valid_identity()
    monkeypatch.setattr(resources, "files", lambda _name: _Resource(None))
    monkeypatch.setattr(provenance, "_source_tree_identity", lambda: expected)
    assert provenance.packaged_build_identity() == expected

    monkeypatch.setattr(resources, "files", lambda _name: _Resource(b"{"))
    with pytest.raises(ProvenanceError, match="not valid UTF-8 JSON"):
        provenance.packaged_build_identity()

    raw = json.dumps(expected).encode()
    monkeypatch.setattr(resources, "files", lambda _name: _Resource(raw))
    assert provenance.packaged_build_identity() == expected


def test_distribution_record_absence_and_read_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = metadata.PackageNotFoundError("hawavoclean")
    monkeypatch.setattr(metadata, "distribution", lambda _name: (_ for _ in ()).throw(missing))
    assert provenance.distribution_record_sha256() is None

    no_files = SimpleNamespace(files=None)
    monkeypatch.setattr(metadata, "distribution", lambda _name: no_files)
    assert provenance.distribution_record_sha256() is None

    record = SimpleNamespace(
        files=["hawavoclean-3.3.0.dist-info/RECORD"],
        locate_file=lambda _entry: tmp_path / "missing-record",
    )
    monkeypatch.setattr(metadata, "distribution", lambda _name: record)
    with pytest.raises(ProvenanceError, match="cannot read installed distribution RECORD"):
        provenance.distribution_record_sha256()


def test_wheel_report_must_match_both_packaged_identity_and_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = _valid_identity("wheel")
    report = {**wheel, "distribution_record_sha256": "c" * 64}
    monkeypatch.setattr(provenance, "packaged_build_identity", lambda: wheel)
    monkeypatch.setattr(provenance, "distribution_record_sha256", lambda: "c" * 64)
    provenance.verify_report_build(report)

    bad_format = {**report, "distribution_record_sha256": "short"}
    with pytest.raises(ProvenanceError, match="RECORD digest is not"):
        provenance.verify_report_build(bad_format)

    other_build = {**wheel, "source_revision": "d" * 40}
    other = {**_recompute(other_build), "distribution_record_sha256": "c" * 64}
    with pytest.raises(ProvenanceError, match="does not match the installed wheel"):
        provenance.verify_report_build(other)

    monkeypatch.setattr(provenance, "distribution_record_sha256", lambda: None)
    with pytest.raises(ProvenanceError, match="no readable RECORD"):
        provenance.verify_report_build(report)

    monkeypatch.setattr(provenance, "distribution_record_sha256", lambda: "d" * 64)
    with pytest.raises(ProvenanceError, match="RECORD digest does not match"):
        provenance.verify_report_build(report)


def test_missing_packages_and_external_version_failures_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = metadata.PackageNotFoundError("absent")
    monkeypatch.setattr(metadata, "version", lambda _name: (_ for _ in ()).throw(missing))
    assert provenance._package_version("absent") is None

    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("gone"))
    )
    assert provenance._command_version("missing").startswith("unavailable (OSError)")

    empty = SimpleNamespace(stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: empty)
    assert provenance._command_version("empty") == "unavailable (empty version output)"
