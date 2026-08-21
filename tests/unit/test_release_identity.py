"""One authored release identity, exact mirrors, and honest report provenance."""

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

import hawavoclean.release as release_module
from hawavoclean import __version__
from hawavoclean.release import (
    RELEASE_IDENTITY,
    REPORT_SCHEMA_VERSION,
    ReleaseIdentityError,
    _validated_identity,
)
from hawavoclean.report.schema import (
    CoreMetadata,
    EnvironmentMetadata,
    GuardMetadata,
    HawaVoCleanReport,
    MediaStats,
    UnitSummary,
    current_release_metadata,
)

ROOT = Path(__file__).resolve().parents[2]
IDENTITY_PATH = ROOT / "src" / "hawavoclean" / "release.json"


def _report_dict() -> dict[str, object]:
    media = MediaStats(
        path="in.wav",
        sha256="a" * 64,
        sample_rate=48000,
        channels=1,
        samples=48000,
        duration_s=1.0,
    )
    report = HawaVoCleanReport(
        schema_version=REPORT_SCHEMA_VERSION,
        release=current_release_metadata(),
        job_id="job",
        config_hash="b" * 64,
        input=media,
        output=media.model_copy(update={"path": "out.wav", "sha256": "c" * 64}),
        core=CoreMetadata(id="core", algorithm="algorithm", params_hash="d" * 64),
        guard=GuardMetadata(id="guard", probe_hash="e" * 64, calibration_id="f" * 64),
        environment=EnvironmentMetadata(
            platform="test",
            os_version="test",
            python_version="3",
            numpy_version="2",
            scipy_version="1",
            soundfile_version="0",
        ),
        summary=UnitSummary(),
    )
    return report.model_dump()


def test_runtime_and_identity_digest_derive_from_packaged_json() -> None:
    raw = IDENTITY_PATH.read_bytes()
    value = json.loads(raw)
    assert __version__ == RELEASE_IDENTITY.version == value["version"] == "3.3.0"
    assert REPORT_SCHEMA_VERSION == value["report_schema_version"] == 2
    assert RELEASE_IDENTITY.identity_sha256 == hashlib.sha256(raw).hexdigest()


def test_every_generated_package_version_matches_the_canonical_identity() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/sync_release_identity.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "5 generated mirrors agree" in result.stdout


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("product", "not-hawavoclean"),
        ("version", "999.0.0"),
        ("report_schema_version", 1),
        ("identity_sha256", "0" * 64),
    ],
)
def test_schema_v2_rejects_fabricated_release_identity(field: str, value: object) -> None:
    raw = _report_dict()
    assert isinstance(raw["release"], dict)
    raw["release"][field] = value
    with pytest.raises(ValidationError, match="does not match the packaged release"):
        HawaVoCleanReport.model_validate(raw)


def test_schema_v2_requires_identity_and_v1_does_not_invent_it() -> None:
    raw = _report_dict()
    del raw["release"]
    with pytest.raises(ValidationError, match="schema-v2 reports require release identity"):
        HawaVoCleanReport.model_validate(raw)

    raw["schema_version"] = 1
    legacy = HawaVoCleanReport.model_validate(raw)
    assert legacy.release is None


def test_schema_v1_rejects_a_backfilled_modern_identity() -> None:
    raw = _report_dict()
    raw["schema_version"] = 1
    with pytest.raises(ValidationError, match="schema-v1 reports cannot claim"):
        HawaVoCleanReport.model_validate(raw)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (b"\xff", "not valid UTF-8 JSON"),
        (b"{", "not valid UTF-8 JSON"),
        (b"[]", "fields must be exactly"),
        (b'{"identity_schema_version":1}', "fields must be exactly"),
        (
            b'{"identity_schema_version":2,"product":"hawavoclean",'
            b'"version":"3.3.0","report_schema_version":2}',
            "unsupported release identity schema",
        ),
        (
            b'{"identity_schema_version":1,"product":"other",'
            b'"version":"3.3.0","report_schema_version":2}',
            "wrong product",
        ),
        (
            b'{"identity_schema_version":1,"product":"hawavoclean",'
            b'"version":330,"report_schema_version":2}',
            "canonical MAJOR.MINOR.PATCH",
        ),
        (
            b'{"identity_schema_version":1,"product":"hawavoclean",'
            b'"version":"03.3.0","report_schema_version":2}',
            "canonical MAJOR.MINOR.PATCH",
        ),
        (
            b'{"identity_schema_version":1,"product":"hawavoclean",'
            b'"version":"3.3.0","report_schema_version":1}',
            "supports report_schema_version 2",
        ),
    ],
)
def test_malformed_release_identity_fails_closed(value: bytes, message: str) -> None:
    with pytest.raises(ReleaseIdentityError, match=message):
        _validated_identity(value)


def test_packaged_identity_read_failure_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    class UnreadableResource:
        def joinpath(self, _name: str) -> "UnreadableResource":
            return self

        def read_bytes(self) -> bytes:
            raise OSError("package data unavailable")

    monkeypatch.setattr(release_module, "files", lambda _package: UnreadableResource())
    with pytest.raises(ReleaseIdentityError, match="cannot read packaged release identity"):
        release_module.load_release_identity()


def test_defensive_schema_guards_reject_constructed_impossible_states() -> None:
    raw = _report_dict()
    current = HawaVoCleanReport.model_validate(raw)

    future = current.model_copy(update={"schema_version": 3})
    with pytest.raises(ValueError, match="schema does not match"):
        cast(Callable[[], HawaVoCleanReport], future.release_matches_schema)()

    assert current.release is not None
    mismatched_release = current.release.model_copy(update={"report_schema_version": 1})
    mismatched = current.model_copy(update={"release": mismatched_release})
    with pytest.raises(ValueError, match="embedded release identity disagree"):
        cast(Callable[[], HawaVoCleanReport], mismatched.release_matches_schema)()
