"""One authored release identity, exact mirrors, and honest report provenance."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from hawavoclean import __version__
from hawavoclean.release import RELEASE_IDENTITY, REPORT_SCHEMA_VERSION
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
