"""The emitted audit report must contain no placeholder or fabricated values."""

import json
import re
import shutil
from pathlib import Path

import pytest

from hawavoclean.pipeline import run_pipeline
from hawavoclean.report.writer import serialize_json_report

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "sample_sorani_podcast.wav"

PLACEHOLDER = re.compile(r"(?i)(placeholder|todo\b|tbd\b|\bxxx\b|fixme|^unknown$|_sha256$)")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _walk_strings(obj: object, path: str = "$") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_walk_strings(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_walk_strings(v, f"{path}[{i}]"))
    elif isinstance(obj, str):
        out.append((path, obj))
    return out


@pytest.mark.integration
def test_report_contains_no_placeholder_values(tmp_path: Path) -> None:
    work = REPO / ".hawavoclean-work"
    shutil.rmtree(work, ignore_errors=True)
    try:
        report = run_pipeline(
            input_path=FIXTURE,
            output_path=tmp_path / "out.wav",
            profile="production",
            overwrite=True,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    data = json.loads(serialize_json_report(report))
    offenders: list[str] = []
    for path, value in _walk_strings(data):
        if value == "":
            offenders.append(f"{path}: empty string")
        elif PLACEHOLDER.search(value):
            offenders.append(f"{path}: placeholder-shaped value {value!r}")
        if path.endswith("sha256") and value and not HEX64.match(value):
            offenders.append(f"{path}: field named sha256 holds non-digest {value!r}")
    assert not offenders, "Report contains placeholders:\n" + "\n".join(offenders)
