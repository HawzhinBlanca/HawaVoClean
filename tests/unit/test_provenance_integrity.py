"""Provenance integrity: every digest in models/ must be real, recomputable, and honest."""

import hashlib
import json
import re
import string
import tomllib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / "src" / "voiceclean" / "resources" / "models"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _known_fabrication_digests() -> set[str]:
    """Digests of trivially-empty inputs, computed here rather than hardcoded."""
    known = {hashlib.sha256(b"").hexdigest()}
    for ch in string.printable:
        known.add(hashlib.sha256(ch.encode()).hexdigest())
    return known


def _collect_string_values(obj: object) -> list[tuple[str, str]]:
    """Flatten (key, value) string pairs from nested dicts/lists."""
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                out.append((str(k), v))
            else:
                out.extend(_collect_string_values(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_collect_string_values(v))
    return out


def _load_models_documents() -> dict[Path, object]:
    docs: dict[Path, object] = {}
    for p in sorted(MODELS_DIR.glob("*.toml")):
        with open(p, "rb") as f:
            docs[p] = tomllib.load(f)
    for p in sorted(MODELS_DIR.glob("*.json")):
        docs[p] = json.loads(p.read_text(encoding="utf-8"))
    return docs


def test_no_digest_is_a_trivial_fabrication() -> None:
    """No 64-hex value anywhere in models/ may be the hash of empty/one-char input."""
    fabrications = _known_fabrication_digests()
    offenders: list[str] = []
    for path, doc in _load_models_documents().items():
        for key, value in _collect_string_values(doc):
            if HEX64.match(value) and value in fabrications:
                offenders.append(f"{path.name}:{key} = {value[:16]}... (trivial-input hash)")
    assert not offenders, "Fabricated digests found:\n" + "\n".join(offenders)


def test_weight_digests_resolve_to_real_files() -> None:
    """Any weight_sha256 table must name files that exist and hash to the value."""
    offenders: list[str] = []
    for path, doc in _load_models_documents().items():
        if not isinstance(doc, dict):
            continue
        table = doc.get("weight_sha256")
        if not isinstance(table, dict):
            continue
        for fname, claimed in table.items():
            candidate = MODELS_DIR / str(fname)
            if not candidate.exists():
                offenders.append(f"{path.name}: weights file missing: {fname}")
                continue
            actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if actual != claimed:
                offenders.append(f"{path.name}: {fname} digest mismatch")
    assert not offenders, "Unverifiable weight digests:\n" + "\n".join(offenders)


def test_no_unverifiable_commit_claims() -> None:
    """models/ must not pin commits of external repos this project cannot verify."""
    offenders: list[str] = []
    for path, doc in _load_models_documents().items():
        for key, value in _collect_string_values(doc):
            if key == "commit":
                offenders.append(f"{path.name}: unverifiable commit claim: {value[:12]}...")
    assert not offenders, "\n".join(offenders)


def test_calibration_artifact_is_internally_consistent() -> None:
    """calibration_id must recompute from the thresholds it claims to lock."""
    from voiceclean.hashing import hash_json_canonical

    calib_path = MODELS_DIR / "guard-calibration.json"
    data = json.loads(calib_path.read_text(encoding="utf-8"))
    expected = hash_json_canonical(data["thresholds"])
    assert data["calibration_id"] == expected, (
        f"calibration_id {data['calibration_id'][:16]}... does not recompute from "
        f"thresholds (expected {expected[:16]}...); it is not derived from anything"
    )


def test_no_hand_written_quality_metrics() -> None:
    """Quality metrics may only appear alongside measurement provenance."""
    calib_path = MODELS_DIR / "guard-calibration.json"
    data = json.loads(calib_path.read_text(encoding="utf-8"))
    metrics = data.get("metrics")
    if metrics is None:
        return  # absence is honest
    assert "measured" in data and data["measured"].get("item_count", 0) > 0, (
        "metrics block present without measurement provenance — "
        "these numbers were typed in, not measured"
    )
