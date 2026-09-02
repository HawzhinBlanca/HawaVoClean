from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hawavoclean.config import EnhancementConfig, HawaVoCleanConfig
from hawavoclean.errors import ConfigError, PreflightError
from hawavoclean.model_packs.errors import ModelPackManifestError
from hawavoclean.model_packs.manifest import _parse_payload, _validated_payload_path
from hawavoclean.natural_contract import (
    _probe_optional_runtime_contract,
    load_natural_route_contract,
)
from hawavoclean.server.policy import PathPolicyError
from hawavoclean.server.source_caps import (
    NativeSourceRegistry,
    resolve_native_selected_path,
)

# --- 1. Natural Contract Edge Cases ---


def test_probe_optional_runtime_contract_corrupt_json_sentinel() -> None:
    fake_completed = MagicMock()
    fake_completed.returncode = 1
    fake_completed.stdout = "@@HAWAVOCLEAN_DEPENDENCY_PROBE@@{corrupted-json\n"

    with (
        patch("subprocess.run", return_value=fake_completed),
        pytest.raises(PreflightError, match="optional runtime import/contract failed"),
    ):
        _probe_optional_runtime_contract(
            core_id="test-corrupt-json",
            probe_reference="mod.func",
            required_modules=("test_mod",),
            search_path=("/fake",),
        )


def test_probe_optional_runtime_contract_non_dict_and_non_str_fields() -> None:
    fake_completed = MagicMock()
    fake_completed.returncode = 1
    # Error and detail are not strings
    fake_completed.stdout = '@@HAWAVOCLEAN_DEPENDENCY_PROBE@@{"error": 12345, "detail": ["list"]}\n'

    with (
        patch("subprocess.run", return_value=fake_completed),
        pytest.raises(PreflightError, match="optional runtime import/contract failed"),
    ):
        _probe_optional_runtime_contract(
            core_id="test-non-str",
            probe_reference="mod.func",
            required_modules=("test_mod",),
            search_path=("/fake",),
        )


def test_natural_route_contract_invalid_sample_rates(tmp_path: Path) -> None:
    config = HawaVoCleanConfig(
        enhancement=EnhancementConfig(
            core_id="wiener-dd-48k-v1",
            phase_coherent=True,
            model_sample_rate=48000,
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
            return_value=(
                {"phase_coherent": True, "expected_sample_rates": ["not_int"]},
                "hash123",
            ),
        ),
        patch("hawavoclean.natural_contract.resolve_core"),
        patch("hawavoclean.natural_contract.hash_file", return_value="sha123"),
        pytest.raises(ConfigError, match="Core lockfile expected_sample_rates is invalid"),
    ):
        load_natural_route_contract("natural-clarity", config=config)


# --- 2. Source Caps Edge Cases ---


def test_resolve_native_selected_path_stat_oserror(tmp_path: Path) -> None:
    f = tmp_path / "test.wav"
    f.write_bytes(b"123")

    with (
        patch.object(Path, "stat", side_effect=OSError("permission denied")),
        pytest.raises(PathPolicyError) as exc,
    ):
        resolve_native_selected_path(str(f))
    assert exc.value.status == 404
    assert exc.value.code == "not_found"


def test_native_source_registry_identity_none(tmp_path: Path) -> None:
    reg = NativeSourceRegistry()

    # Directory is not a regular file -> returns None
    assert reg._identity(tmp_path) is None

    # stat raises OSError -> returns None
    f = tmp_path / "file.wav"
    f.write_bytes(b"data")
    with patch.object(Path, "stat", side_effect=OSError("disk error")):
        assert reg._identity(f) is None


def test_native_source_registry_authorizes_cleanup_on_vanished(tmp_path: Path) -> None:
    f = tmp_path / "source.wav"
    f.write_bytes(b"12345")

    reg = NativeSourceRegistry()
    source = reg.register(str(f))
    assert reg.authorizes(f) is True

    # Now remove the file from disk so _valid_locked returns False
    f.unlink()
    assert reg.authorizes(f) is False
    # Registry should have cleaned up the entry
    assert source.source_id not in reg._entries


# --- 3. Model Packs Manifest Edge Cases ---


def test_validate_payload_edge_cases() -> None:
    # 1. Role mismatch
    with pytest.raises(ModelPackManifestError, match="role mismatch"):
        _parse_payload(
            {
                "role": "verifier",
                "path": "models/model.onnx",
                "sha256": "a" * 64,
                "size_bytes": 100,
            },
            role="model",
            include_role=True,
        )

    # 2. Invalid sha256
    with pytest.raises(ModelPackManifestError, match="invalid SHA-256"):
        _parse_payload(
            {"path": "models/model.onnx", "sha256": "bad-hash", "size_bytes": 100},
            role="model",
            include_role=False,
        )

    # 3. Non-positive size_bytes
    with pytest.raises(ModelPackManifestError, match="positive integer"):
        _parse_payload(
            {"path": "models/model.onnx", "sha256": "a" * 64, "size_bytes": 0},
            role="model",
            include_role=False,
        )


def test_validated_payload_path_edge_cases() -> None:
    # 1. Empty string
    with pytest.raises(ModelPackManifestError, match="empty or too long"):
        _validated_payload_path("")

    # 2. Exceeding 512 bytes
    long_path = "a" * 513
    with pytest.raises(ModelPackManifestError, match="empty or too long"):
        _validated_payload_path(long_path)

    # 3. Non-NFC unicode (NFD composed form)
    nfd_str = "e\u0301"  # 'é' in decomposed form
    with pytest.raises(ModelPackManifestError, match="Unicode NFC"):
        _validated_payload_path(nfd_str)
