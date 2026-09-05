from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hawavoclean.model_packs.errors import (
    ModelPackInstallError,
    ModelPackManifestError,
    ModelPackPayloadError,
    ModelPackSignatureError,
)
from hawavoclean.model_packs.store import (
    _copy_regular_durable,
    _mkdir_owned,
    _require_owned_directory,
)
from hawavoclean.model_packs.trust import (
    ModelPackQualificationPolicy,
    TrustedKey,
    TrustStore,
    _hash_regular_file,
    _read_small_regular,
)
from hawavoclean.runtime import _windows_peak_rss_bytes, process_peak_rss_bytes

# --- 1. Runtime Windows Peak RSS Coverage ---


def test_windows_peak_rss_bytes_mocked() -> None:
    fake_windll = MagicMock()
    fake_kernel32 = MagicMock()
    fake_psapi = MagicMock()

    fake_windll.kernel32 = fake_kernel32
    fake_windll.psapi = fake_psapi

    def fake_get_process_memory_info(_proc: Any, counters_ref: Any, _cb: Any) -> int:
        counters = counters_ref._obj
        counters.PeakWorkingSetSize = 104857600
        return 1

    fake_psapi.GetProcessMemoryInfo.side_effect = fake_get_process_memory_info

    with patch.dict(vars(ctypes), {"windll": fake_windll}):
        peak = _windows_peak_rss_bytes()
        assert peak == 104857600

    # Test error branch: clear side_effect so return_value is used
    fake_psapi.GetProcessMemoryInfo.side_effect = None
    fake_psapi.GetProcessMemoryInfo.return_value = 0
    with (
        patch.dict(vars(ctypes), {"windll": fake_windll}),
        pytest.raises(OSError, match="GetProcessMemoryInfo failed"),
    ):
        _windows_peak_rss_bytes()


def test_process_peak_rss_bytes_windows_dispatch() -> None:
    with (
        patch("sys.platform", "win32"),
        patch("hawavoclean.runtime._windows_peak_rss_bytes", return_value=123456),
    ):
        assert process_peak_rss_bytes() == 123456


# --- 2. Model Packs Trust Coverage ---


def test_trusted_key_validations() -> None:
    # Invalid key_id
    with pytest.raises(ModelPackSignatureError, match="invalid format"):
        TrustedKey(key_id="BAD KEY!", public_key_bytes=b"0" * 32)

    # Invalid length of public_key_bytes
    with pytest.raises(ModelPackSignatureError, match="must contain exactly 32 raw bytes"):
        TrustedKey(key_id="valid_key_01", public_key_bytes=b"short")

    # Invalid revoked type
    with pytest.raises(ModelPackSignatureError, match="revoked flag must be boolean"):
        TrustedKey(key_id="valid_key_01", public_key_bytes=b"0" * 32, revoked="not_bool")  # type: ignore[arg-type]


def test_trust_store_duplicate_keys() -> None:
    k1 = TrustedKey(key_id="key1", public_key_bytes=b"1" * 32)
    k2 = TrustedKey(key_id="key1", public_key_bytes=b"2" * 32)
    with pytest.raises(ModelPackSignatureError, match="duplicate trusted key id"):
        TrustStore([k1, k2])


def test_pack_qualification_policy_validations() -> None:
    # Bad pack_id
    with pytest.raises(ValueError, match="pack_id is invalid"):
        ModelPackQualificationPolicy(
            pack_id="BAD!",
            version="1.0.0",
            manifest_sha256="a" * 64,
            providers=("CPUExecutionProvider",),
        )

    # Bad version
    with pytest.raises(ValueError, match="version is invalid"):
        ModelPackQualificationPolicy(
            pack_id="good_pack",
            version="not.a.version",
            manifest_sha256="a" * 64,
            providers=("CPUExecutionProvider",),
        )

    # Bad manifest_sha256
    with pytest.raises(ValueError, match="manifest_sha256 is invalid"):
        ModelPackQualificationPolicy(
            pack_id="good_pack",
            version="1.0.0",
            manifest_sha256="not_a_sha",
            providers=("CPUExecutionProvider",),
        )

    # Non-tuple providers
    with pytest.raises(ValueError, match="providers must be a canonical tuple"):
        ModelPackQualificationPolicy(
            pack_id="good_pack",
            version="1.0.0",
            manifest_sha256="a" * 64,
            providers=["CPUExecutionProvider"],  # type: ignore[arg-type]
        )

    # Missing CPUExecutionProvider
    with pytest.raises(ValueError, match="must include CPUExecutionProvider"):
        ModelPackQualificationPolicy(
            pack_id="good_pack",
            version="1.0.0",
            manifest_sha256="a" * 64,
            providers=(),
        )


def test_read_limited_file_and_hash_file_errors(tmp_path: Path) -> None:
    # 1. Missing file
    missing = tmp_path / "missing.txt"
    with pytest.raises(
        ModelPackPayloadError, match="required model-pack metadata is missing"
    ) as exc1:
        _read_small_regular(missing, limit=100)
    assert exc1.value.code == "missing_pack_metadata"

    # 2. Too large metadata
    large = tmp_path / "large.txt"
    large.write_bytes(b"A" * 200)
    with pytest.raises(
        ModelPackManifestError, match="model-pack metadata exceeds its safety limit"
    ) as exc2:
        _read_small_regular(large, limit=50)
    assert exc2.value.code == "pack_metadata_too_large"

    # 3. Hash regular file on missing path
    with pytest.raises(ModelPackPayloadError, match="cannot open model-pack payload") as exc3:
        _hash_regular_file(missing)
    assert exc3.value.code == "unreadable_payload"


# --- 3. Model Packs Store Coverage ---


def test_require_owned_directory_errors(tmp_path: Path) -> None:
    missing = tmp_path / "no_dir"
    with pytest.raises(
        ModelPackInstallError, match="cannot inspect model-pack store directory"
    ) as exc:
        _require_owned_directory(missing)
    assert exc.value.code == "unsafe_pack_store"

    regular_file = tmp_path / "file.txt"
    regular_file.write_text("hello", encoding="utf-8")
    with pytest.raises(ModelPackInstallError, match="must be a real directory") as exc:
        _require_owned_directory(regular_file)
    assert exc.value.code == "unsafe_pack_store"


def test_mkdir_owned_oserror(tmp_path: Path) -> None:
    with (
        patch.object(Path, "mkdir", side_effect=OSError("permission denied")),
        pytest.raises(
            ModelPackInstallError, match="cannot create model-pack store directory"
        ) as exc,
    ):
        _mkdir_owned(tmp_path / "cant_create")
    assert exc.value.code == "unwritable_pack_store"


def test_copy_regular_durable_not_reg(tmp_path: Path) -> None:
    dir_source = tmp_path / "dir_src"
    dir_source.mkdir()
    dest = tmp_path / "dest.bin"

    with pytest.raises(ModelPackPayloadError, match="cannot safely copy model-pack file") as exc:
        _copy_regular_durable(dir_source, dest)
    assert exc.value.code == "pack_copy_failed"
