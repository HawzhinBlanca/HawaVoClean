from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hawavoclean.enhancement.dependency_probe import (
    RuntimeDependencyContractError,
    _probe_deepfilternet_runtime_contract,
    install_torchaudio_compat,
    probe_lowband_runtime_contract,
    probe_studio_runtime_contract,
)


def test_install_torchaudio_compat_already_present() -> None:
    fake_mod = types.ModuleType("torchaudio.backend.common")
    with patch.dict(sys.modules, {"torchaudio.backend.common": fake_mod}):
        install_torchaudio_compat()
        assert sys.modules["torchaudio.backend.common"] is fake_mod


def test_install_torchaudio_compat_audio_metadata_present() -> None:
    fake_mod = types.ModuleType("torchaudio.backend.common")
    fake_mod.AudioMetaData = object  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"torchaudio.backend.common": fake_mod}):
        install_torchaudio_compat()
        assert sys.modules["torchaudio.backend.common"].AudioMetaData is object


def test_install_torchaudio_compat_import_error() -> None:
    with (
        patch.dict(sys.modules),
        patch("builtins.__import__", side_effect=ImportError("No torchaudio")),
        pytest.raises(RuntimeDependencyContractError, match="torchaudio import failed"),
    ):
        sys.modules.pop("torchaudio.backend.common", None)
        install_torchaudio_compat()


def test_install_torchaudio_compat_success() -> None:
    fake_torchaudio = types.ModuleType("torchaudio")
    with patch.dict(sys.modules, clear=False):
        sys.modules.pop("torchaudio.backend.common", None)
        sys.modules.pop("torchaudio.backend", None)
        with patch.dict(sys.modules, {"torchaudio": fake_torchaudio}):
            install_torchaudio_compat()
            assert "torchaudio.backend.common" in sys.modules
            assert hasattr(sys.modules["torchaudio.backend.common"], "AudioMetaData")


def test_probe_deepfilternet_runtime_contract_torch_fail() -> None:
    with (
        patch.dict(sys.modules),
        patch("builtins.__import__", side_effect=ImportError("torch not installed")),
        pytest.raises(RuntimeDependencyContractError, match="torch import failed"),
    ):
        _probe_deepfilternet_runtime_contract()


def test_probe_deepfilternet_runtime_contract_missing_torch_symbol() -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.device = lambda: None  # type: ignore[attr-defined]
    # missing from_numpy and no_grad
    with (
        patch.dict(sys.modules, {"torch": fake_torch}),
        pytest.raises(
            RuntimeDependencyContractError, match="torch.from_numpy is missing or is not callable"
        ),
    ):
        _probe_deepfilternet_runtime_contract()


def test_probe_deepfilternet_runtime_contract_df_import_fail() -> None:
    fake_torch = MagicMock()
    orig_import = __import__

    def custom_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("df"):
            raise ImportError("df failure")
        return orig_import(name, *args, **kwargs)

    with (
        patch.dict(sys.modules, {"torch": fake_torch}),
        patch("hawavoclean.enhancement.dependency_probe.install_torchaudio_compat"),
        patch("builtins.__import__", side_effect=custom_import),
        pytest.raises(RuntimeDependencyContractError, match="DeepFilterNet import failed"),
    ):
        _probe_deepfilternet_runtime_contract()


def test_probe_deepfilternet_runtime_contract_symbols_not_callable() -> None:
    fake_torch = MagicMock()
    fake_df_config = MagicMock()

    for bad_attr in ["init_df", "enhance", "get_device"]:
        fake_df_enhance = MagicMock() if bad_attr != "enhance" else "not_callable"
        fake_init_df = MagicMock() if bad_attr != "init_df" else "not_callable"
        fake_get_device = MagicMock() if bad_attr != "get_device" else "not_callable"

        mock_modules = {
            "torch": fake_torch,
            "df": MagicMock(),
            "df.config": types.SimpleNamespace(config=fake_df_config),
            "df.enhance": types.SimpleNamespace(enhance=fake_df_enhance, init_df=fake_init_df),
            "df.utils": types.SimpleNamespace(get_device=fake_get_device),
        }

        with (
            patch.dict(sys.modules, mock_modules),
            patch("hawavoclean.enhancement.dependency_probe.install_torchaudio_compat"),
            pytest.raises(RuntimeDependencyContractError, match="is missing or is not callable"),
        ):
            _probe_deepfilternet_runtime_contract()


def test_probe_studio_runtime_contract_wpe_not_callable() -> None:
    with (
        patch("hawavoclean.enhancement.dependency_probe._probe_deepfilternet_runtime_contract"),
        patch.dict(
            sys.modules,
            {
                "nara_wpe": MagicMock(),
                "nara_wpe.wpe": types.SimpleNamespace(wpe="not_callable"),
            },
        ),
        pytest.raises(
            RuntimeDependencyContractError, match="nara_wpe.wpe.wpe is missing or is not callable"
        ),
    ):
        probe_studio_runtime_contract()


def test_probe_lowband_runtime_contract() -> None:
    with patch(
        "hawavoclean.enhancement.dependency_probe._probe_deepfilternet_runtime_contract"
    ) as mock_dfn:
        probe_lowband_runtime_contract()
        mock_dfn.assert_called_once()


def test_probe_studio_runtime_contract_nara_fail() -> None:
    with (
        patch("hawavoclean.enhancement.dependency_probe._probe_deepfilternet_runtime_contract"),
        patch.dict(sys.modules),
        patch("builtins.__import__", side_effect=ImportError("nara missing")),
        pytest.raises(RuntimeDependencyContractError, match="nara_wpe import failed"),
    ):
        probe_studio_runtime_contract()


def test_probe_studio_runtime_contract_success() -> None:
    fake_nara = types.SimpleNamespace(wpe=lambda x: x)
    mock_modules = {
        "nara_wpe": MagicMock(),
        "nara_wpe.wpe": fake_nara,
    }
    with (
        patch("hawavoclean.enhancement.dependency_probe._probe_deepfilternet_runtime_contract"),
        patch.dict(sys.modules, mock_modules),
    ):
        probe_studio_runtime_contract()
