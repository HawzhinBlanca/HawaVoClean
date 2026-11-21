"""Unit tests for daw-plugin/export_onnx.py causal 960/480 streaming mask export spec."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

_EXPORT_ONNX_PATH = Path(__file__).resolve().parent / "export_onnx.py"


def _load_export_module() -> Any:
    mod_name = "hawavoclean_daw_export_onnx"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, _EXPORT_ONNX_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_streaming_mask_spec_matches_rust_960_480_geometry() -> None:
    mod = _load_export_module()

    assert mod.DAW_SAMPLE_RATE == 48000
    assert mod.DAW_FFT_LEN == 960
    assert mod.DAW_HOP_LEN == 480
    assert mod.DAW_NUM_BINS == 481
    assert mod.DAW_WINDOW == "sqrt_hann"

    export_spec = mod._build_export_spec(
        "deepfilternet3",
        checkpoint_path=None,
        untrained_smoke=True,
        frames=1,
        samples=4096,
    )

    assert export_spec["sample_rate"] == 48000
    assert export_spec["fft_len"] == 960
    assert export_spec["hop_len"] == 480
    assert export_spec["num_bins"] == 481
    assert export_spec["window"] == "sqrt_hann"
    assert export_spec["input_names"] == ["stft_magnitude"]
    assert export_spec["output_names"] == ["gain_mask"]
    assert tuple(export_spec["dummy_input"].shape) == (1, 1, 481)

    module = export_spec["module"]
    with torch.no_grad():
        mask = module(export_spec["dummy_input"])
    assert tuple(mask.shape) == (1, 1, 481)
    assert torch.isfinite(mask).all()
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0


def test_preflight_and_cli_smoke() -> None:
    mod = _load_export_module()
    res = mod.preflight_check()
    assert res.torch_available is True
    assert mod.main(["--preflight-only"]) == 0


@pytest.mark.parametrize("bad_model", ["unknown-model", ""])
def test_unsupported_model_raises_value_error(bad_model: str) -> None:
    mod = _load_export_module()
    with pytest.raises(ValueError, match="Unsupported model"):
        mod._build_export_spec(
            bad_model,
            checkpoint_path=None,
            untrained_smoke=True,
            frames=1,
            samples=4096,
        )
