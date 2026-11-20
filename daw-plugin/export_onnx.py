#!/usr/bin/env python3
"""Standalone ONNX exporter and float-parity verifier for HawaVoClean neural models.

Exports HawaVoClean's pure-PyTorch enhancement / dereverberation models to ONNX for
native real-time DAW plugin execution (ONNX Runtime with CoreML / DirectML / CPU EP):

  - ``mossformer2``   : ``MossFormer2SE48K`` (48 kHz STFT mask network, [B, T, 180] -> [B, T, 961])
  - ``deepfilternet3``: ``DeepFilterNet3`` (48 kHz causal streaming 960/480 STFT, 481 bins;
                        ``--untrained-smoke`` exports the exact DAW plugin single-frame mask
                        signature ``[1, 1, 481]`` magnitude -> ``[1, 1, 481]`` gain mask with
                        ``sqrt(Hann)`` COLA windowing)
  - ``gap-urgenet``   : ``GAP-URGENet`` Predictor ``TFGridNetV3`` (16 kHz waveform dereverberation)

Usage examples::

    python3 daw-plugin/export_onnx.py --preflight-only
    python3 daw-plugin/export_onnx.py --model deepfilternet3 --untrained-smoke --frames 1 --output daw-plugin/models/model.onnx --verify
    python3 daw-plugin/export_onnx.py --model mossformer2 --output build/mossformer2.onnx --verify
    python3 daw-plugin/export_onnx.py --model gap-urgenet --output build/gap_urgenet.onnx --verify
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

# Allow running directly from a source checkout without an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Exact causal streaming STFT geometry shared with daw-plugin/src/lib.rs (48 kHz, 20 ms / 10 ms hop).
DAW_SAMPLE_RATE = 48000
DAW_FFT_LEN = 960
DAW_HOP_LEN = 480
DAW_NUM_BINS = DAW_FFT_LEN // 2 + 1  # 481
DAW_WINDOW = "sqrt_hann"


@dataclasses.dataclass(frozen=True)
class PreflightResult:
    """Dependency & environment readiness snapshot for ONNX export and verification."""

    python_version: str
    platform_info: str
    torch_available: bool
    torch_version: str | None
    onnx_available: bool
    onnx_version: str | None
    onnxruntime_available: bool
    onnxruntime_version: str | None
    onnxruntime_providers: tuple[str, ...]
    hawavoclean_available: bool
    can_export: bool
    can_verify: bool
    remediation: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ParityReport:
    """Numerical parity summary comparing PyTorch eager output against ONNX Runtime."""

    passed: bool
    atol: float
    max_abs_diff: float
    mean_abs_diff: float
    rmse: float
    cosine_similarity: float
    output_shape: tuple[int, ...]
    providers_used: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _probe_module_version(module_name: str) -> tuple[bool, str | None]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return False, None
    try:
        mod = importlib.import_module(module_name)
        ver = getattr(mod, "__version__", None)
        return True, str(ver) if ver is not None else "installed"
    except Exception as exc:
        return False, f"import-error: {type(exc).__name__}: {exc}"


def preflight_check() -> PreflightResult:
    """Inspect optional PyTorch / ONNX / ONNX Runtime packages without crashing."""
    torch_ok, torch_ver = _probe_module_version("torch")
    onnx_ok, onnx_ver = _probe_module_version("onnx")
    ort_ok, ort_ver = _probe_module_version("onnxruntime")
    hawa_ok, _ = _probe_module_version("hawavoclean")

    providers: tuple[str, ...] = ()
    if ort_ok:
        try:
            ort = importlib.import_module("onnxruntime")
            providers = tuple(str(p) for p in ort.get_available_providers())
        except Exception:
            providers = ()

    remediation: list[str] = []
    if not torch_ok:
        remediation.append("Install PyTorch: uv pip install torch torchaudio")
    if not onnx_ok:
        remediation.append("Install ONNX exporter: uv pip install onnx>=1.16.0")
    if not ort_ok:
        remediation.append(
            "Install ONNX Runtime for parity verification: uv pip install onnxruntime>=1.19.0"
        )
    if not hawa_ok:
        remediation.append(
            "Run from the HawaVoClean repository root or install editable package: uv pip install -e ."
        )

    return PreflightResult(
        python_version=sys.version.split()[0],
        platform_info=f"{platform.system()} {platform.machine()}",
        torch_available=torch_ok,
        torch_version=torch_ver,
        onnx_available=onnx_ok,
        onnx_version=onnx_ver,
        onnxruntime_available=ort_ok,
        onnxruntime_version=ort_ver,
        onnxruntime_providers=providers,
        hawavoclean_available=hawa_ok,
        can_export=torch_ok and onnx_ok,
        can_verify=torch_ok and ort_ok,
        remediation=tuple(remediation),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_export_spec(
    model_name: str,
    *,
    checkpoint_path: Path | None,
    untrained_smoke: bool,
    frames: int,
    samples: int,
) -> dict[str, Any]:
    """Construct the pure-PyTorch export wrapper, dummy input, and ONNX tensor metadata."""
    import torch
    import torch.nn as nn

    if model_name == "mossformer2":
        arch_mod = importlib.import_module("hawavoclean.enhancement.mossformer2_arch")
        studio_pro = importlib.import_module("hawavoclean.enhancement.studio_pro")

        class MossFormer2OnnxWrapper(nn.Module):
            """Single-tensor ONNX wrapper around HawaVoClean's pure-PyTorch MossFormer2SE48K."""

            def __init__(self, core: nn.Module) -> None:
                super().__init__()
                self.core = core

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                outputs = self.core(features)
                mask: torch.Tensor = outputs[0]
                return mask

        ckpt_sha: str | None = None
        if untrained_smoke:
            # Lightweight 2-block smoke architecture matching exact I/O shapes (180 -> 961)
            core = arch_mod.MossFormer2SE48K(num_blocks=2)
        else:
            core = studio_pro.load_mossformer2_se_48k("cpu", checkpoint_path=checkpoint_path)
            resolved_ckpt = (
                checkpoint_path
                if checkpoint_path is not None
                else studio_pro.studio_pro_checkpoint_path()
            )
            if resolved_ckpt.is_file():
                ckpt_sha = _sha256_file(resolved_ckpt)

        mf2_wrapper: nn.Module = MossFormer2OnnxWrapper(core).eval()
        torch.manual_seed(42)
        dummy_input = torch.randn(1, max(4, frames), 180, dtype=torch.float32)
        return {
            "model_id": "mossformer2-se-48k",
            "module": mf2_wrapper,
            "dummy_input": dummy_input,
            "input_names": ["fbank_features"],
            "output_names": ["stft_mask"],
            "dynamic_axes": {
                "fbank_features": {0: "batch", 1: "frames"},
                "stft_mask": {0: "batch", 1: "frames"},
            },
            "sample_rate": 48000,
            "fft_len": 1920,
            "hop_len": 384,
            "checkpoint_sha256": ckpt_sha,
        }

    if model_name == "deepfilternet3":
        # Streaming ERB/magnitude mask wrapper compatible with 48 kHz DAW frame buffers
        # (n_fft=960, hop=480, 481 frequency bins). Loads DeepFilterNet3 encoder/erb_dec
        # when installed, or falls back to a verifiable deterministic streaming wrapper
        # in --untrained-smoke mode.
        if not untrained_smoke:
            studio_mod = importlib.import_module("hawavoclean.enhancement.studio")
            df_model, _df_state = studio_mod.load_deepfilternet3("cpu")

            class DeepFilterNet3ErbWrapper(nn.Module):
                """Exports DeepFilterNet3 ERB encoder/decoder front-end for real-time masking."""

                def __init__(self, df_net: nn.Module) -> None:
                    super().__init__()
                    self.df_net = df_net

                def forward(
                    self,
                    spec: torch.Tensor,
                    feat_erb: torch.Tensor,
                    feat_spec: torch.Tensor,
                ) -> torch.Tensor:
                    out = self.df_net(spec, feat_erb, feat_spec)
                    enhanced: torch.Tensor = out[0] if isinstance(out, (tuple, list)) else out
                    return enhanced

            df_wrapper: nn.Module = DeepFilterNet3ErbWrapper(df_model).eval()
            torch.manual_seed(42)
            t_frames = max(4, frames)
            dummy_spec = torch.randn(1, 1, t_frames, 481, 2, dtype=torch.float32)
            dummy_erb = torch.randn(1, 1, t_frames, 32, dtype=torch.float32)
            dummy_fspec = torch.randn(1, 1, t_frames, 96, 2, dtype=torch.float32)
            return {
                "model_id": "deepfilternet3-48k",
                "module": df_wrapper,
                "dummy_input": (dummy_spec, dummy_erb, dummy_fspec),
                "input_names": ["spec", "feat_erb", "feat_spec"],
                "output_names": ["enhanced_spec"],
                "dynamic_axes": {
                    "spec": {0: "batch", 2: "frames"},
                    "feat_erb": {0: "batch", 2: "frames"},
                    "feat_spec": {0: "batch", 2: "frames"},
                    "enhanced_spec": {0: "batch", 2: "frames"},
                },
                "sample_rate": 48000,
                "fft_len": 960,
                "hop_len": 480,
                "checkpoint_sha256": None,
            }

        class StreamingErbMaskWrapper(nn.Module):
            """Causal streaming STFT magnitude -> gain mask network matching daw-plugin/src/lib.rs.

            Input : ``stft_magnitude`` float32 tensor of shape ``[B, T, 481]`` (DAW single-frame: ``[1, 1, 481]``)
            Output: ``gain_mask``      float32 tensor of shape ``[B, T, 481]`` in ``[0, 1]``
            STFT  : 48 kHz, ``fft_len=960`` (20 ms), ``hop_len=480`` (10 ms), ``sqrt(Hann)`` COLA window.
            """

            def __init__(self, n_bins: int = DAW_NUM_BINS, hidden: int = 64) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_bins, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, n_bins),
                    nn.Sigmoid(),
                )

            def forward(self, mag_frame: torch.Tensor) -> torch.Tensor:
                mask: torch.Tensor = self.net(mag_frame)
                return mask

        smoke_wrapper: nn.Module = StreamingErbMaskWrapper(n_bins=DAW_NUM_BINS).eval()
        torch.manual_seed(42)
        t_frames = max(1, frames)
        dummy_input = torch.rand(1, t_frames, DAW_NUM_BINS, dtype=torch.float32)
        return {
            "model_id": "deepfilternet3-streaming-mask-48k",
            "module": smoke_wrapper,
            "dummy_input": dummy_input,
            "input_names": ["stft_magnitude"],
            "output_names": ["gain_mask"],
            "dynamic_axes": {
                "stft_magnitude": {0: "batch", 1: "frames"},
                "gain_mask": {0: "batch", 1: "frames"},
            },
            "sample_rate": DAW_SAMPLE_RATE,
            "fft_len": DAW_FFT_LEN,
            "hop_len": DAW_HOP_LEN,
            "num_bins": DAW_NUM_BINS,
            "window": DAW_WINDOW,
            "checkpoint_sha256": None,
        }

    if model_name == "gap-urgenet":
        dereverb_mod = importlib.import_module("hawavoclean.enhancement.dereverb_neural")

        class GapUrgeNetOnnxWrapper(nn.Module):
            """Waveform-in / waveform-out ONNX wrapper for GAP-URGENet TFGridNetV3."""

            def __init__(self, core: nn.Module) -> None:
                super().__init__()
                self.core = core

            def forward(self, waveform: torch.Tensor) -> torch.Tensor:
                out: torch.Tensor = self.core(waveform)
                return out

        ckpt_sha = None
        if untrained_smoke:
            smoke_cfg = {
                "n_layers": 1,
                "lstm_hidden_units": 32,
                "attn_n_head": 2,
                "attn_qk_output_channel": 4,
                "emb_dim": 16,
                "emb_ks": 4,
                "emb_hs": 1,
                "fft_len_s": 0.032,
                "hop_len_s": 0.016,
                "win_len_s": 0.032,
            }
            core = dereverb_mod._build_tfgridnet_v3(smoke_cfg)
        else:
            core = dereverb_mod.load_dereverb_model("cpu", checkpoint_path=checkpoint_path)
            resolved_ckpt = (
                checkpoint_path
                if checkpoint_path is not None
                else dereverb_mod.dereverb_checkpoint_path()
            )
            if resolved_ckpt.is_file():
                ckpt_sha = _sha256_file(resolved_ckpt)

        gap_wrapper: nn.Module = GapUrgeNetOnnxWrapper(core).eval()
        torch.manual_seed(42)
        n_samples = max(1024, samples)
        dummy_input = torch.randn(1, n_samples, dtype=torch.float32) * 0.1
        return {
            "model_id": "gap-urgenet-predictor-16k",
            "module": gap_wrapper,
            "dummy_input": dummy_input,
            "input_names": ["waveform_16k"],
            "output_names": ["dereverbed_16k"],
            "dynamic_axes": {
                "waveform_16k": {0: "batch", 1: "samples"},
                "dereverbed_16k": {0: "batch", 1: "samples"},
            },
            "sample_rate": 16000,
            "fft_len": 512,
            "hop_len": 256,
            "checkpoint_sha256": ckpt_sha,
        }

    raise ValueError(
        f"Unsupported model '{model_name}'. Expected mossformer2, deepfilternet3, or gap-urgenet."
    )


def verify_onnx_parity(
    module: Any,
    dummy_input: Any,
    onnx_path: Path,
    input_names: list[str],
    *,
    atol: float = 1e-4,
) -> ParityReport:
    """Verify float32 output parity between PyTorch eager execution and ONNX Runtime."""
    import numpy as np
    import torch

    ort = importlib.import_module("onnxruntime")

    module.eval()
    with torch.no_grad():
        if isinstance(dummy_input, tuple):
            pt_out = module(*dummy_input)
            inputs_seq = dummy_input
        else:
            pt_out = module(dummy_input)
            inputs_seq = (dummy_input,)

    pt_np = np.asarray(pt_out.detach().cpu().float().numpy(), dtype=np.float32)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_feed = {
        name: np.asarray(tensor.detach().cpu().float().numpy(), dtype=np.float32)
        for name, tensor in zip(input_names, inputs_seq, strict=True)
    }
    ort_out = np.asarray(session.run(None, ort_feed)[0], dtype=np.float32)

    diff = np.abs(pt_np - ort_out)
    max_abs_diff = float(np.max(diff)) if diff.size > 0 else 0.0
    mean_abs_diff = float(np.mean(diff)) if diff.size > 0 else 0.0
    rmse = float(np.sqrt(np.mean(diff * diff))) if diff.size > 0 else 0.0

    pt_flat = pt_np.reshape(-1).astype(np.float64)
    ort_flat = ort_out.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(pt_flat) * np.linalg.norm(ort_flat))
    cosine_sim = float(np.dot(pt_flat, ort_flat) / denom) if denom > 0.0 else 1.0

    passed = math.isfinite(max_abs_diff) and max_abs_diff <= atol and pt_np.shape == ort_out.shape
    return ParityReport(
        passed=passed,
        atol=float(atol),
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        rmse=rmse,
        cosine_similarity=cosine_sim,
        output_shape=tuple(int(d) for d in ort_out.shape),
        providers_used=tuple(str(p) for p in session.get_providers()),
    )


def export_model_to_onnx(
    model_name: str,
    output_path: Path,
    *,
    opset: int = 17,
    checkpoint_path: Path | None = None,
    untrained_smoke: bool = False,
    frames: int = 32,
    samples: int = 4096,
    verify: bool = False,
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Export the requested HawaVoClean model to ONNX and write sidecar metadata."""
    preflight = preflight_check()
    if not preflight.can_export:
        missing = "; ".join(preflight.remediation) or "Missing torch or onnx"
        raise RuntimeError(f"ONNX export preflight failed: {missing}")

    import torch

    onnx = importlib.import_module("onnx")

    spec = _build_export_spec(
        model_name,
        checkpoint_path=checkpoint_path,
        untrained_smoke=untrained_smoke,
        frames=frames,
        samples=samples,
    )

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            spec["module"],
            spec["dummy_input"],
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=spec["input_names"],
            output_names=spec["output_names"],
            dynamic_axes=spec["dynamic_axes"],
        )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    parity: ParityReport | None = None
    if verify:
        if not preflight.can_verify:
            raise RuntimeError(
                "Parity verification requested (--verify) but onnxruntime is not installed. "
                "Install with: uv pip install onnxruntime>=1.19.0"
            )
        parity = verify_onnx_parity(
            spec["module"],
            spec["dummy_input"],
            output_path,
            spec["input_names"],
            atol=atol,
        )
        if not parity.passed:
            raise RuntimeError(
                f"ONNX float parity check failed: max_abs_diff={parity.max_abs_diff:.6g} > atol={atol}"
            )

    meta = {
        "model_id": spec["model_id"],
        "onnx_path": str(output_path),
        "onnx_sha256": _sha256_file(output_path),
        "opset_version": opset,
        "sample_rate": spec["sample_rate"],
        "fft_len": spec["fft_len"],
        "hop_len": spec["hop_len"],
        "num_bins": spec.get("num_bins", spec["fft_len"] // 2 + 1),
        "window": spec.get("window", DAW_WINDOW),
        "input_names": spec["input_names"],
        "output_names": spec["output_names"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "untrained_smoke": untrained_smoke,
        "parity": parity.to_dict() if parity is not None else None,
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_onnx.py",
        description="Export HawaVoClean neural models to ONNX with float-parity verification.",
    )
    parser.add_argument(
        "--model",
        choices=("mossformer2", "deepfilternet3", "gap-urgenet"),
        default="mossformer2",
        help="Model architecture to export (default: mossformer2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("daw-plugin/models/hawavoclean.onnx"),
        help="Destination path for the exported .onnx file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path (overrides HawaVoClean default lookup).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17, CoreML EP compatible).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=32,
        help="Dummy time frames for spectral models (default: 32).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4096,
        help="Dummy waveform samples for waveform models (default: 4096).",
    )
    parser.add_argument(
        "--untrained-smoke",
        action="store_true",
        help="Export a tiny untrained smoke architecture to test the ONNX pipeline without checkpoints.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify float32 parity between PyTorch and ONNX Runtime after export.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Max absolute float32 tolerance for --verify (default: 1e-4).",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Print JSON dependency/environment diagnostics and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    preflight = preflight_check()
    if args.preflight_only:
        print(json.dumps(preflight.to_dict(), indent=2))
        return 0

    if not preflight.can_export:
        print(
            json.dumps(
                {
                    "status": "preflight_failed",
                    "preflight": preflight.to_dict(),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    try:
        meta = export_model_to_onnx(
            args.model,
            args.output,
            opset=args.opset,
            checkpoint_path=args.checkpoint,
            untrained_smoke=args.untrained_smoke,
            frames=args.frames,
            samples=args.samples,
            verify=args.verify,
            atol=args.atol,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"status": "ok", "artifact": meta}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
