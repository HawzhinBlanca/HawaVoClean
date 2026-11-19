#!/usr/bin/env python3
"""Item 4 (2026) Mono Room Dereverberation Shootout on real 48 kHz Sorani Kurdish speech.

Evaluates strictly Apache-2.0 / MIT mono dereverberation & universal speech
enhancement candidates (2024-2026) on the exact 4 stress cases from
``research/benchmarks/shootout.py`` (``clean``, ``reverb_heavy``,
``reverb_plus_noise``, ``noise_street_hum``) across two real 48 kHz Sorani
Kurdish speakers (``sherko_01_10.wav`` and ``malaka_01_102.wav``).

Runs exclusively inside ``.local-ai/bench-env``; never modifies production
dependencies or ``.venv``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
BENCH_MODELS = ROOT / ".local-ai/bench-models"
BENCH_REPOS = ROOT / ".local-ai/bench-repos"
sys.path.insert(0, str(ROOT / "research/benchmarks"))

from dereverb_shootout import MLXModelRunner  # noqa: E402
from shootout import (  # noqa: E402
    SAMPLE_RATE,
    ClipMetrics,
    LavaSRRunner,
    MossFormer2Runner,
    QualityScorer,
    _format_markdown,
    _match_length,
    _normalize_rms,
    _resample,
    build_stress_cases,
    merge_with_protected_band,
)


def _protect_16k_with_lavasr(
    y_16k_as_48k: np.ndarray[Any, np.dtype[np.float32]],
    lavasr: LavaSRRunner,
    sr: int = SAMPLE_RATE,
    cutoff_hz: float = 7200.0,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float]:
    """Extend a 16 kHz dereverberated signal to 48 kHz with LavaSR v2 + protected band."""
    y_lava, rtf_lava = lavasr.process(y_16k_as_48k, sr, denoise=False)
    merged = merge_with_protected_band(y_16k_as_48k, y_lava, sr, cutoff_hz=cutoff_hz)
    return _match_length(merged, len(y_16k_as_48k)), rtf_lava


class Urgent2026BSRNNRunner:
    """ICASSP 2026 URGENT Track 1 Baseline BSRNN (48 kHz native, MIT weights / Apache-2.0 code)."""

    def __init__(self, ckpt_path: Path, device: str) -> None:
        self.ckpt_path = ckpt_path
        self.device = device
        self._model: Any = None
        self._actual_device = device

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch

        if str(BENCH_REPOS / "urgent2026") not in sys.path:
            sys.path.insert(0, str(BENCH_REPOS / "urgent2026"))
        from baseline_code.models.bsrnn import BSRNN_SE

        ckpt = torch.load(str(self.ckpt_path), map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        se_sd = {k.removeprefix("se_model."): v for k, v in sd.items() if k.startswith("se_model.")}
        if not se_sd:
            se_sd = sd

        # Infer num_channel & num_layer directly from checkpoint weights
        num_channel = 196
        num_layer = 6
        if "hyper_parameters" in ckpt and hasattr(
            ckpt["hyper_parameters"].get("cfg"), "model_configs"
        ):
            mc = ckpt["hyper_parameters"]["cfg"].model_configs
            if isinstance(mc, dict):
                num_channel = int(mc.get("num_channel", num_channel))
                num_layer = int(mc.get("num_layer", num_layer))
        else:
            for k, v in se_sd.items():
                if k.endswith("band_split.fc.0.1.weight"):
                    num_channel = int(v.shape[0])
                    break
            layer_ids = [
                int(k.split(".bsrnn.bsrnn.bsrnn.")[1].split(".")[0])
                for k in se_sd
                if ".bsrnn.bsrnn.bsrnn." in k
            ]
            if layer_ids:
                num_layer = max(layer_ids) + 1

        model = BSRNN_SE(num_channel=num_channel, num_layer=num_layer)
        missing, unexpected = model.load_state_dict(se_sd, strict=False)
        if missing or unexpected:
            print(
                f"[bsrnn] state_dict notice: missing={len(missing)} unexpected={len(unexpected)}",
                file=sys.stderr,
            )
        model.eval()
        # Probe requested device (MPS complex STFT may need CPU fallback)
        if self.device != "cpu":
            try:
                model.to(torch.device(self.device))
                with torch.inference_mode():
                    x = torch.zeros(1, 48000, device=self.device)
                    ln = torch.tensor([48000], device=self.device)
                    _ = model(x, ln, 48000)
                if self.device == "mps":
                    torch.mps.synchronize()
                self._actual_device = self.device
            except Exception as exc:
                print(f"[bsrnn] {self.device} probe failed ({exc}); using cpu", file=sys.stderr)
                model.to(torch.device("cpu"))
                self._actual_device = "cpu"
        else:
            self._actual_device = "cpu"
        self._model = model

    @property
    def active_device(self) -> str:
        self._ensure()
        return self._actual_device

    def process(
        self, audio: np.ndarray[Any, np.dtype[np.float32]], sr: int
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float]:
        import torch

        self._ensure()
        dev = torch.device(self._actual_device)
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0).to(dev)
        length = torch.tensor([len(audio)], dtype=torch.long, device=dev)
        if self._actual_device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            enhanced, _ = self._model(wav, length, sr)
        if self._actual_device == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        y = enhanced.squeeze(0).detach().cpu().float().numpy()
        return _match_length(np.asarray(y, dtype=np.float32), len(audio)), dt / (len(audio) / sr)


class GAPURGENetPredictorRunner:
    """GAP-URGENet Predictive branch (TF-GridNetV3 16 kHz, 88 MB, MIT weights / Apache-2.0 code)."""

    def __init__(self, ckpt_path: Path, device: str) -> None:
        self.ckpt_path = ckpt_path
        self.device = device
        self._model: Any = None
        self._actual_device = device

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch

        if str(BENCH_REPOS / "gap-urgenet") not in sys.path:
            sys.path.insert(0, str(BENCH_REPOS / "gap-urgenet"))
        from models.predictor.tfgridnet import TFGridNet

        ckpt = torch.load(str(self.ckpt_path), map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        model = TFGridNet(**cfg)
        sd = ckpt["generator"] if "generator" in ckpt else ckpt["model"]
        model.load_state_dict(sd, strict=False)
        model.eval()
        if self.device != "cpu":
            try:
                model.to(torch.device(self.device))
                with torch.inference_mode():
                    _ = model(torch.zeros(1, 32000, device=self.device), sr=16000)
                if self.device == "mps":
                    torch.mps.synchronize()
                self._actual_device = self.device
            except Exception as exc:
                print(
                    f"[gap_predictor] {self.device} probe failed ({exc}); using cpu",
                    file=sys.stderr,
                )
                model.to(torch.device("cpu"))
                self._actual_device = "cpu"
        else:
            self._actual_device = "cpu"
        self._model = model

    @property
    def active_device(self) -> str:
        self._ensure()
        return self._actual_device

    def process(
        self, audio: np.ndarray[Any, np.dtype[np.float32]], sr: int
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float]:
        import torch

        self._ensure()
        dev = torch.device(self._actual_device)
        a16 = _resample(audio, sr, 16000)
        x = torch.from_numpy(np.ascontiguousarray(a16, dtype=np.float32)).unsqueeze(0).to(dev)
        if self._actual_device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            y16_t = self._model(x, sr=16000)
        if self._actual_device == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        y16 = y16_t.squeeze(0).detach().cpu().float().numpy()
        y48 = _resample(np.asarray(y16, dtype=np.float32), 16000, sr)
        return _match_length(y48, len(audio)), dt / (len(audio) / sr)


class GAPURGENetFullRunner:
    """ICASSP 2026 URGENT #1 GAP-URGENet (full generative+predictive+48kHz PostNet, MIT)."""

    def __init__(self, model_dir: Path, device: str) -> None:
        self.model_dir = model_dir
        self.device = device
        self._model: Any = None
        self._actual_device = device

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch

        if str(BENCH_REPOS / "gap-urgenet") not in sys.path:
            sys.path.insert(0, str(BENCH_REPOS / "gap-urgenet"))

        # PyTorch 2.15 defaults weights_only=True; GAP-URGENet checkpoints store cfg dicts.
        orig_load = torch.load

        def _compat_load(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("weights_only", False)
            return orig_load(*args, **kwargs)

        torch.load = _compat_load  # type: ignore[assignment]
        try:
            from models.gap_urgenet import GAP_URGENet

            model = GAP_URGENet(
                predictor_ckpt_path=str(self.model_dir / "Predictor.pt"),
                dewavlm_ckpt_path=str(self.model_dir / "DeWavLM-Omni.pt"),
                adapter_ckpt_path=str(self.model_dir / "Adapter.pt"),
                vocoder_ckpt_path=str(self.model_dir / "Vocoder.pt"),
                postnet_ckpt_path=str(self.model_dir / "PostNet.pt"),
            )
        finally:
            torch.load = orig_load  # type: ignore[assignment]

        model.eval()
        if self.device != "cpu":
            try:
                model.to(torch.device(self.device))
                with torch.inference_mode():
                    _ = model(
                        torch.zeros(1, 24000, device=self.device),
                        sr_in=48000,
                        sr_out=48000,
                        enable_plc=False,
                    )
                self._actual_device = self.device
            except Exception as exc:
                print(
                    f"[gap_urgenet] {self.device} probe failed ({exc}); using cpu",
                    file=sys.stderr,
                )
                model.to(torch.device("cpu"))
                self._actual_device = "cpu"
        else:
            self._actual_device = "cpu"
        self._model = model

    @property
    def active_device(self) -> str:
        self._ensure()
        return self._actual_device

    def process(
        self, audio: np.ndarray[Any, np.dtype[np.float32]], sr: int
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float]:
        import torch

        self._ensure()
        dev = torch.device(self._actual_device)
        x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0).to(dev)
        if self._actual_device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            # Process in <= 8s overlapping chunks if longer than 10s (matches official inference_long)
            if len(audio) <= sr * 12:
                y_t = self._model(x, sr_in=sr, sr_out=sr, enable_plc=False)
                y = y_t.squeeze(0).detach().cpu().float().numpy()
            else:
                seg = sr * 8
                hop = sr * 4
                out = np.zeros(len(audio), dtype=np.float32)
                wsum = np.zeros(len(audio), dtype=np.float32)
                win = np.hanning(seg).astype(np.float32)
                for start in range(0, len(audio), hop):
                    end = min(len(audio), start + seg)
                    chunk = x[:, start:end]
                    yc = (
                        self._model(chunk, sr_in=sr, sr_out=sr, enable_plc=False)
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .float()
                        .numpy()
                    )
                    w = win[: len(yc)] if len(yc) == seg else np.ones(len(yc), dtype=np.float32)
                    out[start : start + len(yc)] += yc * w
                    wsum[start : start + len(yc)] += w
                    if end >= len(audio):
                        break
                y = out / np.maximum(wsum, 1e-6)
        if self._actual_device == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        return _match_length(np.asarray(y, dtype=np.float32), len(audio)), dt / (len(audio) / sr)


class ZipEnhancerONNXRunner:
    """Alibaba Tongyi Lab ZipEnhancer 16 kHz (Apache-2.0 portable ONNX)."""

    def __init__(self, onnx_path: Path) -> None:
        self.onnx_path = onnx_path
        self._sess: Any = None

    def _ensure(self) -> None:
        if self._sess is not None:
            return
        import onnxruntime as ort

        self._sess = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])

    def _run_chunk_16k(
        self, chunk: np.ndarray[Any, np.dtype[np.float32]]
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        import torch

        rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
        if rms < 1e-7:
            return chunk.copy()
        normed = torch.from_numpy((chunk / rms).astype(np.float32)).unsqueeze(0)
        win = torch.hann_window(400, periodic=True)
        spec = torch.stft(
            normed,
            n_fft=400,
            hop_length=100,
            win_length=400,
            window=win,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        real, imag = spec.real, spec.imag
        mag = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-9).pow(0.3)
        pha = torch.atan2(imag + 1e-10, real + 1e-5)
        amp_g, pha_g = self._sess.run(
            None,
            {
                "noisy_mag": mag.numpy().astype(np.float32),
                "noisy_pha": pha.numpy().astype(np.float32),
            },
        )
        amp_t = torch.from_numpy(amp_g).clamp(min=0.0).pow(1.0 / 0.3)
        pha_t = torch.from_numpy(pha_g)
        comp = torch.complex(amp_t * torch.cos(pha_t), amp_t * torch.sin(pha_t))
        wav = torch.istft(
            comp,
            n_fft=400,
            hop_length=100,
            win_length=400,
            window=win,
            center=True,
            length=len(chunk),
        )
        return (wav.squeeze(0).numpy() * rms).astype(np.float32)

    def process(
        self, audio: np.ndarray[Any, np.dtype[np.float32]], sr: int
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float]:
        self._ensure()
        a16 = _resample(audio, sr, 16000)
        t0 = time.perf_counter()
        # Model card: valid up to 6 s (961 frames) per block; use 4s chunks with 1s crossfade
        seg = 16000 * 4
        overlap = 16000 * 1
        hop = seg - overlap
        if len(a16) <= 16000 * 5:
            y16 = self._run_chunk_16k(a16)
        else:
            out = np.zeros(len(a16), dtype=np.float32)
            pos = 0
            covered = 0
            while pos < len(a16):
                end = min(len(a16), pos + seg)
                chunk_out = self._run_chunk_16k(a16[pos:end])
                xfade = max(0, min(covered - pos, end - pos)) if pos > 0 else 0
                if xfade > 0:
                    fade_in = 0.5 * (
                        1.0
                        - np.cos(np.linspace(0.0, np.pi, xfade, endpoint=False, dtype=np.float32))
                    )
                    out[pos : pos + xfade] = (
                        out[pos : pos + xfade] * (1.0 - fade_in) + chunk_out[:xfade] * fade_in
                    )
                    out[pos + xfade : end] = chunk_out[xfade:]
                else:
                    out[pos:end] = chunk_out
                covered = end
                if end >= len(a16):
                    break
                pos += hop
            y16 = out
        dt = time.perf_counter() - t0
        y48 = _resample(y16, 16000, sr)
        return _match_length(y48, len(audio)), dt / (len(audio) / sr)


class MPSENetRunner:
    """MP-SENet-DNS 16 kHz (MIT)."""

    def __init__(self, repo_id: str, device: str) -> None:
        self.repo_id = repo_id
        self.device = device
        self._model: Any = None
        self._actual_device = device

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch
        from MPSENet import MPSENet

        model = MPSENet.from_pretrained(self.repo_id)
        model.eval()
        if self.device != "cpu":
            try:
                model.to(torch.device(self.device))
                _ = model(np.zeros(4000, dtype=np.float32))
                self._actual_device = self.device
            except Exception as exc:
                print(f"[mpsenet] {self.device} probe failed ({exc}); using cpu", file=sys.stderr)
                model.to(torch.device("cpu"))
                self._actual_device = "cpu"
        else:
            self._actual_device = "cpu"
        self._model = model

    @property
    def active_device(self) -> str:
        self._ensure()
        return self._actual_device

    def process(
        self, audio: np.ndarray[Any, np.dtype[np.float32]], sr: int
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float]:
        import torch

        self._ensure()
        a16 = _resample(audio, sr, 16000)
        if self._actual_device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        y16, _, _ = self._model(a16)
        if self._actual_device == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        y48 = _resample(np.asarray(y16, dtype=np.float32), 16000, sr)
        return _match_length(y48, len(audio)), dt / (len(audio) / sr)


class FoxJoyONNXRunner:
    """Direct onnxruntime runner for UVR Reverb_HQ_By_FoxJoy.onnx (MIT).

    Avoids mlx-audio-separator's buggy MDX ONNX->MLX weight conversion (which
    left 256/376 parameters untrained) and runs the exact UVR MDX-Net STFT/ISTFT
    contract (n_fft=6144, hop=1024, dim_f=3072, dim_t=512, compensate=1.035,
    primary_stem='Reverb' -> dry = input - reverb * compensate).
    """

    def __init__(self, onnx_path: Path) -> None:
        self.onnx_path = onnx_path
        self._sess: Any = None
        self.n_fft = 6144
        self.hop = 1024
        self.dim_f = 3072
        self.dim_t = 512
        self.compensate = 1.035
        self.sr_internal = 44100

    def _ensure(self) -> None:
        if self._sess is not None:
            return
        import onnxruntime as ort

        self._sess = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])

    def _run_chunk(
        self, stereo_chunk: np.ndarray[Any, np.dtype[np.float32]]
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        import torch

        # stereo_chunk: (2, chunk_size) where chunk_size = hop * (dim_t - 1)
        x = torch.from_numpy(stereo_chunk)
        win = torch.hann_window(self.n_fft, periodic=True)
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.n_fft,
            window=win,
            center=True,
            return_complex=True,
        )  # (2, 3073, 512)
        spec_ri = torch.view_as_real(stft[:, : self.dim_f, : self.dim_t])  # (2, 3072, 512, 2)
        # Pack to (1, 4, dim_f, dim_t): [L_re, L_im, R_re, R_im]
        inp = (
            spec_ri.permute(0, 3, 1, 2)
            .reshape(1, 4, self.dim_f, self.dim_t)
            .numpy()
            .astype(np.float32)
        )
        out = self._sess.run(None, {"input": inp})[0]  # (1, 4, dim_f, dim_t)
        out_t = torch.from_numpy(out).reshape(2, 2, self.dim_f, self.dim_t).permute(0, 2, 3, 1)
        out_c = torch.complex(out_t[..., 0], out_t[..., 1])
        # Restore Nyquist bin (zeros)
        nyq = torch.zeros((2, 1, self.dim_t), dtype=out_c.dtype)
        full_c = torch.cat([out_c, nyq], dim=1)
        reverb = torch.istft(
            full_c,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.n_fft,
            window=win,
            center=True,
            length=stereo_chunk.shape[-1],
        )
        return reverb.numpy().astype(np.float32)

    def process(
        self, audio: np.ndarray[Any, np.dtype[np.float32]], sr: int
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float]:
        self._ensure()
        a44 = _resample(audio, sr, self.sr_internal)
        n = len(a44)
        trim = self.n_fft // 2
        chunk_size = self.hop * (self.dim_t - 1)
        gen_size = chunk_size - 2 * trim
        pad_end = gen_size + trim - (n % gen_size)
        stereo = np.stack([a44, a44], axis=0)
        padded = np.pad(stereo, ((0, 0), (trim, pad_end)), mode="constant")

        t0 = time.perf_counter()
        reverb_pieces: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        pos = 0
        while pos + chunk_size <= padded.shape[-1]:
            chunk = padded[:, pos : pos + chunk_size]
            rev_c = self._run_chunk(chunk)
            reverb_pieces.append(rev_c[:, trim:-trim])
            pos += gen_size
            if pos >= n + trim:
                break
        dt = time.perf_counter() - t0
        reverb_full = np.concatenate(reverb_pieces, axis=-1)[:, :n]
        dry_stereo = stereo - reverb_full * self.compensate
        dry_mono = np.mean(dry_stereo, axis=0).astype(np.float32)
        y48 = _resample(dry_mono, self.sr_internal, sr)
        return _match_length(y48, len(audio)), dt / (len(audio) / sr)


class ClearVoice16KRunner:
    """ClearerVoice-Studio 16 kHz runner (FRCRN_SE_16K / MossFormerGAN_SE_16K, Apache-2.0)."""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self._cv: Any = None
        self._actual_device = device

    def _ensure(self) -> None:
        if self._cv is not None:
            return
        import clearvoice
        import torch

        self._cv = clearvoice.ClearVoice(task="speech_enhancement", model_names=[self.model_name])
        if self.device != "cpu":
            try:
                for mw in getattr(self._cv, "models", []):
                    net = getattr(mw, "model", None)
                    if net is not None and hasattr(net, "to"):
                        net.to(torch.device(self.device))
                    if hasattr(mw, "device"):
                        mw.device = torch.device(self.device)
                self._actual_device = self.device
            except Exception as exc:
                print(
                    f"[{self.model_name}] {self.device} move failed ({exc}); using cpu",
                    file=sys.stderr,
                )
                self._actual_device = "cpu"
        else:
            self._actual_device = "cpu"

    @property
    def active_device(self) -> str:
        self._ensure()
        return self._actual_device

    def process(
        self, audio: np.ndarray[Any, np.dtype[np.float32]], sr: int
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float]:
        self._ensure()
        a16 = _resample(audio, sr, 16000)
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "in.wav"
            sf.write(inp, a16, 16000, subtype="FLOAT")
            t0 = time.perf_counter()
            est = self._cv(input_path=str(inp), online_write=False)
            dt = time.perf_counter() - t0
            if isinstance(est, dict):
                est = next(iter(est.values()))
            y16 = np.asarray(est, dtype=np.float32).squeeze()
            if y16.ndim > 1:
                y16 = y16[0]
            y48 = _resample(y16, 16000, sr)
            return _match_length(y48, len(audio)), dt / (len(audio) / sr)


def _evaluate_gates(rows: list[ClipMetrics]) -> str:
    """Evaluate every engine against the 5 hard Item-4 production gates."""
    by_engine: dict[str, dict[str, list[ClipMetrics]]] = {}
    for r in rows:
        by_engine.setdefault(r.engine, {}).setdefault(r.case, []).append(r)

    def _mean(items: list[ClipMetrics], attr: str) -> float | None:
        vals = [getattr(i, attr) for i in items if getattr(i, attr) is not None]
        clean = [float(v) for v in vals if math.isfinite(float(v))]
        return sum(clean) / len(clean) if clean else None

    lines = [
        "## Hard Production Gate Evaluation",
        "",
        "Gates: `reverb_heavy` (NISQA >= 3.20, ESTOI >= 0.62, WavLM >= 0.72) | "
        "`reverb_plus_noise` (NISQA >= 3.20, WavLM >= 0.64) | "
        "`clean` (WavLM >= 0.985, ESTOI >= 0.98) | `RTF <= 0.35`",
        "",
        "| Engine | Clean WavLM / ESTOI | Reverb Heavy MOS / ESTOI / WavLM | Reverb+Noise MOS / WavLM | Max RTF | Verdict |",
        "| :--- | :--- | :--- | :--- | ---: | :--- |",
    ]

    for eng in sorted(by_engine):
        c_map = by_engine[eng]
        c_wl = _mean(c_map.get("clean", []), "wavlm_phonetic_cos")
        c_es = _mean(c_map.get("clean", []), "estoi")
        rh_mos = _mean(c_map.get("reverb_heavy", []), "nisqa_mos")
        rh_es = _mean(c_map.get("reverb_heavy", []), "estoi")
        rh_wl = _mean(c_map.get("reverb_heavy", []), "wavlm_phonetic_cos")
        rn_mos = _mean(c_map.get("reverb_plus_noise", []), "nisqa_mos")
        rn_wl = _mean(c_map.get("reverb_plus_noise", []), "wavlm_phonetic_cos")
        all_rtfs = [i.rtf for lst in c_map.values() for i in lst]
        max_rtf = max(all_rtfs) if all_rtfs else 0.0

        fails: list[str] = []
        if c_wl is None or c_wl < 0.985:
            fails.append("clean_wavlm")
        if c_es is None or c_es < 0.98:
            fails.append("clean_estoi")
        if rh_mos is None or rh_mos < 3.20:
            fails.append("rh_mos")
        if rh_es is None or rh_es < 0.62:
            fails.append("rh_estoi")
        if rh_wl is None or rh_wl < 0.72:
            fails.append("rh_wavlm")
        if rn_mos is None or rn_mos < 3.20:
            fails.append("rn_mos")
        if rn_wl is None or rn_wl < 0.64:
            fails.append("rn_wavlm")
        if max_rtf > 0.35:
            fails.append("rtf")

        verdict = "PASS" if not fails else f"FAIL ({', '.join(fails)})"

        def _f(v: float | None, fmt: str) -> str:
            return format(v, fmt) if v is not None else "—"

        lines.append(
            f"| **{eng}** | {_f(c_wl, '.4f')} / {_f(c_es, '.3f')} | "
            f"{_f(rh_mos, '.2f')} / {_f(rh_es, '.3f')} / {_f(rh_wl, '.4f')} | "
            f"{_f(rn_mos, '.2f')} / {_f(rn_wl, '.4f')} | {max_rtf:.3f} | {verdict} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["base", "bsrnn", "gap_pred", "clearvoice16k", "zip", "mpsenet", "uvr", "gap_full"],
        help="Candidate groups to run and merge into the scorecard.",
    )
    parser.add_argument("--fresh", action="store_true", help="Ignore prior scorecard JSON.")
    args = parser.parse_args()
    groups = set(args.groups)

    accel = "mps" if torch.backends.mps.is_available() else "cpu"
    kurdish_files = sorted((ROOT / "data/kurdish_speakers").glob("*/*.wav"))
    selected = [kurdish_files[0], kurdish_files[-1]]
    out_dir = ROOT / ".local-ai/bench-results"
    audio_dir = out_dir / "audio_dereverb_2026"
    audio_dir.mkdir(parents=True, exist_ok=True)
    scorecard_json = out_dir / "dereverb_2026_scorecard.json"
    scorecard_md = out_dir / "dereverb_2026_scorecard.md"

    existing: list[ClipMetrics] = []
    if not args.fresh and scorecard_json.is_file():
        existing = [
            ClipMetrics(**r) for r in json.loads(scorecard_json.read_text(encoding="utf-8"))
        ]

    scorer = QualityScorer(device=accel)
    mf2 = MossFormer2Runner("speech_enhancement", "MossFormer2_SE_48K", device=accel)
    lavasr = LavaSRRunner(device=accel)
    if groups - {"base", "bsrnn", "gap_full", "uvr"}:
        # Warm up LavaSR v2 once so BWE timings reflect steady-state execution
        lavasr.process(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE, denoise=False)
        if accel == "mps":
            torch.mps.synchronize()

    bsrnn_runner = (
        Urgent2026BSRNNRunner(BENCH_MODELS / "urgent2026/bsrnn.ckpt", device=accel)
        if "bsrnn" in groups and (BENCH_MODELS / "urgent2026/bsrnn.ckpt").is_file()
        else None
    )
    gap_pred_runner = (
        GAPURGENetPredictorRunner(BENCH_MODELS / "gap-urgenet/Predictor.pt", device=accel)
        if "gap_pred" in groups and (BENCH_MODELS / "gap-urgenet/Predictor.pt").is_file()
        else None
    )
    gap_full_needed = (
        "DeWavLM-Omni.pt",
        "Adapter.pt",
        "Vocoder.pt",
        "Predictor.pt",
        "PostNet.pt",
    )
    gap_full_runner = (
        GAPURGENetFullRunner(BENCH_MODELS / "gap-urgenet", device=accel)
        if "gap_full" in groups
        and all((BENCH_MODELS / "gap-urgenet" / f).is_file() for f in gap_full_needed)
        else None
    )
    frcrn_runner = (
        ClearVoice16KRunner("FRCRN_SE_16K", device=accel)
        if "clearvoice16k" in groups
        and (ROOT / "checkpoints/FRCRN_SE_16K/last_best_checkpoint.pt").is_file()
        else None
    )
    mfgan_runner = (
        ClearVoice16KRunner("MossFormerGAN_SE_16K", device=accel)
        if "clearvoice16k" in groups
        and (ROOT / "checkpoints/MossFormerGAN_SE_16K/last_best_checkpoint.pt").is_file()
        else None
    )
    zip_runner = (
        ZipEnhancerONNXRunner(BENCH_MODELS / "zipenhancer/zipenhancer.onnx")
        if "zip" in groups and (BENCH_MODELS / "zipenhancer/zipenhancer.onnx").is_file()
        else None
    )
    mpsenet_runner = (
        MPSENetRunner("JacobLinCool/MP-SENet-DNS", device=accel) if "mpsenet" in groups else None
    )
    foxjoy_path = Path.home() / ".cache/mlx-audio-separator/models/Reverb_HQ_By_FoxJoy.onnx"
    foxjoy_runner = (
        FoxJoyONNXRunner(foxjoy_path) if "uvr" in groups and foxjoy_path.is_file() else None
    )
    uvr_deecho_runner = (
        MLXModelRunner("UVR-DeEcho-DeReverb.pth", ("_(no reverb)_", "_(noreverb)_"))
        if "uvr" in groups
        else None
    )

    target_cases = ("clean", "reverb_heavy", "reverb_plus_noise", "noise_street_hum")
    new_rows: list[ClipMetrics] = []
    engines_run: set[str] = set()

    def _record(
        case_name: str,
        clip_name: str,
        engine: str,
        device: str,
        rtf: float,
        clean: np.ndarray[Any, np.dtype[np.float32]],
        degraded: np.ndarray[Any, np.dtype[np.float32]],
        est: np.ndarray[Any, np.dtype[np.float32]],
    ) -> None:
        engines_run.add(engine)
        sf.write(
            audio_dir / f"{clip_name}__{case_name}__{engine}.wav",
            _normalize_rms(est),
            SAMPLE_RATE,
            subtype="PCM_24",
        )
        m = scorer.score(case_name, clip_name, engine, device, rtf, clean, degraded, est)
        new_rows.append(m)
        print(
            f"  [{engine}] dev={device} rtf={m.rtf:.3f} mos={m.nisqa_mos} "
            f"estoi={m.estoi} wavlm={m.wavlm_phonetic_cos} sisdr={m.si_sdr_db}",
            flush=True,
        )

    for clip_idx, clip_path in enumerate(selected):
        raw, sr = sf.read(clip_path, dtype="float32")
        if raw.ndim == 2:
            raw = np.mean(raw, axis=1)
        clean = _resample(np.asarray(raw, dtype=np.float32), int(sr), SAMPLE_RATE)
        cases = build_stress_cases(clean, SAMPLE_RATE, seed=100 + clip_idx * 17)
        clip_name = clip_path.stem

        for case_name in target_cases:
            degraded = cases[case_name]
            print(f"\n=== {clip_name} :: {case_name} ===", flush=True)
            y_mf2, rtf_mf2 = mf2.process(degraded, SAMPLE_RATE)

            if "base" in groups:
                _record(case_name, clip_name, "00_degraded", "none", 0.0, clean, degraded, degraded)
                _record(
                    case_name,
                    clip_name,
                    "01_mf2_48k_baseline",
                    accel,
                    rtf_mf2,
                    clean,
                    degraded,
                    y_mf2,
                )

            # Candidate 2: ICASSP 2026 URGENT Track 1 BSRNN (48 kHz native, MIT/Apache-2.0)
            if bsrnn_runner is not None:
                try:
                    y_b, rtf_b = bsrnn_runner.process(degraded, SAMPLE_RATE)
                    dev_b = bsrnn_runner.active_device
                    _record(
                        case_name,
                        clip_name,
                        "10_urgent26_bsrnn_48k",
                        dev_b,
                        rtf_b,
                        clean,
                        degraded,
                        y_b,
                    )
                    if case_name in {"reverb_heavy", "reverb_plus_noise"}:
                        y_mb, rtf_mb = bsrnn_runner.process(y_mf2, SAMPLE_RATE)
                        _record(
                            case_name,
                            clip_name,
                            "10b_mf2_then_urgent26_bsrnn",
                            f"{accel}+{dev_b}",
                            rtf_mf2 + rtf_mb,
                            clean,
                            degraded,
                            y_mb,
                        )
                except Exception as exc:
                    print(f"  [10_urgent26_bsrnn_48k] ERROR: {exc}", file=sys.stderr)

            # Candidate 1a: GAP-URGENet Predictive branch (TF-GridNetV3 16 kHz, MIT/Apache-2.0)
            if gap_pred_runner is not None:
                try:
                    y_p, rtf_p = gap_pred_runner.process(degraded, SAMPLE_RATE)
                    dev_p = gap_pred_runner.active_device
                    _record(
                        case_name,
                        clip_name,
                        "11a_gap_predictor_tfgridnet_16k",
                        dev_p,
                        rtf_p,
                        clean,
                        degraded,
                        y_p,
                    )
                    y_pl, rtf_pl = _protect_16k_with_lavasr(y_p, lavasr)
                    _record(
                        case_name,
                        clip_name,
                        "11b_gap_predictor_plus_lavasr_pb",
                        f"{dev_p}+{accel}",
                        rtf_p + rtf_pl,
                        clean,
                        degraded,
                        y_pl,
                    )
                    if case_name in {"reverb_heavy", "reverb_plus_noise"}:
                        y_mp, rtf_mp = gap_pred_runner.process(y_mf2, SAMPLE_RATE)
                        y_mpl, rtf_mpl = _protect_16k_with_lavasr(y_mp, lavasr)
                        _record(
                            case_name,
                            clip_name,
                            "11c_mf2_then_gap_pred_lavasr",
                            f"{accel}+{dev_p}+{accel}",
                            rtf_mf2 + rtf_mp + rtf_mpl,
                            clean,
                            degraded,
                            y_mpl,
                        )
                except Exception as exc:
                    print(f"  [11_gap_predictor] ERROR: {exc}", file=sys.stderr)

            # Candidate 1b: Full GAP-URGENet (ICASSP 2026 URGENT #1, 48 kHz, MIT)
            if gap_full_runner is not None:
                try:
                    y_g, rtf_g = gap_full_runner.process(degraded, SAMPLE_RATE)
                    dev_g = gap_full_runner.active_device
                    _record(
                        case_name,
                        clip_name,
                        "12a_gap_urgenet_full_48k",
                        dev_g,
                        rtf_g,
                        clean,
                        degraded,
                        y_g,
                    )
                    if case_name in {"reverb_heavy", "reverb_plus_noise"}:
                        y_mg, rtf_mg = gap_full_runner.process(y_mf2, SAMPLE_RATE)
                        _record(
                            case_name,
                            clip_name,
                            "12b_mf2_then_gap_urgenet_full",
                            f"{accel}+{dev_g}",
                            rtf_mf2 + rtf_mg,
                            clean,
                            degraded,
                            y_mg,
                        )
                except Exception as exc:
                    print(f"  [12_gap_urgenet_full] ERROR: {exc}", file=sys.stderr)

            # Candidate 3: ClearerVoice 16 kHz (FRCRN_SE_16K & MossFormerGAN_SE_16K, Apache-2.0) + LavaSR
            for tag, runner in (
                ("13_frcrn16k", frcrn_runner),
                ("14_mfgan16k", mfgan_runner),
            ):
                if runner is None:
                    continue
                try:
                    y16, rtf16 = runner.process(degraded, SAMPLE_RATE)
                    dev16 = runner.active_device
                    y48_pb, rtf_l = _protect_16k_with_lavasr(y16, lavasr)
                    _record(
                        case_name,
                        clip_name,
                        f"{tag}_plus_lavasr_pb",
                        f"{dev16}+{accel}",
                        rtf16 + rtf_l,
                        clean,
                        degraded,
                        y48_pb,
                    )
                    if case_name == "reverb_plus_noise":
                        y16_m, rtf16_m = runner.process(y_mf2, SAMPLE_RATE)
                        y48_m, rtf_lm = _protect_16k_with_lavasr(y16_m, lavasr)
                        _record(
                            case_name,
                            clip_name,
                            f"{tag}_after_mf2_lavasr_pb",
                            f"{accel}+{dev16}+{accel}",
                            rtf_mf2 + rtf16_m + rtf_lm,
                            clean,
                            degraded,
                            y48_m,
                        )
                except Exception as exc:
                    print(f"  [{tag}] ERROR: {exc}", file=sys.stderr)

            # Candidate 4a: ZipEnhancer 16 kHz ONNX (Apache-2.0) + LavaSR v2
            if zip_runner is not None:
                try:
                    y_z, rtf_z = zip_runner.process(degraded, SAMPLE_RATE)
                    y_zl, rtf_zl = _protect_16k_with_lavasr(y_z, lavasr)
                    _record(
                        case_name,
                        clip_name,
                        "15_zipenhancer_plus_lavasr_pb",
                        f"cpu+{accel}",
                        rtf_z + rtf_zl,
                        clean,
                        degraded,
                        y_zl,
                    )
                except Exception as exc:
                    print(f"  [15_zipenhancer] ERROR: {exc}", file=sys.stderr)

            # Candidate 4b: MP-SENet-DNS 16 kHz (MIT) + LavaSR v2
            if mpsenet_runner is not None:
                try:
                    y_m, rtf_m = mpsenet_runner.process(degraded, SAMPLE_RATE)
                    dev_m = mpsenet_runner.active_device
                    y_ml, rtf_ml = _protect_16k_with_lavasr(y_m, lavasr)
                    _record(
                        case_name,
                        clip_name,
                        "16_mpsenet_dns_plus_lavasr_pb",
                        f"{dev_m}+{accel}",
                        rtf_m + rtf_ml,
                        clean,
                        degraded,
                        y_ml,
                    )
                except Exception as exc:
                    print(f"  [16_mpsenet_dns] ERROR: {exc}", file=sys.stderr)

            # Candidate 4c/d: UVR MIT Dereverb models (FoxJoy ONNX + UVR-DeEcho-DeReverb)
            for tag, uvr_r, uvr_dev in (
                ("17a_uvr_foxjoy_reverb_hq", foxjoy_runner, "onnx-cpu"),
                ("18a_uvr_deecho_dereverb", uvr_deecho_runner, "mlx"),
            ):
                if uvr_r is None:
                    continue
                try:
                    y_u, rtf_u = uvr_r.process(degraded, SAMPLE_RATE)
                    _record(case_name, clip_name, tag, uvr_dev, rtf_u, clean, degraded, y_u)
                    if case_name in {"reverb_heavy", "reverb_plus_noise"}:
                        y_mu, rtf_mu = uvr_r.process(y_mf2, SAMPLE_RATE)
                        _record(
                            case_name,
                            clip_name,
                            tag.replace("a_", "b_mf2_then_"),
                            f"{accel}+{uvr_dev}",
                            rtf_mf2 + rtf_mu,
                            clean,
                            degraded,
                            y_mu,
                        )
                        y_um, rtf_um = mf2.process(y_u, SAMPLE_RATE)
                        _record(
                            case_name,
                            clip_name,
                            tag.replace("a_", "c_") + "_then_mf2",
                            f"{uvr_dev}+{accel}",
                            rtf_u + rtf_um,
                            clean,
                            degraded,
                            y_um,
                        )
                except Exception as exc:
                    print(f"  [{tag}] ERROR: {exc}", file=sys.stderr)

    # Merge new rows cleanly (replacing prior rows for engines re-run in this invocation)
    merged = [r for r in existing if r.engine not in engines_run] + new_rows
    scorecard_json.write_text(json.dumps([asdict(r) for r in merged], indent=2), encoding="utf-8")
    md = _format_markdown(merged) + "\n" + _evaluate_gates(merged)
    scorecard_md.write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
