#!/usr/bin/env python3
"""Phase 1c: True MONO speech dereverberation shootout on real 48 kHz Sorani Kurdish speech."""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

# Resemble-Enhance imports DeepSpeed inside train.py/utils even for inference-only calls.
for _mod_name in (
    "deepspeed",
    "deepspeed.accelerator",
    "deepspeed.runtime",
    "deepspeed.runtime.engine",
    "deepspeed.runtime.utils",
):
    _m = types.ModuleType(_mod_name)
    _m.__path__ = []
    sys.modules.setdefault(_mod_name, _m)
sys.modules["deepspeed"].DeepSpeedConfig = object  # type: ignore[attr-defined]
sys.modules["deepspeed.accelerator"].get_accelerator = lambda: types.SimpleNamespace(  # type: ignore[attr-defined]
    device_name=lambda: "cpu"
)
sys.modules["deepspeed.runtime.engine"].DeepSpeedEngine = object  # type: ignore[attr-defined]
sys.modules["deepspeed.runtime.utils"].clip_grad_norm_ = lambda *_a, **_k: None  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[2]
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
    run_hawavoclean_venv_core,
)


def wpe_dereverb(
    audio: np.ndarray[Any, np.dtype[np.float32]],
    sr: int,
    taps: int = 16,
    delay: int = 3,
    iterations: int = 3,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    from nara_wpe.wpe import wpe

    n_fft, hop = 1024, 256
    _, _, stft = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, padded=True)
    derev = wpe(stft[:, None, :], taps=taps, delay=delay, iterations=iterations)[:, 0, :]
    _, out = signal.istft(derev, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    return _match_length(np.asarray(out, dtype=np.float32), len(audio))


class ResembleEnhanceRunner:
    """MIT-licensed Resemble-Enhance (44.1 kHz denoiser + conditional flow-matching enhancer)."""

    def __init__(self, device: str) -> None:
        self.device = device
        self._patched = False

    def _ensure_numpy2_compat(self) -> None:
        if self._patched:
            return
        import scipy.optimize
        from resemble_enhance.enhancer.lcfm.cfm import Solver

        def _exp_decay_map(t: Any, n: int = 4) -> Any:
            def h(x: Any, a: Any) -> Any:
                return (a**x - 1) / (a - 1)

            sol = scipy.optimize.fsolve(lambda a: h(1 / n, a) - 0.5, x0=0)
            a = float(np.asarray(sol).reshape(-1)[0])
            return h(t, a=a)

        Solver.exponential_decay_mapping = staticmethod(_exp_decay_map)  # type: ignore[method-assign]
        self._patched = True

    def process(
        self,
        audio: np.ndarray[Any, np.dtype[np.float32]],
        sr: int,
        *,
        mode: str = "enhance",
        nfe: int = 32,
        lambd: float = 0.9,
        tau: float = 0.5,
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float]:
        import torch
        from resemble_enhance.enhancer.inference import denoise, enhance

        self._ensure_numpy2_compat()

        dwav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        t0 = time.perf_counter()
        if mode == "denoise":
            wav_out, out_sr = denoise(dwav, sr, self.device)
        else:
            wav_out, out_sr = enhance(
                dwav, sr, self.device, nfe=nfe, solver="midpoint", lambd=lambd, tau=tau
            )
        dt = time.perf_counter() - t0
        y = np.asarray(wav_out.detach().cpu().float().numpy(), dtype=np.float32)
        y48 = _resample(y, int(out_sr), sr)
        return _match_length(y48, len(audio)), dt / (len(audio) / sr)


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resemble-only",
        action="store_true",
        help="Run only Resemble-Enhance variants and merge into existing mono_dereverb_scorecard.",
    )
    args = parser.parse_args()

    accel = "mps" if torch.backends.mps.is_available() else "cpu"
    kurdish_files = sorted((ROOT / "data/kurdish_speakers").glob("*/*.wav"))
    selected = [kurdish_files[0], kurdish_files[-1]]
    out_dir = ROOT / ".local-ai/bench-results"
    audio_dir = out_dir / "audio_mono_dereverb"
    audio_dir.mkdir(parents=True, exist_ok=True)

    scorer = QualityScorer(device=accel)
    mf2 = MossFormer2Runner("speech_enhancement", "MossFormer2_SE_48K", device=accel)
    lavasr = None if args.resemble_only else LavaSRRunner(device=accel)
    resemble = ResembleEnhanceRunner(device="cpu")
    # Probe resemble on MPS vs CPU once
    try:
        resemble_mps = ResembleEnhanceRunner(device=accel)
        resemble_mps.process(np.zeros(48000, dtype=np.float32), SAMPLE_RATE, mode="denoise")
        resemble = resemble_mps
        resemble_dev = accel
    except Exception as exc:
        print(f"[resemble] MPS fallback to CPU: {exc}", file=sys.stderr)
        resemble_dev = "cpu"

    roformer_mono = (
        None
        if args.resemble_only
        else MLXModelRunner(
            "dereverb_mel_band_roformer_mono_anvuew.ckpt", ("_(noreverb)_", "_(dry)_")
        )
    )
    frcrn = (
        None
        if args.resemble_only
        else MossFormer2Runner("speech_enhancement", "FRCRN_SE_16K", device=accel)
    )

    results: list[ClipMetrics] = []
    scorecard_json = out_dir / "mono_dereverb_scorecard.json"
    if args.resemble_only and scorecard_json.is_file():
        existing_rows = json.loads(scorecard_json.read_text(encoding="utf-8"))
        results = [
            ClipMetrics(**row)
            for row in existing_rows
            if not str(row.get("engine", "")).startswith("4")
        ]

    target_cases = ("clean", "reverb_heavy", "reverb_plus_noise", "noise_street_hum")

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

            if not args.resemble_only:
                results.append(
                    scorer.score(
                        case_name, clip_name, "0_degraded", "none", 0.0, clean, degraded, degraded
                    )
                )
                y_dfn, rtf_dfn = run_hawavoclean_venv_core("dfn3", degraded, SAMPLE_RATE)
                results.append(
                    scorer.score(
                        case_name, clip_name, "1_hawa_dfn3", "cpu", rtf_dfn, clean, degraded, y_dfn
                    )
                )
                results.append(
                    scorer.score(
                        case_name, clip_name, "2_mf2_48k", accel, rtf_mf2, clean, degraded, y_mf2
                    )
                )

                # 3. WPE + MossFormer2_SE_48K (100% MIT + Apache-2.0!)
                t0 = time.perf_counter()
                y_wpe = wpe_dereverb(degraded, SAMPLE_RATE, taps=18, delay=3, iterations=3)
                dt_wpe = time.perf_counter() - t0
                y_wpe_mf2, rtf_wm = mf2.process(y_wpe, SAMPLE_RATE)
                sf.write(
                    audio_dir / f"{clip_name}__{case_name}__3_wpe_then_mf2.wav",
                    _normalize_rms(y_wpe_mf2),
                    SAMPLE_RATE,
                    subtype="PCM_24",
                )
                results.append(
                    scorer.score(
                        case_name,
                        clip_name,
                        "3_wpe_then_mf2",
                        f"cpu+{accel}",
                        dt_wpe / (len(degraded) / SAMPLE_RATE) + rtf_wm,
                        clean,
                        degraded,
                        y_wpe_mf2,
                    )
                )

            # 4. Resemble-Enhance Denoiser only (MIT, discriminative) & Full Enhancer (MIT, CFM dereverb+BWE)
            for label, mode, nfe in (
                ("4a_resemble_denoise", "denoise", 1),
                ("4b_resemble_enhance_nfe32", "enhance", 32),
            ):
                try:
                    y_r, rtf_r = resemble.process(degraded, SAMPLE_RATE, mode=mode, nfe=nfe)
                    sf.write(
                        audio_dir / f"{clip_name}__{case_name}__{label}.wav",
                        _normalize_rms(y_r),
                        SAMPLE_RATE,
                        subtype="PCM_24",
                    )
                    results.append(
                        scorer.score(
                            case_name, clip_name, label, resemble_dev, rtf_r, clean, degraded, y_r
                        )
                    )
                except Exception as exc:
                    print(f"  [{label}] ERROR: {exc}", file=sys.stderr)

            # 4c. MossFormer2_SE_48K -> Resemble-Enhance (MIT+Apache-2.0 hybrid dereverb check)
            if case_name in {"reverb_heavy", "reverb_plus_noise"}:
                try:
                    y_mr, rtf_mr = resemble.process(
                        y_mf2, SAMPLE_RATE, mode="enhance", nfe=32, lambd=0.5, tau=0.5
                    )
                    sf.write(
                        audio_dir / f"{clip_name}__{case_name}__4c_mf2_then_resemble_enhance.wav",
                        _normalize_rms(y_mr),
                        SAMPLE_RATE,
                        subtype="PCM_24",
                    )
                    results.append(
                        scorer.score(
                            case_name,
                            clip_name,
                            "4c_mf2_then_resemble_enhance",
                            f"{accel}+{resemble_dev}",
                            rtf_mf2 + rtf_mr,
                            clean,
                            degraded,
                            y_mr,
                        )
                    )
                except Exception as exc:
                    print(f"  [4c_mf2_then_resemble_enhance] ERROR: {exc}", file=sys.stderr)

            if args.resemble_only:
                continue

            # 5. FRCRN_SE_16K (Apache-2.0) + LavaSR v2 48 kHz BWE (Apache-2.0)
            if frcrn is not None and lavasr is not None:
                try:
                    y_fr, rtf_fr = frcrn.process(degraded, SAMPLE_RATE)
                    y_fr_48, rtf_l = lavasr.process(y_fr, SAMPLE_RATE, denoise=False)
                    sf.write(
                        audio_dir / f"{clip_name}__{case_name}__5_frcrn_plus_lavasr.wav",
                        _normalize_rms(y_fr_48),
                        SAMPLE_RATE,
                        subtype="PCM_24",
                    )
                    results.append(
                        scorer.score(
                            case_name,
                            clip_name,
                            "5_frcrn_plus_lavasr",
                            accel,
                            rtf_fr + rtf_l,
                            clean,
                            degraded,
                            y_fr_48,
                        )
                    )
                except Exception as exc:
                    print(f"  [5_frcrn_plus_lavasr] ERROR: {exc}", file=sys.stderr)

            # 6. Mono Mel-Band RoFormer De-Reverb (anvuew — GPL-3.0 ceiling reference)
            if roformer_mono is not None:
                try:
                    y_rm, rtf_rm = roformer_mono.process(degraded, SAMPLE_RATE)
                    sf.write(
                        audio_dir / f"{clip_name}__{case_name}__6a_roformer_mono_dereverb.wav",
                        _normalize_rms(y_rm),
                        SAMPLE_RATE,
                        subtype="PCM_24",
                    )
                    results.append(
                        scorer.score(
                            case_name,
                            clip_name,
                            "6a_roformer_mono_dereverb",
                            "mlx",
                            rtf_rm,
                            clean,
                            degraded,
                            y_rm,
                        )
                    )
                    if case_name in {"reverb_heavy", "reverb_plus_noise"}:
                        y_rm_mf2, rtf_rm2 = mf2.process(y_rm, SAMPLE_RATE)
                        sf.write(
                            audio_dir / f"{clip_name}__{case_name}__6b_roformer_mono_then_mf2.wav",
                            _normalize_rms(y_rm_mf2),
                            SAMPLE_RATE,
                            subtype="PCM_24",
                        )
                        results.append(
                            scorer.score(
                                case_name,
                                clip_name,
                                "6b_roformer_mono_then_mf2",
                                f"mlx+{accel}",
                                rtf_rm + rtf_rm2,
                                clean,
                                degraded,
                                y_rm_mf2,
                            )
                        )
                except Exception as exc:
                    print(f"  [6_roformer_mono_dereverb] ERROR: {exc}", file=sys.stderr)

    scorecard_json.write_text(json.dumps([asdict(r) for r in results], indent=2))
    md = _format_markdown(results)
    (out_dir / "mono_dereverb_scorecard.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
