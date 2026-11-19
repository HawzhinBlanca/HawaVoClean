#!/usr/bin/env python3
"""Phase 1b: MIT-licensed neural dereverb + denoise shootout on real 48 kHz Sorani Kurdish speech."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/benchmarks"))

from shootout import (  # noqa: E402
    SAMPLE_RATE,
    ClipMetrics,
    MossFormer2Runner,
    QualityScorer,
    _format_markdown,
    _match_length,
    _normalize_rms,
    _resample,
    build_stress_cases,
    run_hawavoclean_venv_core,
)


class MLXModelRunner:
    """Generic runner for any MIT checkpoint supported by mlx_audio_separator."""

    def __init__(self, filename: str, target_keywords: tuple[str, ...]) -> None:
        self.filename = filename
        self.target_keywords = target_keywords
        self._sep: Any = None

    def process(
        self, audio: np.ndarray[Any, np.dtype[np.float32]], sr: int
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float]:
        from mlx_audio_separator.core import Separator

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            inp = td_path / "in.wav"
            sf.write(inp, np.stack([audio, audio], axis=-1), sr, subtype="FLOAT")
            if self._sep is None:
                self._sep = Separator(output_dir=str(td_path), output_format="WAV")
                self._sep.load_model(self.filename)
            else:
                self._sep.output_dir = str(td_path)
            t0 = time.perf_counter()
            files = self._sep.separate(str(inp))
            dt = time.perf_counter() - t0
            chosen: Path | None = None
            for f in files:
                low = Path(f).name.lower()
                if any(kw in low for kw in self.target_keywords):
                    chosen = Path(f)
                    break
            if chosen is None:
                raise RuntimeError(f"Could not match {self.target_keywords} in {files}")
            if not chosen.is_absolute():
                chosen = td_path / chosen
            y, out_sr = sf.read(chosen, dtype="float32")
            if y.ndim == 2:
                y = np.mean(y, axis=1)
            y48 = _resample(np.asarray(y, dtype=np.float32), int(out_sr), sr)
            return _match_length(y48, len(audio)), dt / (len(audio) / sr)


def main() -> None:
    import torch

    accel = "mps" if torch.backends.mps.is_available() else "cpu"
    kurdish_files = sorted((ROOT / "data/kurdish_speakers").glob("*/*.wav"))
    selected = [kurdish_files[0], kurdish_files[-1]]
    out_dir = ROOT / ".local-ai/bench-results"
    audio_dir = out_dir / "audio_dereverb"
    audio_dir.mkdir(parents=True, exist_ok=True)

    scorer = QualityScorer(device=accel)
    mf2 = MossFormer2Runner("speech_enhancement", "MossFormer2_SE_48K", device=accel)

    candidates = [
        (
            "B_mdx23c_dereverb",
            MLXModelRunner("MDX23C-De-Reverb-aufr33-jarredou.ckpt", ("_(dry)_",)),
        ),
        (
            "C_bs_roformer_dereverb",
            MLXModelRunner("deverb_bs_roformer_8_384dim_10depth.ckpt", ("_(noreverb)_",)),
        ),
        (
            "D_mel_roformer_denoise",
            MLXModelRunner("denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt", ("_(dry)_",)),
        ),
    ]

    results: list[ClipMetrics] = []
    target_cases = ("clean", "reverb_heavy", "noise_street_hum", "reverb_plus_noise")

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
            results.append(
                scorer.score(
                    case_name, clip_name, "0_degraded", "none", 0.0, clean, degraded, degraded
                )
            )
            y_dfn, rtf_dfn = run_hawavoclean_venv_core("dfn3", degraded, SAMPLE_RATE)
            results.append(
                scorer.score(
                    case_name, clip_name, "0b_hawa_dfn3_cpu", "cpu", rtf_dfn, clean, degraded, y_dfn
                )
            )

            y_mf2, rtf_mf2 = mf2.process(degraded, SAMPLE_RATE)
            results.append(
                scorer.score(
                    case_name, clip_name, "1_mf2_48k_only", accel, rtf_mf2, clean, degraded, y_mf2
                )
            )

            for label, runner in candidates:
                try:
                    y, rtf = runner.process(degraded, SAMPLE_RATE)
                    sf.write(
                        audio_dir / f"{clip_name}__{case_name}__{label}.wav",
                        _normalize_rms(y),
                        SAMPLE_RATE,
                        subtype="PCM_24",
                    )
                    results.append(
                        scorer.score(case_name, clip_name, label, "mlx", rtf, clean, degraded, y)
                    )

                    # Test chained dereverb -> MossFormer2_SE_48K on reverb+noise and heavy reverb
                    if label in {"B_mdx23c_dereverb", "C_bs_roformer_dereverb"} and case_name in {
                        "reverb_heavy",
                        "reverb_plus_noise",
                    }:
                        y_chain, rtf_c = mf2.process(y, SAMPLE_RATE)
                        sf.write(
                            audio_dir / f"{clip_name}__{case_name}__{label}_then_mf2.wav",
                            _normalize_rms(y_chain),
                            SAMPLE_RATE,
                            subtype="PCM_24",
                        )
                        results.append(
                            scorer.score(
                                case_name,
                                clip_name,
                                f"{label}_then_mf2",
                                "mlx+mps",
                                rtf + rtf_c,
                                clean,
                                degraded,
                                y_chain,
                            )
                        )
                        # Also test opposite order: mf2 -> dereverb
                        y_rev, rtf_r = runner.process(y_mf2, SAMPLE_RATE)
                        sf.write(
                            audio_dir / f"{clip_name}__{case_name}__mf2_then_{label}.wav",
                            _normalize_rms(y_rev),
                            SAMPLE_RATE,
                            subtype="PCM_24",
                        )
                        results.append(
                            scorer.score(
                                case_name,
                                clip_name,
                                f"mf2_then_{label}",
                                "mps+mlx",
                                rtf_mf2 + rtf_r,
                                clean,
                                degraded,
                                y_rev,
                            )
                        )
                except Exception as exc:
                    print(f"  [{label}] ERROR: {exc}", file=sys.stderr)

    (out_dir / "dereverb_scorecard.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2)
    )
    md = _format_markdown(results)
    (out_dir / "dereverb_scorecard.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
