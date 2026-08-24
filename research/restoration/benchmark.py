"""Restoration benchmark harness evaluating baseline and personalized models across cutoffs.

Computes log-spectral distance (LSD, dB, lower is better), protected-band RMS invariance
error, and cosine similarity from the in-repo deterministic mel-DSP
SpeakerEmbeddingExtractor. Test signals are the synthetic canonical profile fixture WAVs
under profiles/ — the same recordings the personalized model conditions on (non-holdout),
so this harness measures pipeline integrity, not product quality on real held-out speech.
The output JSON records the UniverSR execution mode (official neural weights vs. silent
DSP fallback), the degradation seed, and the cutoffs actually run.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from hawavoclean.restoration.hawarestore_kd import HawaRestoreKD
from hawavoclean.restoration.profiles import load_speaker_profile
from hawavoclean.restoration.speaker_embed import SpeakerEmbeddingExtractor
from hawavoclean.restoration.universr_upstream import UniverSRBaseline
from research.restoration.evaluation.metrics import evaluate_restoration
from research.restoration.simulation.degradation import DegradationSimulator


@dataclass(frozen=True)
class BenchmarkSampleResult:
    """Benchmark outcome for a single test utterance under a specific cutoff."""

    speaker_id: str
    cutoff_hz: float
    method: str
    fullband_lsd_db: float
    highband_lsd_db: float
    protected_band_rms_err: float
    highband_energy_error_db: float
    speaker_cosine_sim: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_restoration_benchmark(
    manifest_path: str | Path | None = None,
    output_json_path: str | Path | None = None,
    profiles_root: str | Path = "profiles",
    seed: int = 42,
) -> dict[str, Any]:
    """Execute complete restoration benchmark across cutoffs and baseline comparisons."""
    cutoffs = [4000.0, 7500.0, 12000.0, 16000.0]
    speakers = [f"character_{i:02d}" for i in range(1, 11)]

    if manifest_path is not None and Path(manifest_path).exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest_cfg = json.load(f)
            if "cutoffs_hz" in manifest_cfg:
                cutoffs = [float(c) for c in manifest_cfg["cutoffs_hz"]]
            if "speakers" in manifest_cfg:
                speakers = [str(s) for s in manifest_cfg["speakers"]]
            if "seed" in manifest_cfg:
                seed = int(manifest_cfg["seed"])

    deg_sim = DegradationSimulator(sample_rate=48000)
    universr = UniverSRBaseline(sample_rate=48000)
    hawarestore = HawaRestoreKD(sample_rate=48000)
    spk_extractor = SpeakerEmbeddingExtractor(sample_rate=48000)

    # Record whether the UniverSR baseline actually ran official neural weights
    # or silently fell back to deterministic DSP spectral extrapolation. The two
    # produce very different numbers, so evidence without this field is ambiguous.
    universr_mode = "neural" if universr._neural_model is not None else "dsp_fallback"

    results: list[dict[str, Any]] = []

    for spk_idx, spk_id in enumerate(speakers, 1):
        print(f"[{spk_idx}/{len(speakers)}] Benchmarking speaker {spk_id}...")
        profile = None
        canonical_audio_path = Path(profiles_root) / spk_id / "canonical" / f"{spk_id}_ref.wav"

        if canonical_audio_path.exists():
            clean, sr = sf.read(canonical_audio_path, dtype="float32")
            if clean.ndim == 2:
                clean = np.mean(clean, axis=1)
        else:
            try:
                profile = load_speaker_profile(spk_id, profiles_root=profiles_root)
                f0 = profile.f0_statistics.median_hz
            except Exception:
                f0 = 150.0
            sr = 48000
            t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False, dtype=np.float32)
            clean = 0.4 * np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
            clean = (clean / (np.max(np.abs(clean)) + 1e-6) * 0.7).astype(np.float32)

        try:
            profile = load_speaker_profile(spk_id, profiles_root=profiles_root)
        except Exception:
            profile = None

        clean_emb = spk_extractor.extract(clean)
        clean_emb_norm = float(np.linalg.norm(clean_emb))

        for cutoff in cutoffs:
            degraded, _ = deg_sim.degrade(clean, cutoff_hz=cutoff, seed=seed)
            deg_emb = spk_extractor.extract(degraded)
            deg_emb_norm = float(np.linalg.norm(deg_emb))
            deg_sim_score = (
                float(np.dot(clean_emb, deg_emb) / (clean_emb_norm * deg_emb_norm))
                if clean_emb_norm > 1e-6 and deg_emb_norm > 1e-6
                else 0.0
            )

            # 1. Degraded Input
            m_deg = evaluate_restoration(clean, degraded, cutoff_hz=cutoff)
            results.append(
                BenchmarkSampleResult(
                    speaker_id=spk_id,
                    cutoff_hz=cutoff,
                    method="degraded_input",
                    fullband_lsd_db=m_deg.fullband_lsd_db,
                    highband_lsd_db=m_deg.highband_lsd_db,
                    protected_band_rms_err=m_deg.protected_band_rms_err,
                    highband_energy_error_db=m_deg.highband_energy_error_db,
                    speaker_cosine_sim=deg_sim_score,
                ).to_dict()
            )

            # 2. UniverSR Baseline
            uni_cands = universr.restore(
                degraded, sample_rate=sr, effective_cutoff_hz=cutoff, strengths=[1.0]
            )
            uni_audio = uni_cands[0].audio
            m_uni = evaluate_restoration(clean, uni_audio, cutoff_hz=cutoff)
            uni_emb = spk_extractor.extract(uni_audio)
            uni_emb_norm = float(np.linalg.norm(uni_emb))
            uni_sim = (
                float(np.dot(clean_emb, uni_emb) / (clean_emb_norm * uni_emb_norm))
                if clean_emb_norm > 1e-6 and uni_emb_norm > 1e-6
                else 0.0
            )
            results.append(
                BenchmarkSampleResult(
                    speaker_id=spk_id,
                    cutoff_hz=cutoff,
                    method="universr_baseline",
                    fullband_lsd_db=m_uni.fullband_lsd_db,
                    highband_lsd_db=m_uni.highband_lsd_db,
                    protected_band_rms_err=m_uni.protected_band_rms_err,
                    highband_energy_error_db=m_uni.highband_energy_error_db,
                    speaker_cosine_sim=uni_sim,
                ).to_dict()
            )

            # 3. Generic Kurdish HawaRestore-KD
            gen_cands = hawarestore.restore(
                degraded, sample_rate=sr, effective_cutoff_hz=cutoff, strengths=[1.0]
            )
            gen_audio = gen_cands[0].audio
            m_gen = evaluate_restoration(clean, gen_audio, cutoff_hz=cutoff)
            gen_emb = spk_extractor.extract(gen_audio)
            gen_emb_norm = float(np.linalg.norm(gen_emb))
            gen_sim = (
                float(np.dot(clean_emb, gen_emb) / (clean_emb_norm * gen_emb_norm))
                if clean_emb_norm > 1e-6 and gen_emb_norm > 1e-6
                else 0.0
            )
            results.append(
                BenchmarkSampleResult(
                    speaker_id=spk_id,
                    cutoff_hz=cutoff,
                    method="hawarestore_generic_kurdish",
                    fullband_lsd_db=m_gen.fullband_lsd_db,
                    highband_lsd_db=m_gen.highband_lsd_db,
                    protected_band_rms_err=m_gen.protected_band_rms_err,
                    highband_energy_error_db=m_gen.highband_energy_error_db,
                    speaker_cosine_sim=gen_sim,
                ).to_dict()
            )

            # 4. Personalized HawaRestore-KD
            spk_emb = profile.embedding_vector if profile else None
            pers_cands = hawarestore.restore(
                degraded,
                sample_rate=sr,
                effective_cutoff_hz=cutoff,
                speaker_id=spk_id,
                speaker_embedding=spk_emb,
                strengths=[1.0],
            )
            pers_audio = pers_cands[0].audio
            m_pers = evaluate_restoration(clean, pers_audio, cutoff_hz=cutoff)
            pers_emb = spk_extractor.extract(pers_audio)
            pers_emb_norm = float(np.linalg.norm(pers_emb))
            pers_sim = (
                float(np.dot(clean_emb, pers_emb) / (clean_emb_norm * pers_emb_norm))
                if clean_emb_norm > 1e-6 and pers_emb_norm > 1e-6
                else 0.0
            )
            results.append(
                BenchmarkSampleResult(
                    speaker_id=spk_id,
                    cutoff_hz=cutoff,
                    method="hawarestore_personalized_kd",
                    fullband_lsd_db=m_pers.fullband_lsd_db,
                    highband_lsd_db=m_pers.highband_lsd_db,
                    protected_band_rms_err=m_pers.protected_band_rms_err,
                    highband_energy_error_db=m_pers.highband_energy_error_db,
                    speaker_cosine_sim=pers_sim,
                ).to_dict()
            )

    # Compute aggregate summary
    methods = [
        "degraded_input",
        "universr_baseline",
        "hawarestore_generic_kurdish",
        "hawarestore_personalized_kd",
    ]
    summary: dict[str, Any] = {}
    for m in methods:
        m_entries = [r for r in results if r["method"] == m]
        if m_entries:
            summary[m] = {
                "mean_fullband_lsd_db": float(np.mean([e["fullband_lsd_db"] for e in m_entries])),
                "mean_highband_lsd_db": float(np.mean([e["highband_lsd_db"] for e in m_entries])),
                "mean_protected_band_rms": float(
                    np.mean([e["protected_band_rms_err"] for e in m_entries])
                ),
                "mean_speaker_similarity": float(
                    np.mean([e["speaker_cosine_sim"] for e in m_entries])
                ),
            }

    report = {
        "benchmark": "HawaVoClean Kurdish Bandwidth Restoration v2.1",
        "num_speakers": len(speakers),
        "cutoffs_hz": cutoffs,
        "seed": seed,
        "universr_mode": universr_mode,
        "test_set_provenance": "synthetic_profile_fixtures, non-holdout",
        "schema_note": (
            "LSD (log-spectral distance) metrics are in dB and lower is better; "
            "speaker_cosine_sim is higher is better. speaker_cosine_sim is computed by the "
            "in-repo deterministic mel-DSP SpeakerEmbeddingExtractor, not a neural speaker "
            "verification model. Test signals are the 10 synthetic canonical profile WAVs "
            "that the personalized model also conditions on (non-holdout)."
        ),
        "summary": summary,
        "results": results,
    }

    if output_json_path is not None:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    report = run_restoration_benchmark(
        manifest_path="research/restoration/configs/benchmark_manifest.json",
        output_json_path="proof/restoration-v2.1/07-benchmarks/benchmark_results.json",
    )
    print("Benchmark complete.")
    print(f"universr_mode: {report['universr_mode']}")
    print(f"seed: {report['seed']}")
    print(f"cutoffs_hz: {report['cutoffs_hz']}")
    print(f"test_set_provenance: {report['test_set_provenance']}")
    print(json.dumps(report["summary"], indent=2))
