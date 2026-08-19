"""Guard calibration: MEASURED accept/revert rates over a labelled corpus.

For every corpus item, the guard is evaluated against (a) corrupted
renderings that must be rejected and (b) benign renderings that should be
accepted. The reported rates are counted from those evaluations — nothing
in the output artifact is hand-written.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from voiceclean.config import GuardConfig
from voiceclean.eval.corpus import load_corpus_manifest
from voiceclean.eval.corruption import (
    corrupt_dropout,
    corrupt_hf_consonant_removal,
    corrupt_repeated_span,
    corrupt_spectral_holes,
    corrupt_syllable_deletion,
)
from voiceclean.guard.spectral_probe import FixedProbe, SpectralSignatureProbe
from voiceclean.guard.verdict import GuardVerdict, evaluate_guard_pass
from voiceclean.hashing import hash_json_canonical

# Corruption operator suites by severity profile. "severe" renderings destroy
# obvious spectral structure; "mild" ones perturb it slightly — a guard tuned
# to reject the former will accept some of the latter, and the measured gap
# between the two rates is evidence the measurement is real.
CORRUPTION_PROFILES: dict[str, list[Any]] = {
    "severe": [
        lambda w, sr: corrupt_syllable_deletion(w, sr, start_time_s=0.5, deletion_ms=400.0),
        lambda w, sr: corrupt_hf_consonant_removal(w, sr, cutoff_hz=800.0),
        lambda w, sr: corrupt_spectral_holes(w, sr, band_low_hz=500.0, band_high_hz=4000.0),
        lambda w, sr: corrupt_dropout(w, sr, start_time_s=0.5, duration_ms=500.0),
        lambda w, sr: corrupt_repeated_span(w, sr, start_time_s=0.5, span_ms=600.0),
    ],
    "standard": [
        lambda w, sr: corrupt_syllable_deletion(w, sr, start_time_s=1.0, deletion_ms=200.0),
        lambda w, sr: corrupt_hf_consonant_removal(w, sr, cutoff_hz=1500.0),
        lambda w, sr: corrupt_spectral_holes(w, sr, band_low_hz=2000.0, band_high_hz=4000.0),
        lambda w, sr: corrupt_dropout(w, sr, start_time_s=1.0, duration_ms=150.0),
    ],
    "mild": [
        lambda w, sr: (w * 0.97).astype(np.float32),  # -0.26 dB gain
        lambda w, sr: corrupt_dropout(w, sr, start_time_s=1.0, duration_ms=8.0),
        lambda w, sr: (w + 0.0005 * np.random.default_rng(0).standard_normal(len(w))).astype(
            np.float32
        ),
    ],
}

# Benign renderings the guard should accept (false reverts counted here).
BENIGN_RENDERINGS: list[Any] = [
    lambda w, sr: w.copy(),  # identity
    lambda w, sr: (w * 0.995).astype(np.float32),  # negligible gain change
]


def _same_length(orig: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Match candidate length to the original (guard compares same spans)."""
    if len(cand) >= len(orig):
        return np.ascontiguousarray(cand[: len(orig)], dtype=np.float32)
    return np.ascontiguousarray(
        np.pad(cand, (0, len(orig) - len(cand))), dtype=np.float32
    )


def run_calibration(
    manifest_path: Path | str,
    output_calibration_path: Path | str,
    corruption_profile: str = "standard",
    use_fixed_probe: bool = False,
) -> dict[str, Any]:
    """Measure guard accept/revert rates over the corpus and write the artifact."""
    if corruption_profile not in CORRUPTION_PROFILES:
        raise ValueError(
            f"Unknown corruption profile {corruption_profile!r}; "
            f"choose from {sorted(CORRUPTION_PROFILES)}"
        )
    manifest = load_corpus_manifest(manifest_path)
    probe = FixedProbe() if use_fixed_probe else SpectralSignatureProbe()

    guard_cfg = GuardConfig()
    thresholds = {
        "min_anchor_confidence": guard_cfg.min_anchor_confidence,
        "max_posterior_js_div": guard_cfg.max_posterior_js_div,
        "max_timing_drift_ms": guard_cfg.max_timing_drift_ms,
        "spectral_hole_thresh": guard_cfg.spectral_hole_thresh,
        "musical_noise_thresh": guard_cfg.musical_noise_thresh,
        "min_hf_preservation_ratio": guard_cfg.min_hf_preservation_ratio,
    }

    corrupted_total = 0
    corrupted_accepted = 0
    benign_total = 0
    benign_rejected = 0

    for item in manifest.items:
        waveform, sr = sf.read(str(item.audio_path), dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform[:, 0]

        for corrupt in CORRUPTION_PROFILES[corruption_profile]:
            cand = _same_length(waveform, corrupt(waveform, sr))
            res, _ = evaluate_guard_pass(
                orig_waveform=waveform,
                cand_waveform=cand,
                sample_rate=sr,
                is_speech=True,
                probe=probe,
                config=guard_cfg,
            )
            corrupted_total += 1
            if res.verdict == GuardVerdict.PASS:
                corrupted_accepted += 1

        for render in BENIGN_RENDERINGS:
            cand = _same_length(waveform, render(waveform, sr))
            res, _ = evaluate_guard_pass(
                orig_waveform=waveform,
                cand_waveform=cand,
                sample_rate=sr,
                is_speech=True,
                probe=probe,
                config=guard_cfg,
            )
            benign_total += 1
            if res.verdict in (GuardVerdict.REVERT, GuardVerdict.UNVERIFIED):
                benign_rejected += 1

    false_accept_rate = corrupted_accepted / corrupted_total if corrupted_total else 0.0
    false_revert_rate = benign_rejected / benign_total if benign_total else 0.0

    artifact = {
        "schema_version": 2,
        "calibration_id": hash_json_canonical(thresholds),
        "guard_id": guard_cfg.guard_id,
        "probe_id": probe.probe_id,
        "provenance": (
            "Rates below were measured by evaluate_guard_pass over the corpus "
            "and corruption suite identified in the 'measured' block."
        ),
        "thresholds": thresholds,
        "metrics": {
            "calibration_false_accept_rate": false_accept_rate,
            "calibration_false_revert_rate": false_revert_rate,
        },
        "measured": {
            "corpus_manifest_sha256": manifest.manifest_sha256,
            "corruption_profile": corruption_profile,
            "item_count": len(manifest.items),
            "corrupted_evaluations": corrupted_total,
            "corrupted_accepted": corrupted_accepted,
            "benign_evaluations": benign_total,
            "benign_rejected": benign_rejected,
        },
    }

    out_file = Path(output_calibration_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    return artifact
