"""Standard speech enhancement quality metrics for research-grade evaluation.

Computes PESQ, STOI, SI-SNR, LSD, and speech/floor separation given a
reference and candidate WAV pair.  All metrics are computed at the reference
sample rate with automatic resampling where needed.

Dependencies live in the ``metrics`` optional extra (pesq, pystoi) and are
imported lazily so the core library never pays for them.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from hawavoclean.audio.resample import resample_audio
from hawavoclean.hashing import hash_file
from hawavoclean.logging import get_logger
from hawavoclean.multipass import speech_floor_separation_db

logger = get_logger("metrics")

# PESQ requires 8 kHz or 16 kHz; we use wideband (16 kHz) by default.
_PESQ_RATE = 16000

# STOI operates at any rate but is defined for 10 kHz bandwidth (≥16 kHz).
_STOI_RATE = 16000


@dataclass(frozen=True)
class MetricsResult:
    """Standard speech enhancement quality metrics for one reference/candidate pair."""

    reference_path: str
    candidate_path: str
    reference_sha256: str
    candidate_sha256: str
    sample_rate: int
    duration_s: float

    #: ITU-T P.862.2 wideband PESQ score (−0.5 to 4.5).
    pesq_wb: float | None

    #: Extended Short-Time Objective Intelligibility (0 to 1).
    estoi: float | None

    #: Scale-Invariant Signal-to-Noise Ratio (dB, higher is better).
    si_snr_db: float

    #: Log-Spectral Distance (dB, lower is better).
    lsd_db: float

    #: Speech/floor separation (dB, existing HawaVoClean metric).
    separation_db: float

    #: Wall-clock computation time for all metrics (seconds).
    compute_time_s: float

    #: Any warnings encountered during computation.
    warnings: list[str]


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load an audio or video file and return mono float32 samples + sample rate."""
    try:
        from hawavoclean.audio.decode import decode_audio
        from hawavoclean.audio.probe import probe_audio

        probe = probe_audio(path)
        buf = decode_audio(probe)
        mono = buf.data.mean(axis=0)
        return mono.astype(np.float32), int(buf.sample_rate)
    except Exception:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        return mono.astype(np.float32), int(sr)


def _match_lengths(
    ref: np.ndarray, cand: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Trim or pad to match lengths (required by all metrics)."""
    min_len = min(len(ref), len(cand))
    return ref[:min_len], cand[:min_len]


def _compute_si_snr(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Noise Ratio in dB.

    SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    where s_target = (<s, s_hat> / ||s||^2) * s
    and   e_noise  = s_hat - s_target
    """
    ref = reference - np.mean(reference)
    cand = candidate - np.mean(candidate)

    dot = float(np.dot(ref, cand))
    s_ref_energy = float(np.dot(ref, ref))

    if s_ref_energy < 1e-10:
        return 0.0

    s_target = (dot / s_ref_energy) * ref
    e_noise = cand - s_target

    si_snr = 10.0 * np.log10(
        max(float(np.dot(s_target, s_target)), 1e-10)
        / max(float(np.dot(e_noise, e_noise)), 1e-10)
    )
    return float(si_snr)


def _compute_lsd(
    reference: np.ndarray,
    candidate: np.ndarray,
    n_fft: int = 2048,
    hop: int = 512,
) -> float:
    """Log-Spectral Distance in dB (lower is better).

    LSD = mean over frames of sqrt(mean over bins of (log S_ref - log S_cand)^2)
    """
    win = np.hanning(n_fft)

    def _stft_power(x: np.ndarray) -> np.ndarray:
        num_frames = max(1, (len(x) - n_fft) // hop + 1)
        power = np.zeros((num_frames, n_fft // 2 + 1))
        for i in range(num_frames):
            chunk = x[i * hop : i * hop + n_fft] * win
            spec = np.fft.rfft(chunk, n=n_fft)
            power[i] = np.abs(spec) ** 2
        return power

    ref_power = _stft_power(reference)
    cand_power = _stft_power(candidate)

    # Floor to avoid log(0)
    ref_log = np.log10(np.maximum(ref_power, 1e-10))
    cand_log = np.log10(np.maximum(cand_power, 1e-10))

    # Per-frame LSD
    frame_lsd = np.sqrt(np.mean((ref_log - cand_log) ** 2, axis=1))
    return float(np.mean(frame_lsd))


def compute_metrics(
    reference_path: Path | str,
    candidate_path: Path | str,
) -> MetricsResult:
    """Compute all standard speech enhancement metrics for a pair of WAV files.

    Lazily imports ``pesq`` and ``pystoi`` so the core library is not burdened.
    If either is unavailable, the corresponding metric is ``None`` with a warning.
    """
    t_start = time.perf_counter()
    warnings: list[str] = []

    ref_path = Path(reference_path).resolve()
    cand_path = Path(candidate_path).resolve()

    ref_hash = hash_file(ref_path)
    cand_hash = hash_file(cand_path)

    ref_mono, ref_sr = _load_mono(ref_path)
    cand_mono, cand_sr = _load_mono(cand_path)

    # Resample candidate to reference rate if different
    if cand_sr != ref_sr:
        cand_mono = resample_audio(
            cand_mono, cand_sr, ref_sr, target_samples=len(ref_mono)
        )
        warnings.append(f"Resampled candidate from {cand_sr} Hz to {ref_sr} Hz")

    ref_mono, cand_mono = _match_lengths(ref_mono, cand_mono)
    duration_s = len(ref_mono) / ref_sr

    # --- SI-SNR (always available, pure numpy) ---
    si_snr = _compute_si_snr(ref_mono, cand_mono)

    # --- LSD (always available, pure numpy) ---
    lsd = _compute_lsd(ref_mono, cand_mono)

    # --- Separation (existing metric) ---
    sep = speech_floor_separation_db(cand_mono)

    # --- PESQ (optional, needs `pesq` package) ---
    pesq_score: float | None = None
    try:
        from pesq import pesq as pesq_fn

        # PESQ needs 8k or 16k
        if ref_sr == _PESQ_RATE:
            pesq_ref, pesq_cand = ref_mono, cand_mono
        else:
            pesq_ref = resample_audio(ref_mono, ref_sr, _PESQ_RATE)
            pesq_cand = resample_audio(cand_mono, ref_sr, _PESQ_RATE)
        pesq_ref, pesq_cand = _match_lengths(pesq_ref, pesq_cand)
        pesq_score = float(pesq_fn(_PESQ_RATE, pesq_ref, pesq_cand, "wb"))
    except ImportError:
        warnings.append("pesq not installed; skipping PESQ (pip install pesq)")
    except Exception as e:
        warnings.append(f"PESQ computation failed: {e}")

    # --- ESTOI (optional, needs `pystoi` package) ---
    estoi_score: float | None = None
    try:
        from pystoi import stoi as stoi_fn

        if ref_sr == _STOI_RATE:
            stoi_ref, stoi_cand = ref_mono, cand_mono
        else:
            stoi_ref = resample_audio(ref_mono, ref_sr, _STOI_RATE)
            stoi_cand = resample_audio(cand_mono, ref_sr, _STOI_RATE)
        stoi_ref, stoi_cand = _match_lengths(stoi_ref, stoi_cand)
        estoi_score = float(stoi_fn(stoi_ref, stoi_cand, _STOI_RATE, extended=True))
    except ImportError:
        warnings.append("pystoi not installed; skipping ESTOI (pip install pystoi)")
    except Exception as e:
        warnings.append(f"ESTOI computation failed: {e}")

    compute_time = time.perf_counter() - t_start

    return MetricsResult(
        reference_path=str(ref_path),
        candidate_path=str(cand_path),
        reference_sha256=ref_hash,
        candidate_sha256=cand_hash,
        sample_rate=ref_sr,
        duration_s=duration_s,
        pesq_wb=pesq_score,
        estoi=estoi_score,
        si_snr_db=si_snr,
        lsd_db=lsd,
        separation_db=sep,
        compute_time_s=compute_time,
        warnings=warnings,
    )


def compute_corpus_metrics(
    pairs: Sequence[tuple[Path | str, Path | str]],
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compute metrics for a list of (reference, candidate) pairs.

    Returns aggregate statistics and per-pair results.  Optionally writes
    the full result to a JSON file.
    """
    results: list[dict[str, Any]] = []
    pesq_vals: list[float] = []
    estoi_vals: list[float] = []
    si_snr_vals: list[float] = []
    lsd_vals: list[float] = []
    sep_vals: list[float] = []

    for ref_path, cand_path in pairs:
        m = compute_metrics(ref_path, cand_path)
        results.append(asdict(m))
        if m.pesq_wb is not None:
            pesq_vals.append(m.pesq_wb)
        if m.estoi is not None:
            estoi_vals.append(m.estoi)
        si_snr_vals.append(m.si_snr_db)
        lsd_vals.append(m.lsd_db)
        sep_vals.append(m.separation_db)

    def _stats(vals: list[float]) -> dict[str, float] | None:
        if not vals:
            return None
        arr = np.array(vals)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "n": len(vals),
        }

    report: dict[str, Any] = {
        "total_pairs": len(pairs),
        "aggregate": {
            "pesq_wb": _stats(pesq_vals),
            "estoi": _stats(estoi_vals),
            "si_snr_db": _stats(si_snr_vals),
            "lsd_db": _stats(lsd_vals),
            "separation_db": _stats(sep_vals),
        },
        "per_pair": results,
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        logger.info("Corpus metrics written to %s", out)

    return report
