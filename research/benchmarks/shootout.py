#!/usr/bin/env python3
"""Phase 1 falsifiable SOTA model shootout on real 48 kHz Sorani Kurdish speech.

Runs inside the isolated ``.local-ai/bench-env`` environment so production
dependencies remain untouched until a candidate wins every objective gate.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hawavoclean.restoration.protected_band import (  # noqa: E402
    compute_transition_mask,
    merge_protected_spectrum,
    verify_protected_band_invariance,
)

SAMPLE_RATE = 48_000


@dataclass
class ClipMetrics:
    case: str
    clip: str
    engine: str
    device: str
    rtf: float
    si_sdr_db: float | None
    pesq_wb: float | None
    estoi: float | None
    lsd_full_db: float | None
    lsd_high_db: float | None
    protected_band_err_db: float | None
    nisqa_mos: float | None
    nisqa_noi: float | None
    nisqa_dis: float | None
    nisqa_col: float | None
    nisqa_loud: float | None
    dnsmos_ovrl: float | None
    dnsmos_sig: float | None
    dnsmos_bak: float | None
    wavlm_phonetic_cos: float | None
    error: str | None = None


def _resample(
    x: np.ndarray[Any, np.dtype[np.float32]], src_sr: int, dst_sr: int
) -> np.ndarray[Any, np.dtype[np.float32]]:
    if src_sr == dst_sr:
        return np.asarray(x, dtype=np.float32)
    g = math.gcd(src_sr, dst_sr)
    y = signal.resample_poly(x, dst_sr // g, src_sr // g)
    return np.asarray(y, dtype=np.float32)


def _match_length(
    x: np.ndarray[Any, np.dtype[np.float32]], target_len: int
) -> np.ndarray[Any, np.dtype[np.float32]]:
    if len(x) >= target_len:
        return np.ascontiguousarray(x[:target_len], dtype=np.float32)
    return np.pad(x, (0, target_len - len(x))).astype(np.float32)


def _normalize_rms(
    x: np.ndarray[Any, np.dtype[np.float32]], target_rms: float = 0.08
) -> np.ndarray[Any, np.dtype[np.float32]]:
    rms = float(np.sqrt(np.mean(x**2) + 1e-12))
    if rms < 1e-6:
        return x.copy()
    y = x * (target_rms / rms)
    peak = float(np.max(np.abs(y)))
    if peak > 0.95:
        y = y * (0.95 / peak)
    return np.asarray(y, dtype=np.float32)


def _make_rir(
    sr: int, rt60: float, seed: int, drr_db: float = 1.0
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Physically calibrated room impulse response with explicit DRR (dB) and RT60 (s).

    Real untreated rooms have DRR ~ +3 to +6 dB (RT60 0.5-0.7s); harsh tiled halls
    have DRR ~ 0 to +2 dB (RT60 0.9-1.2s). Normalizing tail by total L2 energy
    (not per-sample RMS!) guarantees exact physical direct-to-reverberant ratio.
    """
    rng = np.random.default_rng(seed)
    n = int(sr * min(1.8, rt60 * 1.4))
    t = np.arange(n, dtype=np.float64) / sr
    decay = np.exp(-6.9078 * t / rt60)
    tail = rng.standard_normal(n) * decay
    # Natural wall high-frequency absorption
    sos = signal.butter(2, 5500.0, btype="lowpass", fs=sr, output="sos")
    tail = signal.sosfilt(sos, tail)
    # Remove direct-window overlap from diffuse tail (< 10 ms)
    tail[: int(sr * 0.010)] = 0.0

    # Discrete early reflections (12-60 ms)
    early = np.zeros(n, dtype=np.float64)
    for delay_ms, amp in ((14, 0.35), (23, -0.28), (34, 0.22), (47, -0.17), (59, 0.12)):
        idx = int(sr * delay_ms / 1000.0)
        if idx < n:
            early[idx] += amp

    # Combine early + diffuse tail and scale total reverberant energy to exact DRR
    reverb = early + tail / (float(np.sqrt(np.sum(tail**2))) + 1e-12) * 0.85
    reverb_energy = float(np.sum(reverb**2)) + 1e-12
    target_reverb_energy = 10.0 ** (-drr_db / 10.0)  # direct energy = 1.0
    reverb *= math.sqrt(target_reverb_energy / reverb_energy)

    rir = reverb.copy()
    rir[0] = 1.0
    rir /= float(np.sqrt(np.sum(rir**2) + 1e-12))
    return rir.astype(np.float32)


def _make_street_hum_noise(n: int, sr: int, seed: int) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Deterministic traffic pink/brown rumble + modulated bursts + 50/100/150 Hz hum."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    # Pink-ish traffic bed via 2nd-order lowpass + shelf
    sos_low = signal.butter(2, 700.0, btype="lowpass", fs=sr, output="sos")
    traffic = signal.sosfilt(sos_low, white) * 1.6 + white * 0.22
    # Non-stationary bursts (passing car / café clatter)
    t = np.arange(n, dtype=np.float64) / sr
    env = 0.55 + 0.45 * np.sin(2.0 * np.pi * 0.37 * t + 0.4) ** 2
    burst_sos = signal.butter(2, [400.0, 3200.0], btype="bandpass", fs=sr, output="sos")
    bursts = signal.sosfilt(burst_sos, rng.standard_normal(n)) * env * 0.45
    hum = (
        0.18 * np.sin(2.0 * np.pi * 50.0 * t)
        + 0.11 * np.sin(2.0 * np.pi * 100.0 * t + 0.3)
        + 0.06 * np.sin(2.0 * np.pi * 150.0 * t + 0.7)
    )
    noise = traffic + bursts + hum
    return np.asarray(noise / (np.sqrt(np.mean(noise**2)) + 1e-12), dtype=np.float32)


def _make_music_bed(n: int, sr: int, seed: int) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Polyphonic synthetic music bed (bass + warm chords + rhythmic pulse)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / sr
    beat_hz = 2.0  # 120 BPM
    chord_freqs = [
        (130.81, 196.0, 261.63, 329.63),
        (146.83, 220.0, 293.66, 349.23),
        (110.0, 164.81, 220.0, 261.63),
        (98.0, 146.83, 196.0, 246.94),
    ]
    music = np.zeros(n, dtype=np.float64)
    bar_samples = int(sr * 2.0)
    for idx, start in enumerate(range(0, n, bar_samples)):
        end = min(n, start + bar_samples)
        tt = t[start:end]
        freqs = chord_freqs[idx % len(chord_freqs)]
        bar = np.zeros(end - start, dtype=np.float64)
        for k, f0 in enumerate(freqs):
            bar += (0.35 / (k + 1)) * np.sin(2.0 * np.pi * f0 * tt)
            bar += (0.14 / (k + 1)) * np.sin(2.0 * np.pi * (2.0 * f0) * tt + 0.2)
            bar += (0.07 / (k + 1)) * np.sin(2.0 * np.pi * (3.0 * f0) * tt + 0.5)
        # Rhythmic kick + hi-hat pulse
        pulse = np.exp(-12.0 * ((tt * beat_hz) % 1.0))
        kick = 0.5 * pulse * np.sin(2.0 * np.pi * (55.0 + 45.0 * pulse) * tt)
        hat = (
            0.12
            * np.exp(-35.0 * (((tt * beat_hz * 2.0) + 0.5) % 1.0))
            * rng.standard_normal(end - start)
        )
        music[start:end] = bar + kick + hat
    return np.asarray(music / (np.sqrt(np.mean(music**2)) + 1e-12), dtype=np.float32)


def build_stress_cases(
    clean: np.ndarray[Any, np.dtype[np.float32]], sr: int, seed: int
) -> dict[str, np.ndarray[Any, np.dtype[np.float32]]]:
    clean = _normalize_rms(clean, 0.08)
    n = len(clean)
    sig_rms = float(np.sqrt(np.mean(clean**2) + 1e-12))

    # 1. Clean control
    cases: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {"clean": clean.copy()}

    # 2. Heavy reverb (RT60 = 1.1s)
    rir_heavy = _make_rir(sr, rt60=1.1, seed=seed + 1)
    rev = signal.fftconvolve(clean, rir_heavy, mode="full")[:n]
    cases["reverb_heavy"] = _normalize_rms(rev.astype(np.float32), 0.08)

    # 3. Street noise + hum (+5 dB SNR)
    noise = _make_street_hum_noise(n, sr, seed=seed + 2)
    noise_gain = sig_rms * (10.0 ** (-5.0 / 20.0))
    cases["noise_street_hum"] = _normalize_rms(
        (clean + noise * noise_gain).astype(np.float32), 0.08
    )

    # 4. Combined reverb (RT60 = 0.8s) + noise (+7 dB SNR)
    rir_med = _make_rir(sr, rt60=0.8, seed=seed + 3)
    rev_med = signal.fftconvolve(clean, rir_med, mode="full")[:n]
    rev_rms = float(np.sqrt(np.mean(rev_med**2) + 1e-12))
    noise2 = _make_street_hum_noise(n, sr, seed=seed + 4)
    cases["reverb_plus_noise"] = _normalize_rms(
        (rev_med + noise2 * (rev_rms * 10.0 ** (-7.0 / 20.0))).astype(np.float32), 0.08
    )

    # 5. Speech + background music bleed (+4 dB SNR)
    music = _make_music_bed(n, sr, seed=seed + 5)
    music_gain = sig_rms * (10.0 ** (-4.0 / 20.0))
    cases["music_bleed"] = _normalize_rms((clean + music * music_gain).astype(np.float32), 0.08)

    # 6. Band-limited telephone / muffled lapel (3.8 kHz steep lowpass + slight noise)
    sos_nb = signal.butter(8, 3800.0, btype="lowpass", fs=sr, output="sos")
    nb = signal.sosfiltfilt(sos_nb, clean)
    nb_noise = _make_street_hum_noise(n, sr, seed=seed + 6) * (sig_rms * 10.0 ** (-22.0 / 20.0))
    cases["narrowband_phone"] = _normalize_rms((nb + nb_noise).astype(np.float32), 0.08)

    # 7. Hard-clipped hot recording
    hot = np.clip(clean * 3.5, -0.28, 0.28)
    cases["clipped_hot"] = _normalize_rms(hot.astype(np.float32), 0.08)
    return cases


def _delay_align(
    ref: np.ndarray[Any, np.dtype[np.float32]],
    est: np.ndarray[Any, np.dtype[np.float32]],
    max_shift: int = 2400,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.float32]]]:
    n = min(len(ref), len(est))
    r = ref[:n].astype(np.float64)
    e = est[:n].astype(np.float64)
    corr = signal.correlate(e, r, mode="full", method="fft")
    mid = n - 1
    lo = max(0, mid - max_shift)
    hi = min(len(corr), mid + max_shift + 1)
    shift = int(np.argmax(corr[lo:hi]) + lo - mid)
    if shift > 0:
        return ref[: n - shift], est[shift:n]
    if shift < 0:
        s = -shift
        return ref[s:n], est[: n - s]
    return ref[:n], est[:n]


def compute_si_sdr(
    ref: np.ndarray[Any, np.dtype[np.float32]], est: np.ndarray[Any, np.dtype[np.float32]]
) -> float:
    r, e = _delay_align(ref, est)
    r64 = r.astype(np.float64) - float(np.mean(r))
    e64 = e.astype(np.float64) - float(np.mean(e))
    denom = float(np.dot(r64, r64)) + 1e-12
    scale = float(np.dot(e64, r64)) / denom
    target = scale * r64
    noise = e64 - target
    return float(
        10.0
        * math.log10(
            (float(np.dot(target, target)) + 1e-12) / (float(np.dot(noise, noise)) + 1e-12)
        )
    )


def compute_lsd(
    ref: np.ndarray[Any, np.dtype[np.float32]],
    est: np.ndarray[Any, np.dtype[np.float32]],
    sr: int,
    f_lo: float,
    f_hi: float,
) -> float:
    r, e = _delay_align(ref, est)
    r = _normalize_rms(r, 0.08)
    e = _normalize_rms(e, 0.08)
    freqs, _, zr = signal.stft(r, fs=sr, nperseg=2048, noverlap=1536)
    _, _, ze = signal.stft(e, fs=sr, nperseg=2048, noverlap=1536)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    pr = np.log10(np.abs(zr[mask]) ** 2 + 1e-10)
    pe = np.log10(np.abs(ze[mask]) ** 2 + 1e-10)
    return float(np.mean(np.sqrt(np.mean((pr - pe) ** 2, axis=0))))


def merge_with_protected_band(
    observed: np.ndarray[Any, np.dtype[np.float32]],
    restored: np.ndarray[Any, np.dtype[np.float32]],
    sr: int,
    cutoff_hz: float = 3600.0,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    n = min(len(observed), len(restored))
    obs = observed[:n]
    res = restored[:n]
    n_fft, hop = 2048, 512
    _, _, z_obs = signal.stft(
        obs,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        boundary="zeros",
        padded=True,
    )
    _, _, z_res = signal.stft(
        res,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        boundary="zeros",
        padded=True,
    )
    mask = compute_transition_mask(z_obs.shape[0], sr, cutoff_hz=cutoff_hz, transition_hz=500.0)
    merged_z = merge_protected_spectrum(
        z_obs.astype(np.complex64), z_res.astype(np.complex64), mask, strength=1.0
    )
    _, merged = signal.istft(
        merged_z, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=True
    )
    return _match_length(np.asarray(merged, dtype=np.float32), n)


class QualityScorer:
    """Lazy-loaded NISQA v2 + DNSMOS + WavLM phonetic similarity scorer."""

    def __init__(self, device: str) -> None:
        self.device = device
        self._nisqa: Any = None
        self._dnsmos: Any = None
        self._wavlm: Any = None
        self._pesq_fn: Any = None
        self._stoi_fn: Any = None
        self._init_metrics()

    def _init_metrics(self) -> None:
        try:
            self._pesq_fn = importlib.import_module("pesq").pesq
        except Exception:
            self._pesq_fn = None
        try:
            self._stoi_fn = importlib.import_module("pystoi").stoi
        except Exception:
            self._stoi_fn = None
        try:
            nisqa_mod = importlib.import_module("torchmetrics.functional.audio.nisqa")
            dns_mod = importlib.import_module("torchmetrics.functional.audio.dnsmos")
            self._nisqa = getattr(nisqa_mod, "non_intrusive_speech_quality_assessment", None)
            self._dnsmos = getattr(dns_mod, "deep_noise_suppression_mean_opinion_score", None)
        except Exception:
            self._nisqa = None
            self._dnsmos = None
        try:
            torch = importlib.import_module("torch")
            tf = importlib.import_module("transformers")
            model = tf.WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
            model.eval().to(torch.device(self.device))
            self._wavlm = model
        except Exception as exc:
            print(f"[scorer] WavLM unavailable: {exc}", file=sys.stderr)
            self._wavlm = None

    def wavlm_similarity(
        self,
        ref_48k: np.ndarray[Any, np.dtype[np.float32]],
        est_48k: np.ndarray[Any, np.dtype[np.float32]],
    ) -> float | None:
        if self._wavlm is None:
            return None
        torch = importlib.import_module("torch")
        r, e = _delay_align(ref_48k, est_48k)
        r16 = _normalize_rms(_resample(r, SAMPLE_RATE, 16000), 0.08)
        e16 = _normalize_rms(_resample(e, SAMPLE_RATE, 16000), 0.08)
        n = min(len(r16), len(e16))
        with torch.no_grad():
            dev = torch.device(self.device)
            tr = torch.from_numpy(r16[:n]).unsqueeze(0).to(dev)
            te = torch.from_numpy(e16[:n]).unsqueeze(0).to(dev)
            hr = self._wavlm(tr, output_hidden_states=True).hidden_states[6].squeeze(0)
            he = self._wavlm(te, output_hidden_states=True).hidden_states[6].squeeze(0)
            # Score only speech-active frames (top 70% reference energy)
            t_frames = min(hr.shape[0], he.shape[0])
            hr, he = hr[:t_frames], he[:t_frames]
            frame_len = max(1, n // max(1, t_frames))
            energies = np.array(
                [
                    float(np.mean(r16[i * frame_len : (i + 1) * frame_len] ** 2))
                    for i in range(t_frames)
                ]
            )
            active = energies > (float(np.median(energies)) * 0.25)
            cos = torch.nn.functional.cosine_similarity(hr, he, dim=-1).cpu().numpy()
            return float(np.mean(cos[active])) if np.any(active) else float(np.mean(cos))

    def score(
        self,
        case: str,
        clip: str,
        engine: str,
        device: str,
        rtf: float,
        clean_48k: np.ndarray[Any, np.dtype[np.float32]],
        degraded_48k: np.ndarray[Any, np.dtype[np.float32]],
        est_48k: np.ndarray[Any, np.dtype[np.float32]],
    ) -> ClipMetrics:
        torch = importlib.import_module("torch")
        est_48k = _match_length(est_48k, len(clean_48k))
        si_sdr = compute_si_sdr(clean_48k, est_48k)
        lsd_full = compute_lsd(clean_48k, est_48k, SAMPLE_RATE, 50.0, 20000.0)
        lsd_high = compute_lsd(clean_48k, est_48k, SAMPLE_RATE, 4000.0, 20000.0)

        pesq_wb: float | None = None
        estoi_val: float | None = None
        r_a, e_a = _delay_align(clean_48k, est_48k)
        r16 = _resample(r_a, SAMPLE_RATE, 16000)
        e16 = _resample(e_a, SAMPLE_RATE, 16000)
        m16 = min(len(r16), len(e16))
        if self._pesq_fn is not None:
            try:
                pesq_wb = float(self._pesq_fn(16000, r16[:m16], e16[:m16], "wb"))
            except Exception:
                pesq_wb = None
        if self._stoi_fn is not None:
            try:
                estoi_val = float(self._stoi_fn(r16[:m16], e16[:m16], 16000, extended=True))
            except Exception:
                estoi_val = None

        pb_err: float | None = None
        if case == "narrowband_phone":
            ver = verify_protected_band_invariance(
                degraded_48k, est_48k, SAMPLE_RATE, cutoff_hz=3600.0
            )
            pb_err = float(ver.worst_band_energy_deviation_db)

        nisqa_vals: list[float | None] = [None] * 5
        if self._nisqa is not None:
            try:
                out = self._nisqa(torch.from_numpy(_normalize_rms(est_48k, 0.08)), SAMPLE_RATE)
                nisqa_vals = [float(v) for v in out.tolist()]
            except Exception as exc:
                print(f"[nisqa] failed on {engine}/{case}: {exc}", file=sys.stderr)

        dns_vals: list[float | None] = [None] * 3
        if self._dnsmos is not None:
            try:
                e16_norm = _normalize_rms(_resample(est_48k, SAMPLE_RATE, 16000), 0.08)
                out = self._dnsmos(
                    torch.from_numpy(e16_norm), 16000, personalized=False, cache_session=True
                )
                vals = [float(v) for v in out.tolist()]
                # torchmetrics returns [p808_mos, sig, bak, ovrl]
                dns_vals = [vals[3], vals[1], vals[2]]
            except Exception as exc:
                print(f"[dnsmos] failed on {engine}/{case}: {exc}", file=sys.stderr)

        wavlm_cos = self.wavlm_similarity(clean_48k, est_48k)
        return ClipMetrics(
            case=case,
            clip=clip,
            engine=engine,
            device=device,
            rtf=round(rtf, 4),
            si_sdr_db=round(si_sdr, 3),
            pesq_wb=round(pesq_wb, 3) if pesq_wb is not None else None,
            estoi=round(estoi_val, 4) if estoi_val is not None else None,
            lsd_full_db=round(lsd_full, 3),
            lsd_high_db=round(lsd_high, 3),
            protected_band_err_db=round(pb_err, 3) if pb_err is not None else None,
            nisqa_mos=round(nisqa_vals[0], 3) if nisqa_vals[0] is not None else None,
            nisqa_noi=round(nisqa_vals[1], 3) if nisqa_vals[1] is not None else None,
            nisqa_dis=round(nisqa_vals[2], 3) if nisqa_vals[2] is not None else None,
            nisqa_col=round(nisqa_vals[3], 3) if nisqa_vals[3] is not None else None,
            nisqa_loud=round(nisqa_vals[4], 3) if nisqa_vals[4] is not None else None,
            dnsmos_ovrl=round(dns_vals[0], 3) if dns_vals[0] is not None else None,
            dnsmos_sig=round(dns_vals[1], 3) if dns_vals[1] is not None else None,
            dnsmos_bak=round(dns_vals[2], 3) if dns_vals[2] is not None else None,
            wavlm_phonetic_cos=round(wavlm_cos, 4) if wavlm_cos is not None else None,
        )


def run_hawavoclean_venv_core(
    core: str, audio: np.ndarray[Any, np.dtype[np.float32]], sr: int
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float]:
    """Run existing HawaVoClean production or DFN3 core via main .venv for honest baseline."""
    venv_py = ROOT / ".venv/bin/python"
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "in.wav"
        out = Path(td) / "out.wav"
        sf.write(inp, audio, sr, subtype="FLOAT")
        code = (
            "import sys, time, soundfile as sf, numpy as np\n"
            "x, sr = sf.read(sys.argv[1], dtype='float32')\n"
            "if sys.argv[3] == 'dfn3':\n"
            "  from hawavoclean.enhancement.studio import StudioVoiceCore\n"
            "  c = StudioVoiceCore(device='cpu'); c.warmup()\n"
            "else:\n"
            "  from hawavoclean.enhancement.production import WienerSpectralEnhancer\n"
            "  c = WienerSpectralEnhancer(); c.warmup()\n"
            "t0 = time.perf_counter()\n"
            "res = c.enhance(np.asarray(x, dtype=np.float32), int(sr))\n"
            "dt = time.perf_counter() - t0\n"
            "sf.write(sys.argv[2], res.waveform, int(sr), subtype='FLOAT')\n"
            "print(f'{dt:.6f}')\n"
        )
        proc = subprocess.run(
            [str(venv_py), "-c", code, str(inp), str(out), core],
            capture_output=True,
            text=True,
            check=True,
        )
        dt = float(proc.stdout.strip().splitlines()[-1])
        y, _ = sf.read(out, dtype="float32")
        return _match_length(np.asarray(y, dtype=np.float32), len(audio)), dt / (len(audio) / sr)


class MossFormer2Runner:
    """ClearerVoice-Studio MossFormer2_SE_48K wrapper with device selection."""

    def __init__(self, task: str, model_name: str, device: str) -> None:
        self.task = task
        self.model_name = model_name
        self.device = device
        self._cv: Any = None

    def _ensure(self) -> None:
        if self._cv is not None:
            return
        torch = importlib.import_module("torch")
        clearvoice = importlib.import_module("clearvoice")
        self._cv = clearvoice.ClearVoice(task=self.task, model_names=[self.model_name])
        # Move underlying network to requested device (MPS/CPU)
        for model_wrap in getattr(self._cv, "models", []):
            net = getattr(model_wrap, "model", None)
            if net is not None and hasattr(net, "to"):
                net.to(torch.device(self.device))
            if hasattr(model_wrap, "device"):
                model_wrap.device = torch.device(self.device)

    def process(
        self, audio: np.ndarray[Any, np.dtype[np.float32]], sr: int
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float]:
        self._ensure()
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "in.wav"
            out = Path(td) / "out.wav"
            sf.write(inp, audio, sr, subtype="FLOAT")
            t0 = time.perf_counter()
            est = self._cv(input_path=str(inp), online_write=False)
            dt = time.perf_counter() - t0
            if isinstance(est, dict):
                est = next(iter(est.values()))
            y = np.asarray(est, dtype=np.float32).squeeze()
            if y.ndim > 1:
                y = y[0]
            if len(y) == 0:
                self._cv.write(est, output_path=str(out))
                y, _ = sf.read(out, dtype="float32")
            return _match_length(np.asarray(y, dtype=np.float32), len(audio)), dt / (
                len(audio) / sr
            )


class LavaSRRunner:
    """LavaSR v2 48 kHz restoration & BWE runner."""

    def __init__(self, device: str) -> None:
        self.device = device
        self._model: Any = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        mod = importlib.import_module("LavaSR.model")
        self._model = mod.LavaEnhance2("YatharthS/LavaSR", device=self.device)

    def process(
        self, audio: np.ndarray[Any, np.dtype[np.float32]], sr: int, denoise: bool = False
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float]:
        self._ensure()
        torch = importlib.import_module("torch")
        a16 = _resample(audio, sr, 16000)
        t0 = time.perf_counter()
        with torch.no_grad():
            tensor = torch.from_numpy(a16).unsqueeze(0)
            out = self._model.enhance(tensor, denoise=denoise, batch=False)
        dt = time.perf_counter() - t0
        y = np.asarray(out.detach().cpu().float().squeeze().numpy(), dtype=np.float32)
        return _match_length(y, len(audio)), dt / (len(audio) / sr)


class RoFormerMLXRunner:
    """MIT Mel-Band RoFormer vocal separator on Apple Silicon MLX."""

    def __init__(self) -> None:
        self._sep: Any = None

    def _ensure(self, out_dir: Path) -> Any:
        sep_mod = importlib.import_module("mlx_audio_separator.core")
        sep = sep_mod.Separator(output_dir=str(out_dir), output_format="WAV")
        sep.load_model("model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt")
        return sep

    def process(
        self, audio: np.ndarray[Any, np.dtype[np.float32]], sr: int
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float]:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            inp = td_path / "in.wav"
            # RoFormer expects stereo 44.1k/48k cleanly
            stereo = np.stack([audio, audio], axis=-1)
            sf.write(inp, stereo, sr, subtype="FLOAT")
            if self._sep is None:
                self._sep = self._ensure(td_path)
            else:
                self._sep.output_dir = str(td_path)
            t0 = time.perf_counter()
            files = self._sep.separate(str(inp))
            dt = time.perf_counter() - t0
            vocal_file = next(
                (Path(f) for f in files if "vocal" in Path(f).name.lower()), Path(files[0])
            )
            if not vocal_file.is_absolute():
                vocal_file = td_path / vocal_file
            y, out_sr = sf.read(vocal_file, dtype="float32")
            if y.ndim == 2:
                y = np.mean(y, axis=1)
            y48 = _resample(np.asarray(y, dtype=np.float32), int(out_sr), sr)
            return _match_length(y48, len(audio)), dt / (len(audio) / sr)


def _format_markdown(rows: list[ClipMetrics]) -> str:
    # Aggregate by (case, engine)
    groups: dict[tuple[str, str], list[ClipMetrics]] = {}
    for r in rows:
        groups.setdefault((r.case, r.engine), []).append(r)

    def _avg(vals: list[float | None], fmt: str) -> str:
        clean = [v for v in vals if v is not None and math.isfinite(v)]
        return format(sum(clean) / len(clean), fmt) if clean else "—"

    lines = [
        "# Phase 1 SOTA Audio Shootout — Real 48 kHz Sorani Kurdish Speech",
        "",
        "| Case | Engine | Device | RTF | SI-SDR (dB) | PESQ-WB | ESTOI | NISQA MOS | Discont. | Color. | DNSMOS Ovrl | WavLM Cos | LSD High | PB Err (dB) |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in sorted({r.case for r in rows}):
        case_engines = [eng for c, eng in groups if c == case]
        for eng in case_engines:
            items = groups[(case, eng)]
            lines.append(
                f"| `{case}` | **{eng}** | `{items[0].device}` | "
                f"{_avg([i.rtf for i in items], '.3f')} | "
                f"{_avg([i.si_sdr_db for i in items], '.2f')} | "
                f"{_avg([i.pesq_wb for i in items], '.2f')} | "
                f"{_avg([i.estoi for i in items], '.3f')} | "
                f"{_avg([i.nisqa_mos for i in items], '.2f')} | "
                f"{_avg([i.nisqa_dis for i in items], '.2f')} | "
                f"{_avg([i.nisqa_col for i in items], '.2f')} | "
                f"{_avg([i.dnsmos_ovrl for i in items], '.2f')} | "
                f"{_avg([i.wavlm_phonetic_cos for i in items], '.4f')} | "
                f"{_avg([i.lsd_high_db for i in items], '.2f')} | "
                f"{_avg([i.protected_band_err_db for i in items], '.2f')} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / ".local-ai/bench-results")
    parser.add_argument("--clips", type=int, default=2, help="Number of real Sorani clips (1-4)")
    parser.add_argument("--skip-roformer", action="store_true")
    parser.add_argument("--skip-sr", action="store_true")
    args = parser.parse_args()

    torch = importlib.import_module("torch")
    mps_ok = bool(torch.backends.mps.is_available())
    accel = "mps" if mps_ok else "cpu"
    print(f"[shootout] Python {sys.version.split()[0]}, torch {torch.__version__}, accel={accel}")

    kurdish_files = sorted((ROOT / "data/kurdish_speakers").glob("*/*.wav"))
    if not kurdish_files:
        raise SystemExit("No real Kurdish speech files found in data/kurdish_speakers/")
    # Pick speaker-balanced subset (1 fatih + 1 shahen for --clips 2, all 4 for --clips 4)
    selected = (
        [kurdish_files[0], kurdish_files[-1]] if args.clips == 2 else kurdish_files[: args.clips]
    )
    print(f"[shootout] Clips: {[p.name for p in selected]}")

    out_dir: Path = args.out
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    scorer = QualityScorer(device=accel)
    mf2_mps = MossFormer2Runner("speech_enhancement", "MossFormer2_SE_48K", device=accel)
    mf2_cpu = MossFormer2Runner("speech_enhancement", "MossFormer2_SE_48K", device="cpu")
    lavasr = LavaSRRunner(device=accel)
    roformer = None if args.skip_roformer else RoFormerMLXRunner()
    mf2_sr = (
        None
        if args.skip_sr
        else MossFormer2Runner("speech_super_resolution", "MossFormer2_SR_48K", device=accel)
    )

    results: list[ClipMetrics] = []
    for clip_idx, clip_path in enumerate(selected):
        raw, sr = sf.read(clip_path, dtype="float32")
        if raw.ndim == 2:
            raw = np.mean(raw, axis=1)
        clean = _resample(np.asarray(raw, dtype=np.float32), int(sr), SAMPLE_RATE)
        cases = build_stress_cases(clean, SAMPLE_RATE, seed=100 + clip_idx * 17)
        clip_name = clip_path.stem

        for case_name, degraded in cases.items():
            print(f"\n=== {clip_name} :: {case_name} ===", flush=True)
            sf.write(
                audio_dir / f"{clip_name}__{case_name}__degraded.wav",
                degraded,
                SAMPLE_RATE,
                subtype="PCM_24",
            )

            # 0. Degraded baseline
            results.append(
                scorer.score(
                    case_name, clip_name, "0_degraded", "none", 0.0, clean, degraded, degraded
                )
            )

            # 1. Existing HawaVoClean baselines
            for core_key, label in (("production", "1_hawa_wiener"), ("dfn3", "2_hawa_dfn3_cpu")):
                try:
                    y, rtf = run_hawavoclean_venv_core(core_key, degraded, SAMPLE_RATE)
                    sf.write(
                        audio_dir / f"{clip_name}__{case_name}__{label}.wav",
                        _normalize_rms(y),
                        SAMPLE_RATE,
                        subtype="PCM_24",
                    )
                    results.append(
                        scorer.score(case_name, clip_name, label, "cpu", rtf, clean, degraded, y)
                    )
                except Exception as exc:
                    print(f"  [{label}] ERROR: {exc}", file=sys.stderr)

            # 2. MossFormer2_SE_48K (MPS + CPU timing comparison on first clip)
            mf2_out: np.ndarray[Any, np.dtype[np.float32]] | None = None
            try:
                y_mf2, rtf_mf2 = mf2_mps.process(degraded, SAMPLE_RATE)
                mf2_out = y_mf2
                sf.write(
                    audio_dir / f"{clip_name}__{case_name}__3_mossformer2_48k.wav",
                    _normalize_rms(y_mf2),
                    SAMPLE_RATE,
                    subtype="PCM_24",
                )
                results.append(
                    scorer.score(
                        case_name,
                        clip_name,
                        "3_mossformer2_48k",
                        accel,
                        rtf_mf2,
                        clean,
                        degraded,
                        y_mf2,
                    )
                )
            except Exception as exc:
                print(f"  [3_mossformer2_48k] ERROR: {exc}", file=sys.stderr)

            if clip_idx == 0 and case_name == "reverb_plus_noise":
                try:
                    y_cpu, rtf_cpu = mf2_cpu.process(degraded, SAMPLE_RATE)
                    results.append(
                        scorer.score(
                            case_name,
                            clip_name,
                            "3b_mossformer2_48k_cpu",
                            "cpu",
                            rtf_cpu,
                            clean,
                            degraded,
                            y_cpu,
                        )
                    )
                except Exception as exc:
                    print(f"  [3b_mossformer2_48k_cpu] ERROR: {exc}", file=sys.stderr)

            # 3. Bandwidth extension / restoration cases (narrowband_phone + reverb_plus_noise + clipped_hot + clean)
            if case_name in {"narrowband_phone", "reverb_plus_noise", "clipped_hot", "clean"}:
                try:
                    y_lava, rtf_lava = lavasr.process(
                        degraded, SAMPLE_RATE, denoise=(case_name != "narrowband_phone")
                    )
                    sf.write(
                        audio_dir / f"{clip_name}__{case_name}__4a_lavasr_raw.wav",
                        _normalize_rms(y_lava),
                        SAMPLE_RATE,
                        subtype="PCM_24",
                    )
                    results.append(
                        scorer.score(
                            case_name,
                            clip_name,
                            "4a_lavasr_raw",
                            accel,
                            rtf_lava,
                            clean,
                            degraded,
                            y_lava,
                        )
                    )
                    if case_name == "narrowband_phone":
                        y_guarded = merge_with_protected_band(
                            degraded, y_lava, SAMPLE_RATE, cutoff_hz=3600.0
                        )
                        sf.write(
                            audio_dir / f"{clip_name}__{case_name}__4b_lavasr_guarded.wav",
                            _normalize_rms(y_guarded),
                            SAMPLE_RATE,
                            subtype="PCM_24",
                        )
                        results.append(
                            scorer.score(
                                case_name,
                                clip_name,
                                "4b_lavasr_guarded",
                                accel,
                                rtf_lava,
                                clean,
                                degraded,
                                y_guarded,
                            )
                        )
                except Exception as exc:
                    print(f"  [lavasr] ERROR: {exc}", file=sys.stderr)

                if mf2_out is not None and case_name in {"narrowband_phone", "reverb_plus_noise"}:
                    try:
                        y_combo_raw, rtf_combo = lavasr.process(mf2_out, SAMPLE_RATE, denoise=False)
                        y_combo = merge_with_protected_band(
                            mf2_out,
                            y_combo_raw,
                            SAMPLE_RATE,
                            cutoff_hz=3600.0 if case_name == "narrowband_phone" else 7500.0,
                        )
                        sf.write(
                            audio_dir / f"{clip_name}__{case_name}__5_mf2_plus_lavasr_guarded.wav",
                            _normalize_rms(y_combo),
                            SAMPLE_RATE,
                            subtype="PCM_24",
                        )
                        results.append(
                            scorer.score(
                                case_name,
                                clip_name,
                                "5_mf2_plus_lavasr_guarded",
                                accel,
                                rtf_combo,
                                clean,
                                degraded,
                                y_combo,
                            )
                        )
                    except Exception as exc:
                        print(f"  [5_mf2_plus_lavasr_guarded] ERROR: {exc}", file=sys.stderr)

                if mf2_sr is not None and case_name == "narrowband_phone":
                    try:
                        y_sr, rtf_sr = mf2_sr.process(degraded, SAMPLE_RATE)
                        y_sr_g = merge_with_protected_band(
                            degraded, y_sr, SAMPLE_RATE, cutoff_hz=3600.0
                        )
                        sf.write(
                            audio_dir
                            / f"{clip_name}__{case_name}__6_mossformer2_sr_48k_guarded.wav",
                            _normalize_rms(y_sr_g),
                            SAMPLE_RATE,
                            subtype="PCM_24",
                        )
                        results.append(
                            scorer.score(
                                case_name,
                                clip_name,
                                "6_mossformer2_sr_48k_guarded",
                                accel,
                                rtf_sr,
                                clean,
                                degraded,
                                y_sr_g,
                            )
                        )
                    except Exception as exc:
                        print(f"  [6_mossformer2_sr_48k] ERROR: {exc}", file=sys.stderr)

            # 4. Vocal/music separation (Track C: Mel-Band RoFormer on Apple MLX)
            if roformer is not None and case_name in {"music_bleed", "reverb_plus_noise"}:
                try:
                    y_rof, rtf_rof = roformer.process(degraded, SAMPLE_RATE)
                    sf.write(
                        audio_dir / f"{clip_name}__{case_name}__7_roformer_mlx.wav",
                        _normalize_rms(y_rof),
                        SAMPLE_RATE,
                        subtype="PCM_24",
                    )
                    results.append(
                        scorer.score(
                            case_name,
                            clip_name,
                            "7_roformer_mlx",
                            "mlx",
                            rtf_rof,
                            clean,
                            degraded,
                            y_rof,
                        )
                    )
                    if case_name == "music_bleed":
                        # Chained: RoFormer vocal extraction -> MossFormer2 cleanup
                        y_chain, rtf_c = mf2_mps.process(y_rof, SAMPLE_RATE)
                        sf.write(
                            audio_dir / f"{clip_name}__{case_name}__8_roformer_then_mf2.wav",
                            _normalize_rms(y_chain),
                            SAMPLE_RATE,
                            subtype="PCM_24",
                        )
                        results.append(
                            scorer.score(
                                case_name,
                                clip_name,
                                "8_roformer_then_mf2",
                                "mlx+mps",
                                rtf_rof + rtf_c,
                                clean,
                                degraded,
                                y_chain,
                            )
                        )
                except Exception as exc:
                    print(f"  [7_roformer_mlx] ERROR: {exc}", file=sys.stderr)

    (out_dir / "scorecard.json").write_text(json.dumps([asdict(r) for r in results], indent=2))
    md = _format_markdown(results)
    (out_dir / "scorecard.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
