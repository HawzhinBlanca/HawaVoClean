"""Spectral signature probe: deterministic spectral-change detection.

WHAT THIS IS: a comparator primitive. It reduces audio to a per-frame
distribution over an arbitrary symbol alphabet, driven purely by the shape
of the low-frequency spectrum (the first 80 linear FFT bins at 16 kHz,
roughly 0-2.5 kHz — no mel warping is applied). Two renderings of the same
audio produce similar signatures; a rendering whose spectrum changed
produces a diverging one.

WHAT THIS IS NOT: a speech recognizer. There is no acoustic model, no
learned mapping from audio to phonemes, and no transcription. The symbol
alphabet reuses Kurdish Sorani orthography purely as a stable set of glyphs
(kept because a future trained Sorani model would target the same
vocabulary); the symbols DO NOT correspond to recognized phonemes, and the
emitted signature is NOT a transcript. This probe can detect that the
spectrum changed; it cannot detect a word substitution.
"""

from typing import Any

import numpy as np

from voiceclean.audio.resample import resample_audio
from voiceclean.guard.protocol import ProbeResult, SpectralProbe, TokenInfo
from voiceclean.guard.sorani_normalize import normalize_sorani_text
from voiceclean.hashing import hash_bytes

# Symbol alphabet for signature emission. Sorani glyphs are used as opaque
# symbols only — see the module docstring.
SORANI_VOCAB: list[str] = [
    "<blank>",  # 0: silence / no dominant spectral state
    " ",  # 1: space
    "ئ",
    "ا",
    "ب",
    "پ",
    "ت",
    "ج",
    "چ",
    "ح",
    "خ",
    "د",
    "ر",
    "ڕ",
    "ز",
    "ژ",
    "س",
    "ش",
    "ع",
    "غ",
    "ف",
    "ڤ",
    "ق",
    "ک",
    "گ",
    "ل",
    "ڵ",
    "م",
    "ن",
    "و",
    "ۆ",
    "وو",
    "ه",
    "ە",
    "ی",
    "ێ",
]

VOCAB_TO_ID: dict[str, int] = {char: idx for idx, char in enumerate(SORANI_VOCAB)}


class SpectralSignatureProbe(SpectralProbe):
    """Deterministic spectral-shape signature probe."""

    def __init__(
        self,
        probe_id: str = "spectral-signature-v1",
        target_sr: int = 16000,
        frame_step_ms: float = 20.0,
    ) -> None:
        self._probe_id = probe_id
        self._target_sr = target_sr
        self._frame_step_ms = frame_step_ms
        self._probe_hash = hash_bytes(f"{probe_id}:{target_sr}:{len(SORANI_VOCAB)}".encode())

    @property
    def probe_id(self) -> str:
        return self._probe_id

    @property
    def probe_hash(self) -> str:
        return self._probe_hash

    def _compute_spectral_features(
        self,
        waveform: np.ndarray[Any, np.dtype[np.float32]],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Short-term log spectral features: the first 80 linear FFT bins."""
        sr = self._target_sr
        n_fft = 512
        hop_length = int(round(sr * (self._frame_step_ms / 1000.0)))
        win_length = int(round(sr * 0.025))  # 25ms window

        if len(waveform) < win_length:
            waveform = np.pad(waveform, (0, win_length - len(waveform)))

        window = np.hanning(win_length)
        num_frames = max(1, (len(waveform) - win_length) // hop_length + 1)
        stft = np.zeros((num_frames, n_fft // 2 + 1), dtype=np.float32)

        for i in range(num_frames):
            chunk = waveform[i * hop_length : i * hop_length + win_length] * window
            fft_mag = np.abs(np.fft.rfft(chunk, n=n_fft))
            stft[i] = fft_mag

        # First 80 linear bins (0 to ~2.5 kHz at 16 kHz), log-compressed.
        n_bins = 80
        return np.log1p(stft[:, :n_bins]).astype(np.float32)

    def infer(
        self,
        waveform: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int,
    ) -> ProbeResult:
        """Compute the per-frame spectral symbol distribution and signature."""
        if len(waveform) == 0:
            return ProbeResult(
                raw_signature="",
                normalized_signature="",
                probe_id=self.probe_id,
                probe_hash=self.probe_hash,
            )

        # Resample to the probe's internal rate (16kHz)
        audio_16k = resample_audio(waveform, sample_rate, self._target_sr)
        features = self._compute_spectral_features(audio_16k)
        # Temporal smoothing (40ms) to stabilize the frame states
        import scipy.ndimage

        smooth_features = scipy.ndimage.gaussian_filter1d(features, sigma=2.0, axis=0)
        num_frames = len(smooth_features)

        vocab_size = len(SORANI_VOCAB)
        distributions = np.zeros((num_frames, vocab_size), dtype=np.float32)

        # Map spectral shape to a symbol distribution: each symbol slot reads
        # one normalized frequency bin. This is an arbitrary but deterministic
        # assignment — it makes signatures comparable, not meaningful.
        for i in range(num_frames):
            frame_feat = smooth_features[i]
            frame_energy = float(np.sum(frame_feat))

            if frame_energy < 0.1:
                # Near-silence: dominant blank state
                distributions[i, 0] = 0.95
                distributions[i, 1:] = 0.05 / (vocab_size - 1)
            else:
                distributions[i, 0] = 0.15  # blank share
                feat_min = float(np.min(frame_feat))
                feat_ptp = float(np.ptp(frame_feat)) + 1e-6
                norm_feat = (frame_feat - feat_min) / feat_ptp

                char_logits = np.zeros(vocab_size - 1, dtype=np.float32)
                for c_idx in range(vocab_size - 1):
                    bin_idx = (c_idx * len(norm_feat)) // (vocab_size - 1)
                    char_logits[c_idx] = norm_feat[bin_idx] * 6.0

                exp_logits = np.exp(char_logits - np.max(char_logits))
                char_probs = (exp_logits / np.sum(exp_logits)) * 0.85
                distributions[i, 1:] = char_probs

        # Frame timestamps
        frame_times = np.arange(num_frames, dtype=np.float32) * (self._frame_step_ms / 1000.0)

        # Collapse repeated frame states into sustained tokens (>=40ms)
        best_indices = np.argmax(distributions, axis=1)
        tokens: list[TokenInfo] = []
        raw_chars: list[str] = []

        prev_idx = 0
        token_start_time = 0.0
        token_confidences: list[float] = []

        for f_idx, idx in enumerate(best_indices):
            t_s = float(frame_times[f_idx])
            prob = float(distributions[f_idx, idx])

            if idx != prev_idx:
                if prev_idx != 0 and len(token_confidences) >= 2:  # sustained state (>=40ms)
                    char_str = SORANI_VOCAB[prev_idx]
                    token_conf = float(np.max(token_confidences)) if token_confidences else 0.8
                    tokens.append(
                        TokenInfo(
                            token_id=prev_idx,
                            text=char_str,
                            start_time_s=token_start_time,
                            end_time_s=t_s,
                            confidence=token_conf,
                        )
                    )
                    raw_chars.append(char_str)

                token_start_time = t_s
                token_confidences = [prob]
                prev_idx = idx
            else:
                token_confidences.append(prob)

        # Flush trailing token
        if prev_idx != 0 and len(token_confidences) >= 2:
            char_str = SORANI_VOCAB[prev_idx]
            token_conf = float(np.max(token_confidences)) if token_confidences else 0.8
            tokens.append(
                TokenInfo(
                    token_id=prev_idx,
                    text=char_str,
                    start_time_s=token_start_time,
                    end_time_s=float(frame_times[-1]) if len(frame_times) > 0 else 0.0,
                    confidence=token_conf,
                )
            )
            raw_chars.append(char_str)

        raw_signature = "".join(raw_chars)
        norm_audit = normalize_sorani_text(raw_signature)
        mean_conf = float(np.mean([t.confidence for t in tokens])) if tokens else 1.0

        return ProbeResult(
            raw_signature=raw_signature,
            normalized_signature=norm_audit.normalized,
            tokens=tokens,
            frame_distributions=distributions,
            frame_timestamps=frame_times,
            mean_confidence=mean_conf,
            probe_id=self.probe_id,
            probe_hash=self.probe_hash,
        )


class FixedProbe(SpectralProbe):
    """Deterministic fixture probe emitting a constant signature, for tests."""

    def __init__(
        self,
        fixed_signature: str = "سڵاو لە هەمووان ئەمە تاقیکردنەوەیە",
        confidence: float = 0.95,
        probe_id: str = "fixed-probe",
    ) -> None:
        self._signature = fixed_signature
        self._confidence = confidence
        self._probe_id = probe_id
        self._probe_hash = hash_bytes(f"fixed:{fixed_signature}".encode())

    @property
    def probe_id(self) -> str:
        return self._probe_id

    @property
    def probe_hash(self) -> str:
        return self._probe_hash

    def infer(
        self,
        waveform: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int,
    ) -> ProbeResult:
        dur = len(waveform) / sample_rate
        norm = normalize_sorani_text(self._signature)
        words = norm.normalized.split()

        tokens: list[TokenInfo] = []
        if words:
            step = dur / len(words)
            for i, w in enumerate(words):
                tokens.append(
                    TokenInfo(
                        token_id=i + 1,
                        text=w,
                        start_time_s=i * step,
                        end_time_s=(i + 1) * step,
                        confidence=self._confidence,
                    )
                )

        num_frames = max(1, int(dur * 50))
        distributions = (
            np.ones((num_frames, len(SORANI_VOCAB)), dtype=np.float32) / len(SORANI_VOCAB)
        ).astype(np.float32)
        timestamps = np.linspace(0.0, dur, num_frames, dtype=np.float32)

        return ProbeResult(
            raw_signature=self._signature,
            normalized_signature=norm.normalized,
            tokens=tokens,
            frame_distributions=distributions,
            frame_timestamps=timestamps,
            mean_confidence=self._confidence,
            probe_id=self.probe_id,
            probe_hash=self.probe_hash,
        )
