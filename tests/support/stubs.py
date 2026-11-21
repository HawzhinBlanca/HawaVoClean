"""Test-only enhancer and probe stubs moved out of production modules."""

from __future__ import annotations

from typing import Any

import numpy as np

from hawavoclean.enhancement.protocol import EnhancementResult, Enhancer, EnhancerMetadata
from hawavoclean.guard.protocol import ProbeResult, SpectralProbe, TokenInfo
from hawavoclean.guard.sorani_normalize import normalize_sorani_text
from hawavoclean.guard.spectral_probe import SORANI_VOCAB
from hawavoclean.hashing import hash_bytes, hash_json_canonical


class NoOpEnhancer(Enhancer):
    """No-op passthrough enhancer for baseline comparisons and unit testing."""

    def __init__(self, sample_rate: int = 48000) -> None:
        self._sample_rate = sample_rate
        self._metadata = EnhancerMetadata(
            core_id="noop-enhancer",
            version="1.0.0",
            algorithm="identity passthrough",
            sample_rate=sample_rate,
            phase_coherent=True,
            params_hash=hash_json_canonical({"algorithm": "noop"}),
        )

    @property
    def metadata(self) -> EnhancerMetadata:
        return self._metadata

    def warmup(self) -> None:
        pass

    def enhance(
        self,
        waveform: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int,
    ) -> EnhancementResult:
        return EnhancementResult(
            waveform=waveform.copy(),
            sample_rate=sample_rate,
            model_runtime_ms=0.5,
            input_samples=len(waveform),
            output_samples=len(waveform),
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
