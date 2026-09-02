"""Tests for the research-grade metrics module (C1)."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.eval.metrics import (
    MetricsResult,
    _compute_lsd,
    _compute_si_snr,
    compute_corpus_metrics,
    compute_metrics,
)


def _write_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> Path:
    """Write a mono float32 WAV file."""
    sf.write(str(path), samples.astype(np.float32), sr)
    return path


@pytest.fixture()
def clean_sine(tmp_path: Path) -> Path:
    """1 second 440Hz sine at 16 kHz."""
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    return _write_wav(tmp_path / "clean.wav", np.sin(2 * np.pi * 440 * t).astype(np.float32))


@pytest.fixture()
def noisy_sine(tmp_path: Path) -> Path:
    """Same sine + light noise."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    noise = 0.05 * rng.standard_normal(16000).astype(np.float32)
    return _write_wav(tmp_path / "noisy.wav", signal + noise)


class TestSISNR:
    """Scale-Invariant Signal-to-Noise Ratio."""

    def test_identity_is_high(self) -> None:
        """Identical signals should give very high SI-SNR."""
        x = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
        assert _compute_si_snr(x, x) > 100.0

    def test_noise_lowers_si_snr(self) -> None:
        """Adding noise should decrease SI-SNR."""
        rng = np.random.default_rng(42)
        x = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
        noisy = x + 0.1 * rng.standard_normal(1000).astype(np.float32)
        assert _compute_si_snr(x, noisy) < _compute_si_snr(x, x)

    def test_orthogonal_is_low(self) -> None:
        """Orthogonal signals should give low SI-SNR."""
        x = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
        y = np.cos(np.linspace(0, 10, 1000)).astype(np.float32)
        assert _compute_si_snr(x, y) < 20.0

    def test_silent_reference(self) -> None:
        """Silent reference should return 0."""
        x = np.zeros(1000, dtype=np.float32)
        y = np.ones(1000, dtype=np.float32)
        assert _compute_si_snr(x, y) == 0.0


class TestLSD:
    """Log-Spectral Distance."""

    def test_identity_is_zero(self) -> None:
        """Identical signals should have LSD near zero."""
        x = np.sin(np.linspace(0, 10, 4096)).astype(np.float32)
        assert _compute_lsd(x, x) < 0.01

    def test_different_signals_positive(self) -> None:
        """Different signals should have positive LSD."""
        rng = np.random.default_rng(42)
        x = np.sin(np.linspace(0, 10, 4096)).astype(np.float32)
        y = x + 0.3 * rng.standard_normal(4096).astype(np.float32)
        assert _compute_lsd(x, y) > 0.0


class TestComputeMetrics:
    """End-to-end metrics computation."""

    def test_returns_all_fields(self, clean_sine: Path, noisy_sine: Path) -> None:
        """All metric fields should be populated."""
        result = compute_metrics(clean_sine, noisy_sine)
        assert isinstance(result, MetricsResult)
        assert result.si_snr_db != 0.0
        assert result.lsd_db > 0.0
        assert result.separation_db >= 0.0
        assert result.duration_s > 0.0
        assert result.compute_time_s > 0.0
        # PESQ and ESTOI may be None if packages not installed
        assert result.warnings is not None

    def test_identity_metrics(self, clean_sine: Path) -> None:
        """Comparing a file against itself should give perfect scores."""
        result = compute_metrics(clean_sine, clean_sine)
        assert result.si_snr_db > 100.0
        assert result.lsd_db < 0.01

    def test_hashes_are_sha256(self, clean_sine: Path, noisy_sine: Path) -> None:
        """Hashes should be valid hex SHA-256 strings."""
        result = compute_metrics(clean_sine, noisy_sine)
        assert len(result.reference_sha256) == 64
        assert len(result.candidate_sha256) == 64


class TestCorpusMetrics:
    """Corpus-level aggregate computation."""

    def test_corpus_aggregation(self, clean_sine: Path, noisy_sine: Path, tmp_path: Path) -> None:
        """Corpus metrics should aggregate individual pair results."""
        pairs = [(clean_sine, noisy_sine), (clean_sine, clean_sine)]
        out_path = tmp_path / "corpus_results.json"
        report = compute_corpus_metrics(pairs, output_path=out_path)

        assert report["total_pairs"] == 2
        assert out_path.exists()

        # Verify JSON is valid and contains aggregate stats
        data = json.loads(out_path.read_text())
        assert "aggregate" in data
        assert "per_pair" in data
        assert len(data["per_pair"]) == 2

        # SI-SNR aggregate must exist
        si_snr_stats = data["aggregate"]["si_snr_db"]
        assert si_snr_stats is not None
        assert si_snr_stats["n"] == 2

    def test_empty_corpus(self) -> None:
        """Empty corpus should produce empty aggregates."""
        report = compute_corpus_metrics([])
        assert report["total_pairs"] == 0

    def test_load_mono_fallback(self, clean_sine: Path) -> None:
        from unittest.mock import patch

        from hawavoclean.eval.metrics import _load_mono

        with patch("hawavoclean.audio.probe.probe_audio", side_effect=RuntimeError("probe failed")):
            mono, sr = _load_mono(clean_sine)
            assert sr == 16000
            assert len(mono) == 16000

    def test_sample_rate_mismatch_and_mocked_pesq_estoi(self, tmp_path: Path) -> None:
        import sys
        import types
        from unittest.mock import patch

        t_ref = np.linspace(0, 1.0, 16000, endpoint=False)
        ref_16k = _write_wav(
            tmp_path / "ref_16k.wav", np.sin(2 * np.pi * 440 * t_ref).astype(np.float32), sr=16000
        )

        t_cand = np.linspace(0, 1.0, 48000, endpoint=False)
        cand_16k = _write_wav(
            tmp_path / "cand_48k.wav", np.sin(2 * np.pi * 440 * t_cand).astype(np.float32), sr=48000
        )

        fake_pesq_mod = types.ModuleType("pesq")
        fake_pesq_mod.pesq = lambda *_a, **_kw: 3.8  # type: ignore[attr-defined]

        fake_pystoi_mod = types.ModuleType("pystoi")
        fake_pystoi_mod.stoi = lambda *_a, **_kw: 0.95  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"pesq": fake_pesq_mod, "pystoi": fake_pystoi_mod}):
            res = compute_metrics(ref_16k, cand_16k)
            assert res.pesq_wb == 3.8
            assert res.estoi == 0.95
            assert any("Resampled candidate" in w for w in res.warnings)

            # Test corpus aggregation with pesq and estoi populated
            corp = compute_corpus_metrics([(ref_16k, cand_16k)])
            assert corp["aggregate"]["pesq_wb"]["mean"] == 3.8
            assert corp["aggregate"]["estoi"]["mean"] == 0.95

    def test_compute_metrics_pesq_pystoi_exception_handling(
        self, clean_sine: Path, noisy_sine: Path
    ) -> None:
        """Verify fallback None when pesq or pystoi raise exceptions."""
        import sys
        import types
        from unittest.mock import patch

        def broken_pesq(*_args: object, **_kwargs: object) -> float:
            raise ValueError("pesq math overflow")

        def broken_stoi(*_args: object, **_kwargs: object) -> float:
            raise RuntimeError("stoi crashed")

        fake_pesq_mod = types.ModuleType("pesq")
        fake_pesq_mod.pesq = broken_pesq  # type: ignore[attr-defined]

        fake_pystoi_mod = types.ModuleType("pystoi")
        fake_pystoi_mod.stoi = broken_stoi  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"pesq": fake_pesq_mod, "pystoi": fake_pystoi_mod}):
            res = compute_metrics(clean_sine, noisy_sine)
            assert res.pesq_wb is None
            assert res.estoi is None

    def test_48k_reference_resampling(self, tmp_path: Path) -> None:
        import sys
        import types
        from unittest.mock import patch

        t = np.linspace(0, 1.0, 48000, endpoint=False)
        ref_48k = _write_wav(
            tmp_path / "ref_48k.wav", np.sin(2 * np.pi * 440 * t).astype(np.float32), sr=48000
        )
        cand_48k = _write_wav(
            tmp_path / "cand_48k_same.wav", np.sin(2 * np.pi * 440 * t).astype(np.float32), sr=48000
        )

        fake_pesq_mod = types.ModuleType("pesq")
        fake_pesq_mod.pesq = lambda *_a, **_kw: 4.1  # type: ignore[attr-defined]

        fake_pystoi_mod = types.ModuleType("pystoi")
        fake_pystoi_mod.stoi = lambda *_a, **_kw: 0.98  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"pesq": fake_pesq_mod, "pystoi": fake_pystoi_mod}):
            res = compute_metrics(ref_48k, cand_48k)
            assert res.pesq_wb == 4.1
            assert res.estoi == 0.98
