"""Unit tests for the UniverSR upstream baseline restorer."""

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import numpy as np
import pytest
import scipy.signal as signal
import torch

from hawavoclean.errors import ModelProvenanceError
from hawavoclean.hashing import hash_file
from hawavoclean.restoration import universr_upstream
from hawavoclean.restoration.config import RestorationGuardConfig
from hawavoclean.restoration.protected_band import verify_protected_band_invariance
from hawavoclean.restoration.universr_upstream import UniverSRBaseline

SR = 48000


def _lowpassed_signal(duration_s: float = 0.3, cutoff_hz: float = 4000.0) -> np.ndarray:
    """Band-limited deterministic test signal: tones lowpassed below cutoff."""
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False, dtype=np.float32)
    raw = (0.5 * np.sin(2 * np.pi * 300.0 * t) + 0.3 * np.sin(2 * np.pi * 2500.0 * t)).astype(
        np.float32
    )
    sos = signal.butter(6, cutoff_hz / (SR / 2), btype="lowpass", output="sos")
    return np.asarray(signal.sosfiltfilt(sos, raw), dtype=np.float32)


def _high_band_power(audio: np.ndarray, floor_hz: float = 6000.0) -> float:
    freqs, psd = signal.welch(audio, fs=SR, nperseg=2048)
    return float(np.sum(psd[freqs > floor_hz]))


class _FakeNeuralUniverSR:
    """Stands in for the official flow-matching model and records every inference request."""

    def __init__(self, trim_to: int | None = None, fail: bool = False) -> None:
        self.trim_to = trim_to
        self.fail = fail
        self.sr_khz_calls: list[int] = []
        self.input_lengths: list[int] = []
        self.ode_calls: list[tuple[str, int, float]] = []

    def _inference(
        self,
        audio: torch.Tensor,
        sr_khz: int,
        ode_method: str,
        ode_steps: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        self.sr_khz_calls.append(sr_khz)
        self.input_lengths.append(int(audio.shape[-1]))
        self.ode_calls.append((ode_method, ode_steps, guidance_scale))
        if self.fail:
            raise RuntimeError("synthetic neural failure")
        if self.trim_to is not None:
            return audio[..., : self.trim_to]
        return audio


class _FakeOfficialUniverSR:
    """Replaces the vendored ``universr.inference.UniverSR`` entry point during init tests."""

    from_pretrained_calls: ClassVar[list[tuple[str, str]]] = []

    @classmethod
    def from_pretrained(cls, repo_id: str, device: str) -> "_FakeOfficialUniverSR":
        cls.from_pretrained_calls.append((repo_id, device))
        return cls()


def _dsp_baseline() -> UniverSRBaseline:
    return UniverSRBaseline(sample_rate=SR, use_neural=False, device="cpu")


def _install_fake_vendor_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``from universr.inference import UniverSR`` resolve to the fake entry point."""
    fake_pkg = ModuleType("universr")
    fake_inference = ModuleType("universr.inference")
    setattr(fake_inference, "UniverSR", _FakeOfficialUniverSR)  # noqa: B010
    monkeypatch.setitem(sys.modules, "universr", fake_pkg)
    monkeypatch.setitem(sys.modules, "universr.inference", fake_inference)
    _FakeOfficialUniverSR.from_pretrained_calls.clear()


def _make_fake_download(
    files: dict[str, Path], calls: list[tuple[str, str]]
) -> Callable[[str, str], str]:
    def fake_download(repo_id: str, filename: str) -> str:
        calls.append((repo_id, filename))
        return str(files[filename])

    return fake_download


def _write_artifacts(tmp_path: Path) -> dict[str, Path]:
    weights = tmp_path / "pytorch_model.bin"
    weights.write_bytes(b"fake universr weights payload")
    config = tmp_path / "config.yaml"
    config.write_bytes(b"fake: config\n")
    return {"pytorch_model.bin": weights, "config.yaml": config}


# ---------------------------------------------------------------------------
# DSP restore() contracts
# ---------------------------------------------------------------------------


def test_dsp_restore_generates_default_strength_ladder() -> None:
    """The DSP baseline must return one candidate per ladder strength with intact shapes."""
    lp = _lowpassed_signal()
    cands = _dsp_baseline().restore(lp, sample_rate=SR, effective_cutoff_hz=4000.0)

    assert [c.strength for c in cands] == [1.0, 0.75, 0.5, 0.25, 0.0]
    for c in cands:
        assert c.audio.shape == lp.shape
        assert c.audio.dtype == np.float32
        assert c.cutoff_hz == 4000.0
        assert np.all(np.isfinite(c.audio))
        assert float(np.max(np.abs(c.audio))) <= 1.0, "candidates must be clipped to [-1, 1]"


def test_dsp_restore_strength_zero_is_bit_identical_to_input() -> None:
    """Strength 0.0 is the Natural-safe fallback: it must be the untouched input."""
    lp = _lowpassed_signal()
    cands = _dsp_baseline().restore(lp, sample_rate=SR, effective_cutoff_hz=4000.0)

    zero = next(c for c in cands if c.strength == 0.0)
    np.testing.assert_array_equal(zero.audio, lp)


def test_dsp_restore_adds_high_band_energy_monotonically_with_strength() -> None:
    """Higher ladder strengths must inject strictly more energy above the cutoff."""
    lp = _lowpassed_signal()
    cands = _dsp_baseline().restore(lp, sample_rate=SR, effective_cutoff_hz=4000.0)

    powers = {c.strength: _high_band_power(c.audio) for c in cands}
    assert powers[1.0] > 1e-5, "full strength must synthesize measurable high-band content"
    assert powers[1.0] > powers[0.75] > powers[0.5] > powers[0.25] > powers[0.0]
    assert powers[0.0] < 1e-9, "the lowpassed input has no high band of its own"


def test_dsp_restore_preserves_protected_band_at_guard_tolerance() -> None:
    """The strongest candidate must keep the observed band within Guard R's production gate."""
    lp = _lowpassed_signal()
    cands = _dsp_baseline().restore(lp, sample_rate=SR, effective_cutoff_hz=4000.0)
    full = next(c for c in cands if c.strength == 1.0)

    guard_cfg = RestorationGuardConfig()
    check = verify_protected_band_invariance(
        lp,
        full.audio,
        sample_rate=SR,
        cutoff_hz=4000.0,
        tolerance_rms=guard_cfg.protected_band_threshold,
        tolerance_stft=guard_cfg.protected_band_threshold * 2.0,
    )
    assert check.passes_invariance
    assert check.rms_waveform_error <= guard_cfg.protected_band_threshold

    # The untouched strength-0.0 candidate must pass even the strict default tolerances.
    zero = next(c for c in cands if c.strength == 0.0)
    strict = verify_protected_band_invariance(lp, zero.audio, sample_rate=SR, cutoff_hz=4000.0)
    assert strict.passes_invariance
    assert strict.rms_waveform_error == 0.0


def test_dsp_restore_multichannel_processes_channels_independently() -> None:
    """Stereo input must come back stereo, each channel restored as it would be alone."""
    lp = _lowpassed_signal()
    stereo = np.stack([lp, (lp * 0.5).astype(np.float32)], axis=0)

    restorer = _dsp_baseline()
    stereo_cands = restorer.restore(
        stereo, sample_rate=SR, effective_cutoff_hz=4000.0, strengths=[1.0, 0.0]
    )
    mono_cands = restorer.restore(
        lp, sample_rate=SR, effective_cutoff_hz=4000.0, strengths=[1.0, 0.0]
    )

    for c in stereo_cands:
        assert c.audio.shape == stereo.shape

    stereo_full = next(c for c in stereo_cands if c.strength == 1.0)
    mono_full = next(c for c in mono_cands if c.strength == 1.0)
    np.testing.assert_array_equal(stereo_full.audio[0], mono_full.audio)

    stereo_zero = next(c for c in stereo_cands if c.strength == 0.0)
    np.testing.assert_array_equal(stereo_zero.audio, stereo)


def test_dsp_restore_input_shorter_than_window_passes_through_unchanged() -> None:
    """Input shorter than one analysis window cannot be restored; every candidate is the input."""
    short = _lowpassed_signal()[:1000]
    restorer = _dsp_baseline()
    assert len(short) < restorer.win_length

    cands = restorer.restore(short, sample_rate=SR, effective_cutoff_hz=4000.0)
    assert [c.strength for c in cands] == [1.0, 0.75, 0.5, 0.25, 0.0]
    for c in cands:
        np.testing.assert_array_equal(c.audio, short)


def test_dsp_restore_cutoff_below_analysis_floor_invents_nothing() -> None:
    """With no usable low band to extrapolate from, the restorer must not invent content."""
    lp = _lowpassed_signal()
    assert _high_band_power(lp, floor_hz=1000.0) > 1e-7, "input carries mid-band energy"

    # A 150 Hz cutoff sits below the extrapolation source floor (bin 10 at ~234 Hz).
    cands = _dsp_baseline().restore(
        lp, sample_rate=SR, effective_cutoff_hz=150.0, strengths=[1.0, 0.0]
    )

    full = next(c for c in cands if c.strength == 1.0)
    assert np.all(np.isfinite(full.audio))
    assert _high_band_power(full.audio, floor_hz=1000.0) < 1e-10, (
        "an empty generated spectrum must leave the untrusted band silent, not hallucinated"
    )

    zero = next(c for c in cands if c.strength == 0.0)
    np.testing.assert_array_equal(zero.audio, lp)


def test_device_selection_priority_and_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto-detection prefers cuda over mps over cpu; an explicit device always wins."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert UniverSRBaseline(sample_rate=SR, use_neural=False).device == "cpu"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert UniverSRBaseline(sample_rate=SR, use_neural=False).device == "mps"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert UniverSRBaseline(sample_rate=SR, use_neural=False).device == "cuda"

    assert UniverSRBaseline(sample_rate=SR, use_neural=False, device="cpu").device == "cpu"


# ---------------------------------------------------------------------------
# Neural path contracts (fake model, no network, no vendored weights)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cutoff_hz", "expected_sr_khz"),
    [
        (4000.0, 8),
        (5000.0, 8),
        (6000.0, 12),
        (7000.0, 12),
        (9000.0, 16),
        (10000.0, 16),
        (12000.0, 24),
    ],
)
def test_cutoff_maps_to_official_universr_input_rate(
    cutoff_hz: float, expected_sr_khz: int
) -> None:
    """Effective cutoff must map onto the official UniverSR input-rate grid."""
    fake = _FakeNeuralUniverSR()
    restorer = _dsp_baseline()
    restorer._neural_model = fake

    restorer.restore(
        _lowpassed_signal(), sample_rate=SR, effective_cutoff_hz=cutoff_hz, strengths=[1.0]
    )
    assert fake.sr_khz_calls == [expected_sr_khz]


def test_neural_input_padded_to_min_samples_and_output_trimmed() -> None:
    """Short inputs are padded to the model's 32768-sample floor, outputs trimmed back."""
    fake = _FakeNeuralUniverSR()
    restorer = _dsp_baseline()
    restorer._neural_model = fake

    lp = _lowpassed_signal()
    assert len(lp) < 32_768
    cands = restorer.restore(lp, sample_rate=SR, effective_cutoff_hz=4000.0, strengths=[1.0])

    assert fake.input_lengths == [32_768]
    assert cands[0].audio.shape == lp.shape


def test_neural_inference_uses_pinned_deterministic_ode_settings() -> None:
    """The official model must be driven with the pinned midpoint/4-step/no-guidance config."""
    fake = _FakeNeuralUniverSR()
    restorer = _dsp_baseline()
    restorer._neural_model = fake

    restorer.restore(
        _lowpassed_signal(), sample_rate=SR, effective_cutoff_hz=4000.0, strengths=[1.0]
    )
    assert fake.ode_calls == [("midpoint", 4, 0.0)]


def test_neural_output_shape_mismatch_falls_back_to_dsp_extrapolation() -> None:
    """A neural output of the wrong length must be discarded, not merged."""
    lp = _lowpassed_signal()
    dsp_full = _dsp_baseline().restore(
        lp, sample_rate=SR, effective_cutoff_hz=4000.0, strengths=[1.0]
    )[0]

    # Longer than one analysis window, but shorter than the input: STFT frame counts differ.
    fake = _FakeNeuralUniverSR(trim_to=8000)
    restorer = _dsp_baseline()
    restorer._neural_model = fake
    cand = restorer.restore(lp, sample_rate=SR, effective_cutoff_hz=4000.0, strengths=[1.0])[0]

    assert fake.input_lengths, "the neural model must have been attempted first"
    np.testing.assert_array_equal(cand.audio, dsp_full.audio)


def test_neural_inference_exception_falls_back_to_dsp_extrapolation() -> None:
    """A crashing neural model must degrade to the deterministic DSP path, not fail restore()."""
    lp = _lowpassed_signal()
    dsp_full = _dsp_baseline().restore(
        lp, sample_rate=SR, effective_cutoff_hz=4000.0, strengths=[1.0]
    )[0]

    restorer = _dsp_baseline()
    restorer._neural_model = _FakeNeuralUniverSR(fail=True)
    cand = restorer.restore(lp, sample_rate=SR, effective_cutoff_hz=4000.0, strengths=[1.0])[0]

    np.testing.assert_array_equal(cand.audio, dsp_full.audio)


# ---------------------------------------------------------------------------
# Upstream artifact provenance
# ---------------------------------------------------------------------------


def test_verify_upstream_artifacts_accepts_matching_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both attested artifacts are downloaded and accepted when their hashes match."""
    files = _write_artifacts(tmp_path)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _make_fake_download(files, calls))

    repo_info = {
        "repo_id": "fake/universr-speech",
        "sha256": hash_file(files["pytorch_model.bin"]),
        "config_sha256": hash_file(files["config.yaml"]),
    }
    _dsp_baseline()._verify_upstream_artifacts(repo_info)

    assert calls == [
        ("fake/universr-speech", "pytorch_model.bin"),
        ("fake/universr-speech", "config.yaml"),
    ]


def test_verify_upstream_artifacts_rejects_weights_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint whose hash differs from the attestation must raise, naming the file."""
    files = _write_artifacts(tmp_path)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _make_fake_download(files, calls))

    repo_info = {
        "repo_id": "fake/universr-speech",
        "sha256": "0" * 64,
        "config_sha256": hash_file(files["config.yaml"]),
    }
    with pytest.raises(ModelProvenanceError, match="pytorch_model.bin") as excinfo:
        _dsp_baseline()._verify_upstream_artifacts(repo_info)

    message = str(excinfo.value)
    assert "fake/universr-speech" in message
    assert "0" * 64 in message, "the error must state the attested hash"
    assert hash_file(files["pytorch_model.bin"]) in message, "and the hash actually observed"


def test_verify_upstream_artifacts_rejects_config_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The config artifact is verified too, not just the weights."""
    files = _write_artifacts(tmp_path)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _make_fake_download(files, calls))

    repo_info = {
        "repo_id": "fake/universr-speech",
        "sha256": hash_file(files["pytorch_model.bin"]),
        "config_sha256": "f" * 64,
    }
    with pytest.raises(ModelProvenanceError, match="config.yaml"):
        _dsp_baseline()._verify_upstream_artifacts(repo_info)


def test_verify_upstream_artifacts_skips_files_without_attested_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no hash attested there is nothing to verify: no download may be attempted."""

    def refuse_download(repo_id: str, filename: str) -> str:
        raise AssertionError(f"unexpected download of {repo_id}/{filename}")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", refuse_download)
    _dsp_baseline()._verify_upstream_artifacts({"repo_id": "fake/universr-speech"})


# ---------------------------------------------------------------------------
# _init_neural_model contracts
# ---------------------------------------------------------------------------


def test_init_neural_import_failure_falls_back_to_dsp_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A missing vendored package must degrade to DSP extrapolation, loudly."""
    monkeypatch.delitem(sys.modules, "universr.inference", raising=False)
    monkeypatch.setitem(sys.modules, "universr", ModuleType("universr"))

    with caplog.at_level(logging.WARNING, logger="hawavoclean.restoration.universr"):
        restorer = UniverSRBaseline(sample_rate=SR, use_neural=True, device="cpu")

    assert restorer._neural_model is None
    assert "Could not load official UniverSR neural weights" in caplog.text

    # The fallback restorer must still be able to restore.
    lp = _lowpassed_signal()
    cands = restorer.restore(lp, sample_rate=SR, effective_cutoff_hz=4000.0, strengths=[1.0, 0.0])
    assert len(cands) == 2
    assert _high_band_power(cands[0].audio) > 1e-5


def test_init_neural_provenance_error_propagates_instead_of_falling_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hash mismatch is a safety failure: it must abort init, never silently degrade to DSP."""
    _install_fake_vendor_package(monkeypatch)

    junk = tmp_path / "pytorch_model.bin"
    junk.write_bytes(b"re-uploaded upstream weights that nobody attested")

    def fake_download(repo_id: str, filename: str) -> str:  # noqa: ARG001
        return str(junk)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    with pytest.raises(ModelProvenanceError):
        UniverSRBaseline(sample_rate=SR, use_neural=True, device="cpu")

    assert _FakeOfficialUniverSR.from_pretrained_calls == [], (
        "the model must not be loaded from unverified artifacts"
    )


def test_init_neural_loads_model_after_successful_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When artifacts verify against the registry, the official model is loaded on the device."""
    _install_fake_vendor_package(monkeypatch)
    files = _write_artifacts(tmp_path)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _make_fake_download(files, calls))
    monkeypatch.setitem(
        universr_upstream.UNIVERSR_WEIGHTS_REGISTRY,
        "universr-speech",
        {
            "repo_id": "fake/universr-speech",
            "sha256": hash_file(files["pytorch_model.bin"]),
            "config_sha256": hash_file(files["config.yaml"]),
            "license": "CC-BY-4.0",
        },
    )

    restorer = UniverSRBaseline(sample_rate=SR, use_neural=True, device="cpu")

    assert isinstance(restorer._neural_model, _FakeOfficialUniverSR)
    assert _FakeOfficialUniverSR.from_pretrained_calls == [("fake/universr-speech", "cpu")]


# ---------------------------------------------------------------------------
# STFT / ISTFT roundtrip
# ---------------------------------------------------------------------------


def test_stft_istft_roundtrip_reconstructs_signal() -> None:
    """istft(stft(x)) must reconstruct x over its full length within float32 precision."""
    restorer = _dsp_baseline()
    lp = _lowpassed_signal()  # 14400 samples: an exact multiple of the 480-sample hop

    Zxx = restorer._stft(lp)
    assert Zxx.dtype == np.complex64
    assert Zxx.shape[0] == restorer.n_fft // 2 + 1

    rt = restorer._istft(Zxx)
    assert rt.dtype == np.float32
    assert len(rt) == len(lp)
    np.testing.assert_allclose(rt, lp, atol=1e-5)


def test_stft_istft_roundtrip_covers_non_hop_aligned_lengths() -> None:
    """Lengths that do not divide the hop must round-trip without losing samples."""
    restorer = _dsp_baseline()
    lp = np.concatenate([_lowpassed_signal(), _lowpassed_signal()[:100]])
    assert len(lp) % restorer.hop_length != 0

    rt = restorer._istft(restorer._stft(lp))
    assert len(rt) >= len(lp), "reconstruction must never drop trailing samples"
    np.testing.assert_allclose(rt[: len(lp)], lp, atol=1e-5)
