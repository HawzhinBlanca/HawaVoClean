"""The ``[runtime]`` and ``[input]`` keys that used to be declared and never read.

Four keys — ``device``, ``num_threads``, ``worker_memory_limit_mb`` and
``input.supported_sample_rates`` — had zero uses outside ``config.py``. These
tests are the receipts that each one now does something, and that doing it did
not move a single published sample.
"""

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from hawavoclean import runtime
from hawavoclean.audio.probe import MIN_SUPPORTED_SAMPLE_RATE, probe_audio
from hawavoclean.config import HawaVoCleanConfig, InputConfig, RuntimeConfig, load_config
from hawavoclean.enhancement.factory import CORE_REGISTRY
from hawavoclean.enhancement.production import WienerSpectralEnhancer
from hawavoclean.enhancement.studio import STUDIO_PARAMS
from hawavoclean.errors import ConfigError, InvalidUserInputError, WorkerOOMError
from hawavoclean.paths import profile_config_path
from hawavoclean.report.schema import EnvironmentMetadata

PROD = "wiener-dd-48k-v1"
STUDIO = "studio-dfn3-48k-v1"


# ---- device: resolution ---------------------------------------------------


def test_auto_resolves_to_cpu_so_the_default_stays_reproducible() -> None:
    res = runtime.resolve_device("auto")
    assert res.resolved == "cpu"
    assert res.requested == "auto"
    assert "opt-in" in res.reason


def test_auto_walks_the_preference_ladder_and_takes_the_first_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "device_available", lambda n: n in ("mps", "cpu"))
    assert runtime.resolve_device("auto", preference=("cuda", "mps", "cpu")).resolved == "mps"
    assert runtime.resolve_device("auto", preference=("cuda", "cpu")).resolved == "cpu"


def test_auto_falls_back_to_cpu_when_nothing_in_the_ladder_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "device_available", lambda _n: False)
    assert runtime.resolve_device("auto", preference=("cuda",)).resolved == "cpu"


def test_explicit_unavailable_device_is_a_designed_error_not_a_silent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "device_available", lambda n: n == "cpu")
    with pytest.raises(ConfigError) as exc:
        runtime.resolve_device("cuda")
    # The whole point: a run must never be quietly attributed to the CPU
    # after being told to use a GPU.
    assert "not available" in str(exc.value)
    assert "fall back" in str(exc.value)


def test_explicit_available_device_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "device_available", lambda _n: True)
    assert runtime.resolve_device("mps", core_id=STUDIO).resolved == "mps"


def test_unknown_device_name_is_rejected() -> None:
    with pytest.raises(ConfigError):
        runtime.resolve_device("tpu")


def test_cpu_is_always_available_without_importing_torch() -> None:
    assert runtime.device_available("cpu") is True
    assert runtime.device_available("nonsense") is False


# ---- device: the report can never name the wrong compute path -------------


def test_cpu_dsp_core_reports_cpu_even_when_a_gpu_was_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "device_available", lambda _n: True)
    res = runtime.resolve_device("mps", core_id=PROD)
    assert res.resolved == "cpu"
    assert "CPU DSP" in res.reason


def test_unknown_core_does_not_invent_a_downgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "device_available", lambda _n: True)
    assert runtime.resolve_device("mps", core_id="not-a-core").resolved == "mps"


def test_registry_marks_exactly_the_neural_core_as_device_aware() -> None:
    assert CORE_REGISTRY[STUDIO].device_aware is True
    assert CORE_REGISTRY[PROD].device_aware is False


def test_active_device_defaults_to_cpu_when_no_run_has_been_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(runtime.DEVICE_ENV_VAR, raising=False)
    assert runtime.active_device() == "cpu"
    monkeypatch.setenv(runtime.DEVICE_ENV_VAR, "garbage")
    assert runtime.active_device() == "cpu"
    monkeypatch.setenv(runtime.DEVICE_ENV_VAR, "mps")
    assert runtime.active_device() == "mps"


def test_report_environment_records_the_device_that_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(runtime.DEVICE_ENV_VAR, "mps")
    env = EnvironmentMetadata(
        platform="p",
        os_version="o",
        python_version="3",
        numpy_version="n",
        scipy_version="s",
        soundfile_version="sf",
    )
    assert env.compute_device == "mps"


def test_loading_a_profile_arms_the_device_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runtime.DEVICE_ENV_VAR, raising=False)
    cfg = load_config(profile_config_path("production"), is_production=True)
    assert cfg.runtime.device == "auto"
    assert os.environ[runtime.DEVICE_ENV_VAR] == "cpu"
    assert runtime.active_device() == "cpu"


def test_a_profile_asking_for_an_absent_device_fails_before_any_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "device_available", lambda n: n == "cpu")
    cfg = tmp_path / "gpu.toml"
    cfg.write_text('[runtime]\ndevice = "cuda"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="cuda"):
        load_config(cfg, is_production=True)


def test_mps_is_reachable_but_never_the_default() -> None:
    """``mps`` must be a nameable choice, and must not be what ``auto`` picks."""
    assert "mps" in RuntimeConfig.model_fields["device"].annotation.__args__  # type: ignore[union-attr]
    assert "mps" not in runtime.AUTO_DEVICE_PREFERENCE
    assert runtime.AUTO_DEVICE_PREFERENCE == ("cpu",)


def test_device_is_not_part_of_core_identity() -> None:
    """The lockfile pins what the model IS; the device belongs to the run."""
    assert "device" not in STUDIO_PARAMS
    assert not any("device" in str(k).lower() for k in STUDIO_PARAMS)


# ---- num_threads: a real worker-pool budget -------------------------------


def test_pool_size_is_num_threads_clamped_to_the_machine() -> None:
    assert runtime.worker_pool_size(1, cpu_count=14) == 1
    assert runtime.worker_pool_size(4, cpu_count=14) == 4
    assert runtime.worker_pool_size(64, cpu_count=8) == 8
    assert runtime.worker_pool_size(0, cpu_count=8) == 1


def test_threads_per_worker_divides_the_machine_between_pooled_workers() -> None:
    assert runtime.threads_per_worker(1, cpu_count=14) == 14
    assert runtime.threads_per_worker(2, cpu_count=14) == 7
    assert runtime.threads_per_worker(14, cpu_count=14) == 1
    assert runtime.threads_per_worker(64, cpu_count=3) == 1


def test_a_single_worker_leaves_the_numeric_stack_exactly_as_it_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in runtime.THREAD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    assert runtime.apply_thread_budget(1, cpu_count=14) is None
    assert all(v not in os.environ for v in runtime.THREAD_ENV_VARS)


def test_a_pool_publishes_a_per_worker_thread_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in runtime.THREAD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    assert runtime.apply_thread_budget(2, cpu_count=14) == 7
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert os.environ["MKL_NUM_THREADS"] == "7"


def test_an_operators_own_thread_setting_outranks_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    runtime.apply_thread_budget(2, cpu_count=14)
    assert os.environ["OMP_NUM_THREADS"] == "3"


def test_runtime_config_exposes_the_pool_numbers() -> None:
    cfg = RuntimeConfig(num_threads=2)
    assert cfg.pool_size() == runtime.worker_pool_size(2)
    assert cfg.threads_per_worker() == runtime.threads_per_worker(2)


# ---- worker_memory_limit_mb: self-policing, fail-closed -------------------


def test_no_budget_published_means_no_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runtime.MEMORY_LIMIT_ENV_VAR, raising=False)
    assert runtime.memory_budget_mb() is None
    runtime.check_memory_budget()  # must not raise


def test_a_malformed_or_zero_budget_is_treated_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(runtime.MEMORY_LIMIT_ENV_VAR, "not-a-number")
    assert runtime.memory_budget_mb() is None
    monkeypatch.setenv(runtime.MEMORY_LIMIT_ENV_VAR, "0")
    assert runtime.memory_budget_mb() is None


def test_peak_rss_is_a_plausible_positive_number() -> None:
    peak = runtime.process_peak_rss_bytes()
    assert peak > 8 * 1024 * 1024  # a live CPython is never under 8 MB


def test_peak_rss_uses_native_windows_counter_without_importing_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hawavoclean.runtime.sys.platform", "win32")
    monkeypatch.setattr(runtime, "_windows_peak_rss_bytes", lambda: 123_456_789)
    assert runtime.process_peak_rss_bytes() == 123_456_789


def test_an_overrun_worker_refuses_further_units(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(runtime.MEMORY_LIMIT_ENV_VAR, "1")
    with pytest.raises(WorkerOOMError, match="worker_memory_limit_mb"):
        runtime.check_memory_budget("studio")


def test_a_core_over_budget_raises_rather_than_returning_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The enhancement path must fail, so the unit falls back to ORIGINAL."""
    monkeypatch.setenv(runtime.MEMORY_LIMIT_ENV_VAR, "1")
    core = WienerSpectralEnhancer()
    with pytest.raises(WorkerOOMError):
        core.enhance(np.zeros(4800, dtype=np.float32), 48000)


def test_the_real_default_budget_has_room_for_a_real_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """8192 MB must not fire on ordinary work — measured peak is ~1.3 GB."""
    monkeypatch.setenv(runtime.MEMORY_LIMIT_ENV_VAR, "8192")
    core = WienerSpectralEnhancer()
    core.enhance(np.zeros(48000 * 5, dtype=np.float32), 48000)  # must not raise


def test_loading_a_profile_publishes_its_memory_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(runtime.MEMORY_LIMIT_ENV_VAR, raising=False)
    cfg = load_config(profile_config_path("production"), is_production=True)
    assert runtime.memory_budget_mb() == cfg.runtime.worker_memory_limit_mb == 8192


# ---- input.supported_sample_rates: an enforced envelope -------------------


def _write_wav(path: Path, sample_rate: int, seconds: float = 0.25) -> Path:
    data = np.zeros(int(sample_rate * seconds), dtype=np.float32)
    data[::17] = 0.1
    sf.write(str(path), data, samplerate=sample_rate, subtype="PCM_16", format="WAV")
    return path


def test_the_probe_floor_comes_from_the_configuration_not_a_restated_constant() -> None:
    assert min(InputConfig().supported_sample_rates) == MIN_SUPPORTED_SAMPLE_RATE


def test_a_rate_below_the_envelope_is_refused(tmp_path: Path) -> None:
    low = _write_wav(tmp_path / "low.wav", 4000)
    with pytest.raises(InvalidUserInputError, match="below the minimum supported"):
        probe_audio(low)


def test_a_rate_above_the_envelope_is_refused(tmp_path: Path) -> None:
    high = _write_wav(tmp_path / "high.wav", 96000)
    with pytest.raises(InvalidUserInputError, match="exceeds maximum supported"):
        probe_audio(high, max_sample_rate=48000)


def test_a_configured_envelope_overrides_the_declared_default(tmp_path: Path) -> None:
    low = _write_wav(tmp_path / "low.wav", 4000)
    probe = probe_audio(low, supported_sample_rates=[4000, 48000])
    assert probe.sample_rate == 4000


def test_a_rate_between_listed_values_is_accepted_because_it_is_an_envelope(
    tmp_path: Path,
) -> None:
    """11.025 kHz is not in the list and is processed correctly; refusing it
    would be a capability regression dressed up as rigour."""
    odd = _write_wav(tmp_path / "odd.wav", 11025)
    assert probe_audio(odd).sample_rate == 11025


def test_an_empty_envelope_is_rejected_by_the_schema() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        InputConfig(supported_sample_rates=[])


def test_a_negative_rate_is_rejected_by_the_schema() -> None:
    with pytest.raises(ValueError, match="positive"):
        InputConfig(supported_sample_rates=[-1, 48000])


def test_an_envelope_that_cannot_accept_anything_is_rejected() -> None:
    with pytest.raises(ValueError, match="could ever be accepted"):
        InputConfig(supported_sample_rates=[44100, 48000], max_sample_rate=16000)


def test_the_envelope_is_normalised_and_min_is_derived() -> None:
    cfg = InputConfig(supported_sample_rates=[48000, 8000, 8000, 16000])
    assert cfg.supported_sample_rates == [8000, 16000, 48000]
    assert cfg.min_sample_rate == 8000


# ---- the non-negotiable ---------------------------------------------------


def test_making_these_keys_real_did_not_move_the_config_hash() -> None:
    """The config hash feeds the job id, which seeds the master's dither, so a
    schema change here would rewrite the last bits of every published sample.
    This is the committed hash of the production profile."""
    cfg = load_config(profile_config_path("production"), is_production=True)
    assert cfg.compute_hash() == (
        "0935c0be95aa2ea0955e37ec80dffd6d604272994ae1e86c95004431c8437835"
    )


def test_every_runtime_and_input_key_is_read_by_something() -> None:
    """A key nothing reads is a promise the tool does not keep. Guard the shape
    of the two sections so a new dead key cannot be added silently."""
    assert set(RuntimeConfig.model_fields) == {
        "device",
        "isolated_worker",
        "worker_timeout_s",
        "worker_memory_limit_mb",
        "development",
        "num_threads",
    }
    assert set(InputConfig.model_fields) == {
        "max_sample_rate",
        "supported_sample_rates",
        "channel_mode",
        "output_bit_depth",
    }


def test_defaults_construct_and_hash_without_a_toml_file() -> None:
    cfg: Any = HawaVoCleanConfig()
    assert cfg.runtime.device == "auto"
    assert len(cfg.compute_hash()) == 64


def test_a_second_profile_revises_our_own_budget_but_not_an_operators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long-lived process (the UI server) loads more than one profile. Ours
    must not go stale, and theirs must not be trampled."""
    monkeypatch.setattr(runtime, "_published_thread_budget", None)
    for var in runtime.THREAD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    runtime.apply_thread_budget(2, cpu_count=12)
    assert os.environ["OMP_NUM_THREADS"] == "6"
    runtime.apply_thread_budget(4, cpu_count=12)
    assert os.environ["OMP_NUM_THREADS"] == "3"

    monkeypatch.setenv("OMP_NUM_THREADS", "9")
    runtime.apply_thread_budget(2, cpu_count=12)
    assert os.environ["OMP_NUM_THREADS"] == "9"


def test_a_base_install_without_torch_reports_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default path of a non-studio install must not need torch at all."""
    monkeypatch.setitem(sys.modules, "torch", None)
    assert runtime.device_available("cuda") is False
    assert runtime.device_available("mps") is False
    assert runtime.device_available("cpu") is True
    assert runtime.resolve_device("auto").resolved == "cpu"


def test_cuda_availability_is_probed_through_torch() -> None:
    """Exercises the cuda branch on whatever this machine actually has."""
    import importlib.util

    expected = False
    if importlib.util.find_spec("torch") is not None:
        import torch

        expected = bool(torch.cuda.is_available())
    assert runtime.device_available("cuda") is expected


def test_activating_without_a_memory_limit_leaves_the_channel_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(runtime.MEMORY_LIMIT_ENV_VAR, raising=False)
    runtime.activate_runtime("auto", core_id=PROD, num_threads=1)
    assert runtime.memory_budget_mb() is None
    assert runtime.active_device() == "cpu"
