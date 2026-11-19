"""Separation boundary tests; none substitutes fake audio for quality evidence."""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import hawavoclean.separation.roformer as roformer
import hawavoclean.separation.runtime as runtime
from hawavoclean.process_supervisor import ProcessSupervisor
from hawavoclean.separation import installation


@pytest.fixture(autouse=True)
def isolated_installation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "_LOCAL_ROOT", tmp_path / "local-ai")
    monkeypatch.setattr(roformer, "_LOCAL_ROOT", tmp_path / "local-ai")
    monkeypatch.delenv(runtime.PYTHON_ENV, raising=False)
    monkeypatch.delenv(runtime.ROOT_ENV, raising=False)
    monkeypatch.delenv(roformer.ROFORMER_PYTHON_ENV, raising=False)
    monkeypatch.setenv(roformer.ROFORMER_MODEL_DIR_ENV, str(tmp_path / "no-roformer-models"))


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "source.wav"
    sf.write(path, np.full((1000, 2), 0.1, dtype=np.float32), 44100, subtype="FLOAT")
    return path


def test_missing_runtime_is_blocked_without_importing_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(runtime.PYTHON_ENV, raising=False)
    monkeypatch.delenv(runtime.ROOT_ENV, raising=False)
    status = runtime.probe_runtime()
    assert status.available is False
    assert "not installed" in status.reason
    assert status.providers == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"prompt": ""},
        {"prompt": "x\nrun"},
        {"prompt": "x" * 513},
        {"operation": "erase-everything"},
        {"model_size": "unknown"},
        {"quality": "best-ever"},
        {"seed": True},
        {"seed": -1},
        {"device": "cloud"},
    ],
)
def test_invalid_settings_never_reach_worker(
    source: Path, tmp_path: Path, changes: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("worker must not start")

    monkeypatch.setattr(runtime, "_run_worker", forbidden)
    options = {"prompt": "music", "operation": "remove", **changes}
    with pytest.raises(runtime.SeparationError):
        runtime.separate_audio(source, tmp_path / "output.wav", **options)


def test_existing_destination_is_untouched(source: Path, tmp_path: Path) -> None:
    output = tmp_path / "output.wav"
    output.write_bytes(b"existing file")
    with pytest.raises(runtime.SeparationError, match="never replaces"):
        runtime.separate_audio(source, output, prompt="music", operation="remove")
    assert output.read_bytes() == b"existing file"


def test_multichannel_source_is_rejected_before_worker(tmp_path: Path) -> None:
    source = tmp_path / "three.wav"
    sf.write(source, np.zeros((100, 3), dtype=np.float32), 48000)
    with pytest.raises(runtime.SeparationError, match="mono/stereo"):
        runtime.separate_audio(source, tmp_path / "out.wav", prompt="music", operation="remove")


def test_long_audio_is_rejected_from_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert runtime.MAX_DURATION_S == 3600.0
    source = tmp_path / "long.wav"
    with sf.SoundFile(source, mode="w", samplerate=8000, channels=1, subtype="FLOAT") as file:
        file.write(np.zeros(8000 * 121, dtype=np.float32))
    # 121 s is now allowed under the lifted 60-minute limit, while exceeding
    # the per-pass worker ceiling (120 s) is still rejected cleanly.
    with runtime.validate_audio_file(source) as ok_audio:
        assert ok_audio.frames == 8000 * 121
    with pytest.raises(runtime.SeparationError, match="2 minutes"):
        runtime.validate_audio_file(source, max_duration_s=runtime.WORKER_MAX_DURATION_S)
    monkeypatch.setattr(runtime, "MAX_DURATION_S", 120.0)
    with pytest.raises(runtime.SeparationError, match="2 minutes"):
        runtime.validate_audio_file(source)


@pytest.mark.parametrize("bad", ["wrong_rate", "wrong_channels", "wrong_length", "nan", "extreme"])
def test_worker_output_must_pass_independent_validation(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    def worker(request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        audio, rate = sf.read(source, dtype="float32", always_2d=True)
        if bad == "wrong_rate":
            rate = 48000
        elif bad == "wrong_channels":
            audio = audio[:, :1]
        elif bad == "wrong_length":
            audio = audio[:-1]
        elif bad == "nan":
            audio[0, 0] = np.nan
        else:
            audio[0, 0] = 100
        sf.write(request["output"], audio, rate, subtype="FLOAT")
        return {"selected_stem": "residual"}

    monkeypatch.setattr(runtime, "_run_worker", worker)
    output = tmp_path / "output.wav"
    with pytest.raises(runtime.SeparationError):
        runtime.separate_audio(source, output, prompt="music", operation="remove")
    assert not output.exists()
    assert not list(tmp_path.glob(".hawavoclean-separation-*"))


def test_quiet_residual_is_valid_and_keeps_exact_timeline(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def worker(request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        assert request["operation"] == "remove"
        sf.write(request["output"], np.zeros((1000, 2), dtype=np.float32), 44100, subtype="FLOAT")
        return {"selected_stem": "residual", "model_sha256": "a" * 64}

    monkeypatch.setattr(runtime, "_run_worker", worker)
    output = tmp_path / "output.wav"
    original = source.read_bytes()
    result = runtime.separate_audio(source, output, prompt="music", operation="remove")
    assert source.read_bytes() == original
    assert (sf.info(output).frames, sf.info(output).samplerate, sf.info(output).channels) == (
        1000,
        44100,
        2,
    )
    assert result["selected_stem"] == "residual"
    assert result["qualification"] == "experimental_unqualified"


@pytest.fixture
def subprocess_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    script = tmp_path / "fake_worker.py"
    script.write_text("""import json, sys, time
r=json.loads(sys.stdin.readline())
if r.get("hang"):
    time.sleep(60)
if r.get("bad_json"):
    print("HAWAVOCLEAN_SEPARATION_V1={bad", flush=True)
else:
    print("HAWAVOCLEAN_SEPARATION_V1="+json.dumps({"kind":"progress","fraction":0.5,"message":"Working"}), flush=True)
    print("HAWAVOCLEAN_SEPARATION_V1="+json.dumps({"kind":"result","ok":True,"metadata":{"ready":True}}), flush=True)
""")
    monkeypatch.setattr(runtime, "_runtime_paths", lambda: (Path(sys.executable), tmp_path))
    original = ProcessSupervisor.spawn
    calls: list[list[str]] = []

    def spawn(command: list[str], **options: Any) -> Any:
        calls.append(command)
        assert options["env"]["HF_HUB_OFFLINE"] == "1"
        assert options["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert "HF_TOKEN" not in options["env"]
        return original([*command[:-1], str(script)], **options)

    monkeypatch.setattr(ProcessSupervisor, "spawn", spawn)
    return calls


def test_subprocess_protocol_isolated_and_prompt_not_in_arguments(
    subprocess_worker: list[list[str]],
) -> None:
    updates = []
    result = runtime._run_worker(
        {"prompt": "private sound description"},
        timeout_s=3,
        progress=lambda fraction, message: updates.append((fraction, message)),
    )
    assert result == {"ready": True}
    assert updates == [(0.5, "Working")]
    assert subprocess_worker[0][1:3] == ["-I", "-B"]
    assert "private sound description" not in json.dumps(subprocess_worker)


def test_worker_timeout_terminates_owned_process(subprocess_worker: list[list[str]]) -> None:
    started = time.monotonic()
    with pytest.raises(runtime.SeparationError, match="deadline"):
        runtime._run_worker({"hang": True}, timeout_s=0.15)
    assert len(subprocess_worker) == 1
    assert time.monotonic() - started < 3


def test_worker_cancellation_terminates_owned_process(subprocess_worker: list[list[str]]) -> None:
    started = time.monotonic()
    with pytest.raises(runtime.SeparationCancelledError):
        runtime._run_worker(
            {"hang": True}, timeout_s=3, should_cancel=lambda: time.monotonic() - started > 0.15
        )
    assert len(subprocess_worker) == 1
    assert time.monotonic() - started < 3


def test_invalid_worker_protocol_fails_closed(subprocess_worker: list[list[str]]) -> None:
    with pytest.raises(runtime.SeparationError, match="invalid or excessive"):
        runtime._run_worker({"bad_json": True}, timeout_s=3)
    assert len(subprocess_worker) == 1


def test_access_check_without_credential_does_not_contact_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("No authenticated access check without credentials")

    hub = types.SimpleNamespace(get_token=lambda: None, get_hf_file_metadata=forbidden)
    monkeypatch.setattr(importlib, "import_module", lambda _name: hub)
    assert installation.model_access_status("small") == "credential_not_configured"


def test_access_check_does_not_expose_authentication_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("sensitive fake credential should never reach a diagnostic")

    hub = types.SimpleNamespace(
        get_token=lambda: "private",
        hf_hub_url=lambda *_args: "https://example.invalid",
        get_hf_file_metadata=refused,
    )
    monkeypatch.setattr(importlib, "import_module", lambda _name: hub)
    assert installation.model_access_status("small") == "access_check_unavailable"


def test_partial_explicit_runtime_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(runtime.PYTHON_ENV, sys.executable)
    assert "both AudioGhost" in runtime.probe_runtime().reason


@pytest.mark.parametrize(
    "change", [{"extra": True}, {"commit": "wrong"}, {runtime.ROOT_ENV: "/tmp/other"}]
)
def test_installer_manifest_rejects_unexpected_fields_or_locations(change: dict[str, Any]) -> None:
    runtime._LOCAL_ROOT.mkdir()
    value = {
        runtime.PYTHON_ENV: str(runtime._LOCAL_ROOT / "audioghost-env/bin/python"),
        runtime.ROOT_ENV: str(runtime._LOCAL_ROOT / "audioghost-ai"),
        "commit": runtime.AUDIOGHOST_COMMIT,
        "models_downloaded": False,
        **change,
    }
    (runtime._LOCAL_ROOT / "audioghost-runtime.json").write_text(json.dumps(value))
    with pytest.raises(runtime.SeparationError, match="manifest is invalid"):
        runtime._manifest_paths()


def test_installer_manifest_is_bounded() -> None:
    runtime._LOCAL_ROOT.mkdir()
    (runtime._LOCAL_ROOT / "audioghost-runtime.json").write_bytes(b" " * 4097)
    with pytest.raises(runtime.SeparationError, match="manifest is invalid"):
        runtime._manifest_paths()


def test_absent_model_cache_does_not_import_heavy_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "_runtime_paths", lambda: (Path(sys.executable), tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf-cache"))

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("An absent cache must not start a model-runtime import")

    monkeypatch.setattr(runtime, "_run_worker", forbidden)
    status = runtime.probe_runtime()
    assert status.available is False
    assert "model files are not installed" in status.reason


def test_shared_libraries_are_configured_only_in_worker_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_AUDIOGHOST_FFMPEG_LIB", str(tmp_path))
    library_key = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    if os.name == "nt":
        library_key = "PATH"
    monkeypatch.setenv(library_key, "/existing/library")
    child = runtime._worker_environment()
    assert child[library_key].startswith(str(tmp_path))
    assert os.environ[library_key] == "/existing/library"


def test_shared_library_override_must_be_absolute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_AUDIOGHOST_FFMPEG_LIB", "relative-library")
    with pytest.raises(runtime.SeparationError, match="library directory is invalid"):
        runtime._worker_environment()


def test_installation_diagnostics_branches(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(ValueError, match="Unsupported SAM-Audio"):
        installation.model_access_status("giant")

    def missing_hub(_name: str) -> Any:
        raise ImportError("no hub")

    monkeypatch.setattr(importlib, "import_module", missing_hub)
    assert installation.model_access_status("small") == "huggingface_client_not_installed"

    hub_ok = types.SimpleNamespace(
        get_token=lambda: "tok",
        hf_hub_url=lambda *_args: "https://example.invalid/config.json",
        get_hf_file_metadata=lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(importlib, "import_module", lambda _name: hub_ok)
    assert installation.model_access_status("small") == "access_granted"

    class GatedRepoError(RuntimeError):
        pass

    def gated(*_args: Any, **_kwargs: Any) -> None:
        raise GatedRepoError("gated")

    hub_gated = types.SimpleNamespace(
        get_token=lambda: "tok",
        hf_hub_url=lambda *_args: "https://example.invalid/config.json",
        get_hf_file_metadata=gated,
    )
    monkeypatch.setattr(importlib, "import_module", lambda _name: hub_gated)
    assert installation.model_access_status("base") == "model_access_not_approved"

    class HttpErr(RuntimeError):
        response = types.SimpleNamespace(status_code=403)

    def denied(*_args: Any, **_kwargs: Any) -> None:
        raise HttpErr("denied")

    hub_403 = types.SimpleNamespace(
        get_token=lambda: "tok",
        hf_hub_url=lambda *_args: "https://example.invalid/config.json",
        get_hf_file_metadata=denied,
    )
    monkeypatch.setattr(importlib, "import_module", lambda _name: hub_403)
    assert installation.model_access_status("large") == "credential_or_model_access_refused"

    monkeypatch.setattr(
        installation,
        "probe_runtime",
        lambda _size, _dev: runtime.RuntimeStatus(True, "ok", ("cpu",)),
    )
    info = installation.check_installation("small", "cpu", check_access=False)
    assert info["runtime"]["available"] is True and info["model_access"] == "not_checked"

    monkeypatch.setattr(sys, "argv", ["installation", "--model-size", "small"])
    with pytest.raises(SystemExit) as exc_ok:
        installation.main()
    assert exc_ok.value.code == 0
    assert json.loads(capsys.readouterr().out)["runtime"]["available"] is True

    monkeypatch.setattr(
        installation,
        "probe_runtime",
        lambda _size, _dev: runtime.RuntimeStatus(False, "missing"),
    )
    with pytest.raises(SystemExit) as exc_fail:
        installation.main()
    assert exc_fail.value.code == 2


def test_runtime_paths_cache_and_error_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: Path
) -> None:
    fake_py = tmp_path / "py"
    fake_py.write_text("#!/bin/sh\n")
    fake_py.chmod(0o755)
    root = tmp_path / "ag"
    root.mkdir()
    monkeypatch.setenv(runtime.PYTHON_ENV, str(fake_py))
    monkeypatch.setenv(runtime.ROOT_ENV, str(root))
    with pytest.raises(runtime.SeparationError, match="incomplete"):
        runtime._runtime_paths()

    for rel in runtime._ENGINE_BLOBS:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tampered")
    with pytest.raises(runtime.SeparationError, match="do not match"):
        runtime._runtime_paths()

    monkeypatch.setenv(runtime.PYTHON_ENV, str(tmp_path / "missing-py"))
    with pytest.raises(runtime.SeparationError, match="separate Python runtime"):
        runtime._runtime_paths()

    monkeypatch.setenv(runtime.PYTHON_ENV, str(fake_py))
    monkeypatch.setenv(runtime.ROOT_ENV, str(tmp_path / "missing-root"))
    with pytest.raises(runtime.SeparationError, match="reviewed engine checkout"):
        runtime._runtime_paths()

    # Cache lookup via HF_HOME and XDG_CACHE_HOME
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    hf_home = tmp_path / "hf_home"
    snap = hf_home / "hub" / "models--facebook--sam-audio-small" / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "checkpoint.pt").write_bytes(b"w")
    monkeypatch.setenv("HF_HOME", str(hf_home))
    assert runtime._cached_model_present("small") is True
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert runtime._cached_model_present("small") is False

    # probe_runtime ready, not ready, and OSError paths
    real_run_worker = runtime._run_worker
    monkeypatch.setattr(runtime, "_runtime_paths", lambda: (Path(sys.executable), tmp_path))
    monkeypatch.setattr(runtime, "_cached_model_present", lambda _s: True)
    monkeypatch.setattr(runtime, "_run_worker", lambda *_a, **_k: {"ready": True})
    assert runtime.probe_runtime("small", "cpu").available is True
    monkeypatch.setattr(runtime, "_run_worker", lambda *_a, **_k: {"ready": False})
    assert runtime.probe_runtime("small", "cpu").available is False

    def raise_os(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise OSError("spawn failed")

    monkeypatch.setattr(runtime, "_run_worker", raise_os)
    assert runtime.probe_runtime("small", "cpu").available is False

    # _run_worker deadline/cancel pre-checks
    monkeypatch.setattr(runtime, "_run_worker", real_run_worker)
    with pytest.raises(runtime.SeparationError, match="finite and positive"):
        runtime._run_worker({}, timeout_s=0.0)
    with pytest.raises(runtime.SeparationCancelledError):
        runtime._run_worker({}, timeout_s=5.0, should_cancel=lambda: True)

    # validate_audio_file bad path
    with pytest.raises(runtime.SeparationError, match="decoded WAV"):
        runtime.validate_audio_file(tmp_path / "missing.wav")

    # separate_audio post-worker cancel & FileExistsError race
    def good_worker(request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        data, rate = sf.read(source, dtype="float32", always_2d=True)
        sf.write(request["output"], data, rate, subtype="FLOAT")
        return {"selected_stem": "target"}

    monkeypatch.setattr(runtime, "_run_worker", good_worker)
    calls = [0]

    def cancel_after_worker() -> bool:
        calls[0] += 1
        return calls[0] >= 1

    with pytest.raises(runtime.SeparationCancelledError):
        runtime.separate_audio(
            source,
            tmp_path / "cancel.wav",
            prompt="speech",
            operation="extract",
            should_cancel=cancel_after_worker,
        )

    def link_exists(_src: Any, _dst: Any) -> None:
        raise FileExistsError("race")

    monkeypatch.setattr(os, "link", link_exists)
    with pytest.raises(runtime.SeparationError, match="already exists"):
        runtime.separate_audio(source, tmp_path / "race.wav", prompt="speech", operation="extract")


def test_separation_worker_module_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata
    import io

    import torch

    import hawavoclean.separation.worker as worker

    # sha256 + emit (emit writes to sys.__stdout__ by design)
    buf = io.StringIO()
    monkeypatch.setattr(sys, "__stdout__", buf)
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"hello-world")
    assert len(worker.sha256(blob)) == 64
    worker.emit({"kind": "progress", "fraction": 0.25})
    assert "HAWAVOCLEAN_SEPARATION_V1=" in buf.getvalue()

    # cached_assets branches
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"text_encoder": {"name": "t5-base"}}))
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"ckpt")
    txt_w = tmp_path / "model.safetensors"
    txt_w.write_bytes(b"txt")
    txt_cfg = tmp_path / "t5_config.json"
    txt_cfg.write_text("{}")

    cache_map: dict[tuple[str, str], str | None] = {
        ("facebook/sam-audio-small", "config.json"): str(cfg),
        ("facebook/sam-audio-small", "checkpoint.pt"): str(ckpt),
        ("t5-base", "model.safetensors"): str(txt_w),
        ("t5-base", "config.json"): str(txt_cfg),
    }
    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.try_to_load_from_cache = lambda repo, fname: cache_map.get((repo, fname))  # type: ignore[attr-defined]
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *_a, **_k: object()
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    assert worker.cached_assets("facebook/sam-audio-small") == (cfg, ckpt, txt_w)

    cache_map[("t5-base", "model.safetensors")] = None
    with pytest.raises(FileNotFoundError, match="missing_assets"):
        worker.cached_assets("facebook/sam-audio-small")
    cache_map[("facebook/sam-audio-small", "checkpoint.pt")] = None
    with pytest.raises(FileNotFoundError, match="missing_assets"):
        worker.cached_assets("facebook/sam-audio-small")

    # worker.run happy path (probe + extract + remove) and guards
    import scipy.signal  # pre-import before patching np.__version__

    _ = scipy.signal.resample_poly
    cache_map[("facebook/sam-audio-small", "checkpoint.pt")] = str(ckpt)
    cache_map[("t5-base", "model.safetensors")] = str(txt_w)

    real_distribution = importlib.metadata.distribution
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda pkg: "2.9.0" if pkg in {"torch", "torchaudio"} else "1.0.0",
    )
    monkeypatch.setattr(np, "__version__", "1.26.4")
    fake_dist = types.SimpleNamespace(
        read_text=lambda _n: json.dumps({"vcs_info": {"commit_id": worker.SAM_AUDIO_COMMIT}})
    )
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: fake_dist if name == "sam-audio" else real_distribution(name),
    )

    class _FakeLite:
        sample_rate = 16000
        precision = "fp32"

    def fake_separate(
        _lite: Any,
        audio_t: torch.Tensor,
        _prompt: str,
        *,
        progress: Any,
        decode_target: bool,
        decode_residual: bool,
        **_kwargs: Any,
    ) -> Any:
        progress(0.5, "half")
        progress(1.0, "done")
        return types.SimpleNamespace(
            target=audio_t * 0.5 if decode_target else None,
            residual=audio_t * 0.25 if decode_residual else None,
            steps=4,
            chunks=1,
        )

    fake_sam_lite = types.ModuleType("workers.sam_lite")
    fake_sam_lite.QUALITY_STEPS = {"high": 16}  # type: ignore[attr-defined]
    fake_sam_lite.model_repo_id = lambda size: f"facebook/sam-audio-{size}"  # type: ignore[attr-defined]
    fake_sam_lite.load_lite_model = lambda *_a, **_k: _FakeLite()  # type: ignore[attr-defined]
    fake_sam_lite.separate = fake_separate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "workers.sam_lite", fake_sam_lite)
    monkeypatch.setitem(sys.modules, "sam_audio", types.ModuleType("sam_audio"))

    probe_meta = worker.run(
        {"action": "probe", "model_size": "small", "device": "cpu", "root": str(tmp_path)}
    )
    assert probe_meta["ready"] is True

    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    sf.write(in_wav, np.full((1600, 1), 0.1, dtype=np.float32), 48000, subtype="FLOAT")
    sep_meta = worker.run(
        {
            "action": "separate",
            "input": str(in_wav),
            "output": str(out_wav),
            "prompt": "speech",
            "operation": "extract",
            "model_size": "small",
            "quality": "high",
            "seed": 0,
            "device": "cpu",
            "root": str(tmp_path),
        }
    )
    assert sep_meta["selected_stem"] == "target"
    assert sf.info(out_wav).frames == 1600 and sf.info(out_wav).samplerate == 48000

    # Incompatible commit triggers runtime_incompatible
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda _n: types.SimpleNamespace(read_text=lambda _f: "{}"),
    )
    with pytest.raises(RuntimeError, match="runtime_incompatible"):
        worker.run(
            {"action": "probe", "model_size": "small", "device": "cpu", "root": str(tmp_path)}
        )

    # worker.main stdin + error-code mapping branches
    monkeypatch.setenv("HAWAVOCLEAN_SEPARATION_PARENT_PID", str(os.getppid()))
    monkeypatch.setattr(worker, "run", lambda _req: {"ready": True})
    monkeypatch.setattr(
        sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(b'{"action":"probe"}\n'))
    )
    assert worker.main() == 0

    for exc, expected_code in (
        (ImportError("bad"), "runtime_incompatible"),
        (FileNotFoundError("no file"), "missing_assets"),
        (RuntimeError("CUDA out of memory"), "out_of_memory"),
        (RuntimeError("unexpected boom"), "inference_failed"),
    ):

        def _boom(_req: Any, _exc: Exception = exc) -> dict[str, Any]:
            raise _exc

        buf.seek(0)
        buf.truncate(0)
        monkeypatch.setattr(worker, "run", _boom)
        monkeypatch.setattr(
            sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(b'{"action":"probe"}\n'))
        )
        assert worker.main() == 2
        assert expected_code in buf.getvalue()

    # Oversized request & non-dict request
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(b"x" * 16386)))
    assert worker.main() == 2
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(b"[]\n")))
    assert worker.main() == 2

    # Parent PID mismatch exits 130 on POSIX
    if os.name == "posix":
        monkeypatch.setenv("HAWAVOCLEAN_SEPARATION_PARENT_PID", "1")
        assert worker.main() == 130
