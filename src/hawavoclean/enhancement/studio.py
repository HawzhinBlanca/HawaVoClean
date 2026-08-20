"""Studio voice restoration core: WPE dereverberation + DeepFilterNet3 denoising.

This core contains a REAL neural model — DeepFilterNet3 (MIT OR Apache-2.0,
https://github.com/Rikorose/DeepFilterNet), whose weights are vendored in
``hawavoclean/resources/models/deepfilternet3/`` and hash-locked in
``studio-core.lock.toml``. It is preceded by single-channel WPE
dereverberation (nara_wpe, MIT).

Scope, honestly: DeepFilterNet3 is a *speech* enhancer. It attenuates
everything it does not classify as speech — including sustained musical
tones. On musical material expect aggressive suppression; the fidelity
guard remains in the loop and reverts units whose spectral content it
cannot verify. The output is NOT phase-coherent with the input, so the
policy evaluates only the full-strength candidate (no residual blending).

This is the one core whose inference honours ``runtime.device``. The device is
resolved by :mod:`hawavoclean.runtime` (``auto`` is CPU today — see that
module for the measurements), pinned into DeepFilterNet's own device lookup so
the model and its feature tensors cannot disagree, and recorded in the report.
It is NOT part of ``STUDIO_PARAMS``: the lockfile pins what the model is, and
that is the same model on every device.
"""

import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

from hawavoclean.audio.resample import resample_audio
from hawavoclean.enhancement.protocol import EnhancementResult, Enhancer, EnhancerMetadata
from hawavoclean.hashing import hash_file, hash_json_canonical
from hawavoclean.logging import get_logger
from hawavoclean.runtime import active_device, check_memory_budget, resolve_device

logger = get_logger("studio-core")

_MODEL_DIR = Path(__file__).resolve().parents[1] / "resources" / "models" / "deepfilternet3"

# Frozen studio parameters. Changing any of these is a new core and requires
# relocking studio-core.lock.toml (verified at preflight and by audit-models).
STUDIO_PARAMS: dict[str, float | int | bool | str] = {
    "model": "DeepFilterNet3",
    "internal_sample_rate": 48000,
    "wpe_dereverb": True,
    "wpe_n_fft": 1024,
    "wpe_hop": 256,
    "wpe_taps": 10,
    "wpe_delay": 3,
    "wpe_iterations": 3,
    "atten_lim_db": 0,  # 0 = unlimited noise attenuation
    # Decay-gated late-reverb suppression after DFN3 (see finishing/dereverb.py).
    # Setting chosen 2026-08-19 on a real recording as the strongest that
    # clears the spectral-hole guard on EVERY unit (worst 0.085 < 0.10):
    # phrase tails -5.6 dB @50 ms / -7.5 dB @100 ms, voice tonality unchanged.
    # Hotter settings (rt60 0.5 / floor -15) scored 0.14 and the continuity
    # rule then cascaded the whole take back to original.
    "tail_suppress": True,
    "tail_rt60_s": 0.4,
    "tail_floor_db": -10.0,
    "tail_onset_protect_db": 3.0,
}


def studio_params_hash() -> str:
    """Canonical hash of the studio core's parameters AND its weights digests.

    Importable without torch, so provenance can be audited in a base install.
    """
    payload: dict[str, object] = dict(STUDIO_PARAMS)
    payload["weights_sha256"] = studio_weight_digests()
    return hash_json_canonical(payload)


def studio_weight_digests() -> dict[str, str]:
    """Recompute the vendored weight digests from disk."""
    digests: dict[str, str] = {}
    for rel in ("config.ini", "checkpoints/model_120.ckpt.best"):
        p = _MODEL_DIR / rel
        digests[f"deepfilternet3/{rel}"] = hash_file(p) if p.exists() else "MISSING"
    return digests


def _install_torchaudio_compat() -> None:
    """deepfilternet 0.5.6 imports torchaudio.backend.common.AudioMetaData,
    removed in torchaudio >= 2.2. It is used only as a type annotation in a
    module we never call; provide the symbol so the import succeeds."""
    if "torchaudio.backend.common" in sys.modules:
        return
    import torchaudio  # type: ignore[import-untyped]

    common = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:  # annotation-only stand-in
        pass

    common.AudioMetaData = getattr(torchaudio, "AudioMetaData", AudioMetaData)  # type: ignore[attr-defined]
    backend = types.ModuleType("torchaudio.backend")
    backend.common = common  # type: ignore[attr-defined]
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common


def load_deepfilternet3(device: str) -> tuple[Any, Any]:
    """Initialise DeepFilterNet3 from the vendored, hash-locked weights.

    Every core that runs DFN3 loads it through here, so ``_MODEL_DIR`` — the
    directory whose digests the lockfiles pin — is named exactly once.
    """
    _install_torchaudio_compat()
    import torch
    from df.config import config as df_config
    from df.enhance import init_df

    model, df_state, _ = init_df(model_base_dir=str(_MODEL_DIR), log_level="ERROR", log_file=None)
    # init_df chooses independently, while df.enhance re-reads this global
    # setting for every feature tensor. Pin both the model and feature path to
    # the runtime decision; never save(), which would mutate locked config.ini.
    df_config.set("DEVICE", device, str, "train")
    from df.utils import get_device

    effective = str(get_device())
    if effective != device:
        raise ValueError(
            f"DeepFilterNet resolved device {effective!r} but this run is "
            f"configured for {device!r}; a stray DEVICE environment variable "
            "is overriding runtime.device. Unset DEVICE."
        )
    model = model.to(torch.device(device))
    return model, df_state


def run_deepfilternet3(
    model: Any,
    df_state: Any,
    audio_48k: np.ndarray[Any, np.dtype[np.float32]],
    atten_lim_db: float,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Run DFN3 over 48 kHz mono audio.

    ``atten_lim_db`` follows the locked convention shared by every DFN3 core:
    0 means *unlimited* attenuation, and is passed to the model as ``None``.
    """
    import torch
    from df.enhance import enhance as df_enhance

    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(audio_48k)).unsqueeze(0)
        out = df_enhance(
            model,
            df_state,
            tensor,
            atten_lim_db=atten_lim_db if atten_lim_db > 0 else None,
        )
    return np.asarray(out.squeeze(0).numpy(), dtype=np.float32)


class StudioVoiceCore(Enhancer):
    """WPE dereverberation followed by DeepFilterNet3 speech enhancement."""

    def __init__(
        self,
        core_id: str = "studio-dfn3-48k-v1",
        sample_rate: int = 48000,
        phase_coherent: bool = False,
        device: str | None = None,
    ) -> None:
        if phase_coherent:
            raise ValueError(
                "StudioVoiceCore is not phase-coherent (deep filtering "
                "modifies phase); configure phase_coherent = false"
            )
        internal_sr = int(STUDIO_PARAMS["internal_sample_rate"])
        if sample_rate != internal_sr:
            raise ValueError(
                f"StudioVoiceCore runs at {internal_sr} Hz internally; "
                f"model_sample_rate must be {internal_sr}, got {sample_rate}"
            )
        self._core_id = core_id
        self._sample_rate = internal_sr
        # The device is settled here, in whichever process holds the core —
        # the parent for an in-process run, the spawned worker otherwise —
        # and both read the same published value, so the report's
        # environment.compute_device names the device that really ran.
        self._device = (
            active_device() if device is None else resolve_device(device, core_id=core_id).resolved
        )
        self._metadata = EnhancerMetadata(
            core_id=core_id,
            version="1.0.0",
            algorithm=(
                "single-channel WPE dereverberation + DeepFilterNet3 neural "
                "speech enhancement (vendored weights)"
            ),
            sample_rate=self._sample_rate,
            phase_coherent=False,  # deep filtering modifies phase
            params_hash=studio_params_hash(),
        )
        self._model: Any = None
        self._df_state: Any = None

    @property
    def metadata(self) -> EnhancerMetadata:
        return self._metadata

    @property
    def device(self) -> str:
        """The compute device this core's inference runs on."""
        return self._device

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._model, self._df_state = load_deepfilternet3(self._device)
        if self._device != "cpu":
            logger.warning(
                f"Studio core running on {self._device!r}. GPU arithmetic does not "
                "match the CPU reference bit-for-bit; the report records the device."
            )

    def warmup(self) -> None:
        self._ensure_model()
        dummy = np.zeros(self._sample_rate // 2, dtype=np.float32)
        self.enhance(dummy, self._sample_rate)

    def _wpe_dereverb(
        self, audio: np.ndarray[Any, np.dtype[np.float32]]
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        import scipy.signal
        from nara_wpe.wpe import wpe

        n_fft = int(STUDIO_PARAMS["wpe_n_fft"])
        hop = int(STUDIO_PARAMS["wpe_hop"])
        _, _, stft = scipy.signal.stft(audio, nperseg=n_fft, noverlap=n_fft - hop, padded=True)
        # nara_wpe expects (F, D, T)
        dereverbed = wpe(
            stft[:, None, :],
            taps=int(STUDIO_PARAMS["wpe_taps"]),
            delay=int(STUDIO_PARAMS["wpe_delay"]),
            iterations=int(STUDIO_PARAMS["wpe_iterations"]),
        )[:, 0, :]
        _, out = scipy.signal.istft(dereverbed, nperseg=n_fft, noverlap=n_fft - hop)
        out32 = np.asarray(out, dtype=np.float32)
        if len(out32) < len(audio):
            out32 = np.pad(out32, (0, len(audio) - len(out32)))
        return out32[: len(audio)]

    def enhance(
        self,
        waveform: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int,
    ) -> EnhancementResult:
        # Before taking the unit, not after: a worker that has already blown
        # runtime.worker_memory_limit_mb stops accepting work rather than
        # discarding a result it has already paid for.
        check_memory_budget("studio")
        t_start = time.perf_counter()
        orig_len = len(waveform)
        if orig_len == 0:
            return EnhancementResult(
                waveform=waveform.copy(),
                sample_rate=sample_rate,
                model_runtime_ms=0.0,
                input_samples=0,
                output_samples=0,
            )

        self._ensure_model()
        audio_48k = resample_audio(waveform, sample_rate, self._sample_rate)

        if bool(STUDIO_PARAMS["wpe_dereverb"]):
            audio_48k = self._wpe_dereverb(audio_48k)

        enhanced_48k = run_deepfilternet3(
            self._model, self._df_state, audio_48k, float(STUDIO_PARAMS["atten_lim_db"])
        )

        if bool(STUDIO_PARAMS["tail_suppress"]):
            from hawavoclean.finishing.dereverb import suppress_late_reverb

            enhanced_48k = suppress_late_reverb(
                enhanced_48k,
                self._sample_rate,
                rt60_s=float(STUDIO_PARAMS["tail_rt60_s"]),
                floor_db=float(STUDIO_PARAMS["tail_floor_db"]),
                onset_protect_db=float(STUDIO_PARAMS["tail_onset_protect_db"]),
            )

        enhanced_out = resample_audio(
            enhanced_48k, self._sample_rate, sample_rate, target_samples=orig_len
        )
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return EnhancementResult(
            waveform=enhanced_out,
            sample_rate=sample_rate,
            model_runtime_ms=t_elapsed_ms,
            input_samples=orig_len,
            output_samples=len(enhanced_out),
        )
