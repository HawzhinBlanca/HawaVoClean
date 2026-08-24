"""UniverSR upstream architecture reference and official model adapter.

Pinned to commit: 26dc21c44e11f9f19e823f02b0d4641dd5ea5af2 (MIT License).
Paper: arXiv 2510.00771.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import signal

from hawavoclean.errors import ModelProvenanceError
from hawavoclean.hashing import hash_file
from hawavoclean.restoration.base import RestorationCandidate, Restorer
from hawavoclean.restoration.protected_band import (
    compute_transition_mask,
    merge_protected_spectrum,
)

logger = logging.getLogger("hawavoclean.restoration.universr")

# Official UniverSR checkpoints
UNIVERSR_WEIGHTS_REGISTRY: dict[str, dict[str, str]] = {
    "universr-speech": {
        "repo_id": "woongzip1/universr-speech",
        "sha256": "d29130a1c1558dbacaa89c0555bb34dd353c2e2c3a76e91eff876aa13166bc9b",
        "config_sha256": "2e035dacc833368319f9d5ba1a34e5efd343faf749008d433e309f3c65f4dac9",
        "license": "CC-BY-4.0",
    },
    "universr-audio": {
        "repo_id": "woongzip1/universr-audio",
        "sha256": "eb99f98943cc32fa82226c2da14b32b5d890416070af4946acbce442b30dc20b",
        "config_sha256": "2e035dacc833368319f9d5ba1a34e5efd343faf749008d433e309f3c65f4dac9",
        "license": "CC-BY-4.0",
    },
}


class UniverSRBaseline(Restorer):
    """UniverSR official baseline restorer operating in the complex-STFT domain.

    Supports both official neural flow-matching inference (via vendors/universr)
    and deterministic complex-STFT spectral extrapolation.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        model_name: str = "universr-speech",
        n_fft: int = 2048,
        hop_length: int = 480,
        win_length: int = 2048,
        device: str | None = None,
        use_neural: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.model_name = model_name
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.use_neural = use_neural

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self._neural_model: Any = None
        if self.use_neural:
            self._init_neural_model()

    def _verify_upstream_artifacts(self, repo_info: dict[str, str]) -> None:
        """Check the downloaded checkpoint against the hashes this module attests.

        The registry records a sha256 for every upstream artifact. Recording one
        and never checking it is worse than recording nothing: benchmark evidence
        would carry a provenance hash that nothing ever compared against, so an
        upstream re-upload would go unnoticed.
        """
        from huggingface_hub import hf_hub_download

        for filename, expected_key in (
            ("pytorch_model.bin", "sha256"),
            ("config.yaml", "config_sha256"),
        ):
            expected = repo_info.get(expected_key)
            if not expected:
                continue
            local = Path(hf_hub_download(repo_id=repo_info["repo_id"], filename=filename))
            actual = hash_file(local)
            if actual != expected:
                raise ModelProvenanceError(
                    f"UniverSR artifact {repo_info['repo_id']}/{filename} does not match the "
                    f"attested hash: expected {expected}, got {actual}"
                )

    def _init_neural_model(self) -> None:
        """Initialize official UniverSR neural model if available."""
        repo_info = UNIVERSR_WEIGHTS_REGISTRY.get(
            self.model_name, UNIVERSR_WEIGHTS_REGISTRY["universr-speech"]
        )
        try:
            import sys

            vendor_path = (
                Path(__file__).resolve().parent.parent.parent.parent / "vendors" / "universr"
            )
            if vendor_path.exists() and str(vendor_path) not in sys.path:
                sys.path.insert(0, str(vendor_path))

            from universr.inference import UniverSR  # type: ignore[import-not-found]

            self._verify_upstream_artifacts(repo_info)
            self._neural_model = UniverSR.from_pretrained(repo_info["repo_id"], device=self.device)
            logger.info(
                "Loaded official UniverSR model: %s on %s (weights %s)",
                repo_info["repo_id"],
                self.device,
                repo_info["sha256"][:12],
            )
        except ModelProvenanceError:
            # A hash mismatch means the upstream artifact is not the one this
            # baseline claims to run. Falling back to DSP would bury that.
            raise
        except Exception as e:
            logger.warning(
                "Could not load official UniverSR neural weights (%s); using DSP extrapolation", e
            )
            self._neural_model = None

    def _stft(self, audio: np.ndarray) -> np.ndarray:
        """Compute complex STFT."""
        _, _, Zxx = signal.stft(
            audio,
            fs=self.sample_rate,
            window="hann",
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            boundary="zeros",
            padded=True,
        )
        return np.asarray(Zxx, dtype=np.complex64)

    def _istft(self, Zxx: np.ndarray) -> np.ndarray:
        """Compute inverse complex STFT."""
        _, x = signal.istft(
            Zxx,
            fs=self.sample_rate,
            window="hann",
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            boundary="zeros",
        )
        return np.asarray(x, dtype=np.float32)

    def restore(
        self,
        audio_48k: np.ndarray,
        sample_rate: int,  # noqa: ARG002
        effective_cutoff_hz: float,
        speaker_id: str | None = None,  # noqa: ARG002
        speaker_embedding: np.ndarray | None = None,  # noqa: ARG002
        f0_trajectory: np.ndarray | None = None,  # noqa: ARG002
        vuv_mask: np.ndarray | None = None,  # noqa: ARG002
        strengths: list[float] | None = None,
        seed: int = 42,
    ) -> list[RestorationCandidate]:
        """Generate restored candidates using UniverSR flow matching or baseline extrapolation."""
        if strengths is None:
            strengths = [1.0, 0.75, 0.5, 0.25, 0.0]

        is_multichannel = audio_48k.ndim == 2
        channels = audio_48k if is_multichannel else audio_48k[np.newaxis, :]
        n_channels = channels.shape[0]
        n_samples = channels.shape[1]

        candidates: list[RestorationCandidate] = []
        restored_channels_by_strength: dict[float, list[np.ndarray]] = {s: [] for s in strengths}

        # Map effective cutoff Hz to standard UniverSR input_sr
        if effective_cutoff_hz <= 5000:
            mapped_sr = 8000
        elif effective_cutoff_hz <= 7000:
            mapped_sr = 12000
        elif effective_cutoff_hz <= 10000:
            mapped_sr = 16000
        else:
            mapped_sr = 24000

        for ch_idx in range(n_channels):
            ch_audio = channels[ch_idx]
            if len(ch_audio) < self.win_length:
                for s in strengths:
                    restored_channels_by_strength[s].append(ch_audio)
                continue

            Z_obs = self._stft(ch_audio)
            n_freqs, n_frames = Z_obs.shape
            mask = compute_transition_mask(n_freqs, self.sample_rate, effective_cutoff_hz)

            Z_gen: np.ndarray | None = None

            # Try neural enhancement first
            if self._neural_model is not None:
                try:
                    torch.manual_seed(seed)
                    with torch.no_grad():
                        tensor_audio = (
                            torch.from_numpy(ch_audio)
                            .float()
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .to(self.device)
                        )
                        sr_khz = mapped_sr // 1000
                        MIN_SAMPLES = 32_768
                        orig_len = tensor_audio.shape[-1]
                        padded = torch.nn.functional.pad(
                            tensor_audio, (0, max(0, MIN_SAMPLES - orig_len))
                        )
                        enhanced_t = self._neural_model._inference(
                            padded,
                            sr_khz=sr_khz,
                            ode_method="midpoint",
                            ode_steps=4,
                            guidance_scale=0.0,
                        )
                        enhanced_np = enhanced_t.squeeze().cpu().numpy()[:orig_len]
                        Z_gen = self._stft(enhanced_np)
                        if Z_gen.shape != Z_obs.shape:
                            Z_gen = None
                except Exception as e:
                    logger.debug("Neural UniverSR inference failed: %s; falling back to DSP", e)
                    Z_gen = None

            # Fallback to spectral extrapolation if neural is unavailable
            if Z_gen is None:
                freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)
                cutoff_bin = int(np.argmin(np.abs(freqs - effective_cutoff_hz)))
                Z_gen = np.zeros_like(Z_obs)
                if cutoff_bin > 10:
                    low_band = Z_obs[10:cutoff_bin, :]
                    n_rep = (n_freqs - cutoff_bin + len(low_band) - 1) // len(low_band)
                    tiled = np.tile(low_band, (n_rep + 1, 1))[: n_freqs - cutoff_bin, :]
                    hf_freqs = freqs[cutoff_bin:]
                    tilt = np.exp(-0.00015 * (hf_freqs - effective_cutoff_hz))[:, np.newaxis]
                    Z_gen[cutoff_bin:, :] = tiled * tilt

            for s in strengths:
                if s == 0.0:
                    restored_channels_by_strength[s].append(ch_audio)
                else:
                    Z_merged = merge_protected_spectrum(Z_obs, Z_gen, mask, strength=s)
                    x_rec = self._istft(Z_merged)
                    if len(x_rec) < n_samples:
                        x_rec = np.pad(x_rec, (0, n_samples - len(x_rec)))
                    else:
                        x_rec = x_rec[:n_samples]
                    restored_channels_by_strength[s].append(np.clip(x_rec, -1.0, 1.0))

        for s in strengths:
            arr_list = restored_channels_by_strength[s]
            merged_audio = np.stack(arr_list, axis=0) if is_multichannel else arr_list[0]
            candidates.append(
                RestorationCandidate(
                    strength=s,
                    audio=merged_audio.astype(np.float32),
                    cutoff_hz=effective_cutoff_hz,
                )
            )

        return candidates
