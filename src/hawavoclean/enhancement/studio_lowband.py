"""Band-split restoration core: DeepFilterNet3's low band over the original.

The problem this core exists for
--------------------------------
A muffled recording whose noise is low-frequency tonal rumble defeats both
of the other cores. The Wiener core (``wiener-dd-48k-v1``) cannot get under
the rumble without hitting its gain floor, and leaves the voice sitting on
it. The full-band studio core (``studio-dfn3-48k-v1``) removes the rumble,
but on this material DeepFilterNet3 also takes most of the consonant band
with it — measured on the Teat1vo lab fixture, DFN3's full-band output
retained 0.22 of the original 2-8 kHz energy, the fidelity guard called it,
and the whole take came back as original.

What this core does
-------------------
Run DFN3 over the FULL band, so the model sees the signal it was trained on,
then keep only its low band and hand the rest back to the original::

    enhanced = DFN3(x)                       # unlimited attenuation
    out      = lowpass(enhanced) + (x - lowpass(x))

Below the crossover the output is DFN3's; above it, it converges on the
untouched original. Consonants are therefore preserved by construction
rather than by hoping the model spares them: measured consonant retention is
~1.0 and the guard's spectral-hole score stays at 0.066 against its 0.100
threshold.

Why one filter, subtracted, and not two
---------------------------------------
Both bands come from the SAME lowpass, and the high band is the arithmetic
complement of the low one. That makes the crossover exact: if DFN3 ever
returned its input unchanged, ``lowpass(x) + (x - lowpass(x))`` is ``x`` to
a float rounding error, at every frequency. Two independently designed
filters — a lowpass for one path and a highpass for the other — would sum to
a magnitude ripple and a phase step right where the voice's first formant
lives. ``test_studio_lowband_core.py`` pins the exactness.

Where the crossover came from
-----------------------------
Measured on the Teat1vo lab fixture, against the guard's own spectral-hole
score (threshold 0.100):

    crossover      hole score     consonant retention
    ~700 Hz          0.050              1.000
    1000 Hz          0.066              ~0.998      <- shipped
    1300 Hz          0.088              0.988
    1500 Hz          0.103              0.964       <- guard rejects
    2500 Hz          0.187              0.497

Above roughly 1.1 kHz DFN3 starts reaching into the consonant band, the
score crosses the threshold, and the guard reverts the unit — so the
crossover is not a taste setting, it is the line between a core that ships
audio and one that always hands back the original. It is part of
``params_hash``: moving it is a new core and a relock.

Scope, honestly: this core is NOT phase-coherent — DFN3 rewrites phase below
the crossover — so the policy evaluates only the full-strength candidate.
The guard judges every unit exactly as it does for any other core, and a
unit it rejects keeps its original audio.
"""

import time
from typing import Any

import numpy as np

from hawavoclean.audio.resample import resample_audio
from hawavoclean.enhancement.protocol import EnhancementResult, Enhancer, EnhancerMetadata
from hawavoclean.enhancement.studio import (
    load_deepfilternet3,
    run_deepfilternet3,
    studio_weight_digests,
)
from hawavoclean.hashing import hash_json_canonical
from hawavoclean.logging import get_logger
from hawavoclean.runtime import active_device, check_memory_budget, resolve_device

logger = get_logger("studio-lowband-core")

# Frozen band-split parameters. Changing any of these — the crossover most of
# all — is a new core and requires relocking studio-lowband-core.lock.toml
# (verified at preflight and by audit-models).
STUDIO_LOWBAND_PARAMS: dict[str, float | int | bool | str] = {
    "model": "DeepFilterNet3",
    "internal_sample_rate": 48000,
    # The crossover: DFN3's output below, the original above. See the table in
    # the module docstring for the measurement that chose it.
    "crossover_hz": 1000.0,
    "crossover_order": 4,
    # One zero-phase Butterworth, used for both paths: high = input - low.
    "crossover_form": "complementary_zero_phase_butterworth",
    "atten_lim_db": 0,  # 0 = unlimited noise attenuation
    # The full-band studio core's dereverberation and tail suppression are
    # deliberately NOT part of this chain. Both were tuned against full-band
    # speech, and this core's job is one thing: get the rumble out from under
    # the voice without touching the consonants.
    "wpe_dereverb": False,
    "tail_suppress": False,
}


def studio_lowband_params_hash() -> str:
    """Canonical hash of this core's parameters AND the shared weights digests.

    Importable without torch, so provenance can be audited in a base install.
    """
    payload: dict[str, object] = dict(STUDIO_LOWBAND_PARAMS)
    payload["weights_sha256"] = studio_weight_digests()
    return hash_json_canonical(payload)


def lowpass_zero_phase(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    crossover_hz: float,
    order: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Zero-phase Butterworth lowpass.

    Forward-backward, so the result stays sample-aligned with the input and
    the complement ``input - lowpass(input)`` is a genuine high band rather
    than a delayed residual. Signals too short to pad for filtfilt get a
    shorter pad instead of an exception; one sample or fewer has no band
    structure at all and is returned unchanged.
    """
    import scipy.signal

    x32 = np.asarray(waveform, dtype=np.float32)
    if len(x32) < 2:
        return x32.copy()
    sos = scipy.signal.butter(order, crossover_hz, btype="low", fs=sample_rate, output="sos")
    # scipy's default pad is 3*(2*n_sections+1) samples at each end; a unit
    # shorter than that is not an error here, just a shorter pad.
    padlen = min(3 * (2 * len(sos) + 1), len(x32) - 1)
    return np.asarray(
        scipy.signal.sosfiltfilt(sos, x32.astype(np.float64), padlen=padlen), dtype=np.float32
    )


def crossover_mix(
    enhanced: np.ndarray[Any, np.dtype[np.float32]],
    original: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    crossover_hz: float,
    order: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """``enhanced`` below the crossover, ``original`` above, summing exactly.

    Both bands are taken with the same lowpass, the high one by subtraction,
    so ``crossover_mix(x, x, ...) == x`` to a float rounding error — the
    property that guarantees no notch or bump at the crossover.
    """
    low = lowpass_zero_phase(enhanced, sample_rate, crossover_hz, order)
    orig_low = lowpass_zero_phase(original, sample_rate, crossover_hz, order)
    return (low + (original - orig_low)).astype(np.float32)


def _match_length(
    arr: np.ndarray[Any, np.dtype[np.float32]], target: int
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Pad or trim to exactly ``target`` samples.

    DFN3 works in frames and can hand back a few samples more or fewer than
    it was given; the two paths must line up sample-for-sample before they
    are crossed over, or the high band would be added at the wrong offset.
    """
    if len(arr) == target:
        return arr
    if len(arr) < target:
        return np.pad(arr, (0, target - len(arr)))
    return arr[:target]


class StudioLowBandCore(Enhancer):
    """DeepFilterNet3 below the crossover, the original signal above it."""

    def __init__(
        self,
        core_id: str = "studio-dfn3-lowband-48k-v1",
        sample_rate: int = 48000,
        phase_coherent: bool = False,
        device: str | None = None,
    ) -> None:
        if phase_coherent:
            raise ValueError(
                "StudioLowBandCore is not phase-coherent (deep filtering "
                "modifies phase below the crossover); configure "
                "phase_coherent = false"
            )
        internal_sr = int(STUDIO_LOWBAND_PARAMS["internal_sample_rate"])
        if sample_rate != internal_sr:
            raise ValueError(
                f"StudioLowBandCore runs at {internal_sr} Hz internally; "
                f"model_sample_rate must be {internal_sr}, got {sample_rate}"
            )
        self._core_id = core_id
        self._sample_rate = internal_sr
        self._device = (
            active_device() if device is None else resolve_device(device, core_id=core_id).resolved
        )
        self._metadata = EnhancerMetadata(
            core_id=core_id,
            version="1.0.0",
            algorithm=(
                "DeepFilterNet3 (unlimited attenuation) crossed over with the "
                f"original at {float(STUDIO_LOWBAND_PARAMS['crossover_hz']):g} Hz: "
                "model below, untouched original above, complementary "
                "zero-phase split"
            ),
            sample_rate=self._sample_rate,
            phase_coherent=False,  # deep filtering modifies phase below the crossover
            params_hash=studio_lowband_params_hash(),
        )
        self._model: Any = None
        self._df_state: Any = None

    @property
    def metadata(self) -> EnhancerMetadata:
        return self._metadata

    @property
    def device(self) -> str:
        """The compute device this core's DFN3 inference runs on."""
        return self._device

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._model, self._df_state = load_deepfilternet3(self._device)
        if self._device != "cpu":
            logger.warning(
                f"Lowband core running on {self._device!r}. GPU arithmetic does not "
                "match the CPU reference bit-for-bit; the report records the device."
            )

    def warmup(self) -> None:
        self._ensure_model()
        dummy = np.zeros(self._sample_rate // 2, dtype=np.float32)
        self.enhance(dummy, self._sample_rate)

    def enhance(
        self,
        waveform: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int,
    ) -> EnhancementResult:
        check_memory_budget("lowband")
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

        # DFN3 sees the FULL band — it is trained on full-band speech, and a
        # lowpassed input is out of distribution for it. Only its low band is
        # kept; the crossover is what confines its effect.
        deep = run_deepfilternet3(
            self._model, self._df_state, audio_48k, float(STUDIO_LOWBAND_PARAMS["atten_lim_db"])
        )
        enhanced_48k = crossover_mix(
            _match_length(deep, len(audio_48k)),
            audio_48k,
            self._sample_rate,
            float(STUDIO_LOWBAND_PARAMS["crossover_hz"]),
            int(STUDIO_LOWBAND_PARAMS["crossover_order"]),
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
