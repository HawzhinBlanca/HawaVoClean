"""Enhancer core registry: core_id -> implementation class and lockfile.

Every registered core carries the name of its lockfile and a callable that
recomputes the implementation's params hash WITHOUT loading the model, so
provenance is verifiable in any install.
"""

from collections.abc import Callable
from dataclasses import dataclass

from hawavoclean.enhancement.production import WienerSpectralEnhancer, wiener_params_hash
from hawavoclean.enhancement.protocol import Enhancer
from hawavoclean.enhancement.studio import StudioVoiceCore, studio_params_hash
from hawavoclean.enhancement.studio_lowband import StudioLowBandCore, studio_lowband_params_hash


@dataclass(frozen=True)
class CoreRegistration:
    """A runnable enhancement core and its provenance anchors."""

    enhancer_class: type[Enhancer]
    lock_filename: str
    implementation_params_hash: Callable[[], str]
    requires_modules: tuple[str, ...] = ()
    #: Does this core's inference actually run on ``runtime.device``? A
    #: classical-DSP core is numpy on the CPU whatever the config asks for,
    #: and the report must name the device that ran, not the one requested.
    device_aware: bool = False


CORE_REGISTRY: dict[str, CoreRegistration] = {
    "wiener-dd-48k-v1": CoreRegistration(
        enhancer_class=WienerSpectralEnhancer,
        lock_filename="production-core.lock.toml",
        implementation_params_hash=wiener_params_hash,
    ),
    "studio-dfn3-48k-v1": CoreRegistration(
        enhancer_class=StudioVoiceCore,
        lock_filename="studio-core.lock.toml",
        implementation_params_hash=studio_params_hash,
        requires_modules=("df", "torch", "nara_wpe"),
        device_aware=True,
    ),
    # Shares the studio core's vendored DFN3 weights; no WPE, so nara_wpe is
    # not among its requirements.
    "studio-dfn3-lowband-48k-v1": CoreRegistration(
        enhancer_class=StudioLowBandCore,
        lock_filename="studio-lowband-core.lock.toml",
        implementation_params_hash=studio_lowband_params_hash,
        requires_modules=("df", "torch"),
        device_aware=True,
    ),
}


def resolve_core(core_id: str) -> CoreRegistration:
    """Look up a registered core or fail loudly."""
    if core_id not in CORE_REGISTRY:
        raise KeyError(
            f"Unknown enhancement core {core_id!r}; registered cores: {sorted(CORE_REGISTRY)}"
        )
    return CORE_REGISTRY[core_id]
