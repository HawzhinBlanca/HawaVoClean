"""Enhancer core registry: core_id -> implementation class and lockfile.

Every registered core carries the name of its lockfile and a callable that
recomputes the implementation's params hash WITHOUT loading the model, so
provenance is verifiable in any install.
"""

from collections.abc import Callable
from dataclasses import dataclass

from voiceclean.enhancement.production import WienerSpectralEnhancer, wiener_params_hash
from voiceclean.enhancement.protocol import Enhancer
from voiceclean.enhancement.studio import StudioVoiceCore, studio_params_hash


@dataclass(frozen=True)
class CoreRegistration:
    """A runnable enhancement core and its provenance anchors."""

    enhancer_class: type[Enhancer]
    lock_filename: str
    implementation_params_hash: Callable[[], str]
    requires_modules: tuple[str, ...] = ()


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
    ),
}


def resolve_core(core_id: str) -> CoreRegistration:
    """Look up a registered core or fail loudly."""
    if core_id not in CORE_REGISTRY:
        raise KeyError(
            f"Unknown enhancement core {core_id!r}; registered cores: {sorted(CORE_REGISTRY)}"
        )
    return CORE_REGISTRY[core_id]
