"""Resource path resolution independent of the current working directory.

Configs and model artifacts ship inside the package under
``voiceclean/resources/``; the job workspace defaults to a per-user cache
directory. All three roots are overridable through environment variables so
tests and deployments can isolate state:

- ``VOICECLEAN_CONFIG_DIR``: directory holding ``production.toml`` etc.
- ``VOICECLEAN_MODEL_DIR``: directory holding lockfile and calibration.
- ``VOICECLEAN_WORK_DIR``: root for per-job scratch workspaces.
"""

import os
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent


def config_dir() -> Path:
    """Directory containing the built-in profile TOML files."""
    override = os.environ.get("VOICECLEAN_CONFIG_DIR")
    if override:
        return Path(override).resolve()
    return _PACKAGE_ROOT / "resources" / "configs"


def models_dir() -> Path:
    """Directory containing the core lockfile and guard calibration artifact."""
    override = os.environ.get("VOICECLEAN_MODEL_DIR")
    if override:
        return Path(override).resolve()
    return _PACKAGE_ROOT / "resources" / "models"


def work_root() -> Path:
    """Root directory for per-job scratch workspaces."""
    override = os.environ.get("VOICECLEAN_WORK_DIR")
    if override:
        return Path(override).resolve()
    return Path.home() / ".cache" / "voiceclean" / "work"


def profile_config_path(profile: str) -> Path:
    """Path of the TOML config for a named profile."""
    return config_dir() / f"{profile}.toml"


def resolve_calibration_file(configured: str) -> Path:
    """Resolve the guard calibration artifact path.

    Absolute paths are honored as-is; relative paths resolve against the
    models directory.
    """
    p = Path(configured)
    if p.is_absolute():
        return p
    return models_dir() / p
