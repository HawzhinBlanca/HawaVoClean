"""Resource path resolution independent of the current working directory.

Configs and model artifacts ship inside the package under
``hawavoclean/resources/``; the job workspace defaults to a per-user cache
directory. All three roots are overridable through environment variables so
tests and deployments can isolate state:

- ``HAWAVOCLEAN_CONFIG_DIR``: directory holding ``production.toml`` etc.
- ``HAWAVOCLEAN_MODEL_DIR``: directory holding lockfile and calibration.
- ``HAWAVOCLEAN_WORK_DIR``: root for per-job scratch workspaces.
"""

import os
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent


def config_dir() -> Path:
    """Directory containing the built-in profile TOML files."""
    override = os.environ.get("HAWAVOCLEAN_CONFIG_DIR")
    if override:
        return Path(override).resolve()
    return _PACKAGE_ROOT / "resources" / "configs"


def models_dir() -> Path:
    """Directory containing the core lockfile and guard calibration artifact."""
    override = os.environ.get("HAWAVOCLEAN_MODEL_DIR")
    if override:
        return Path(override).resolve()
    return _PACKAGE_ROOT / "resources" / "models"


def restoration_checkpoint_path() -> Path:
    """Path of the HawaRestore-KD checkpoint.

    Resolution must never depend on the working directory: a relative lookup
    silently misses when the CLI is run from the user's audio folder, and the
    restorer would then fall back to untrained weights while the report still
    attests a checkpoint. The env override wins, then the packaged models
    directory, then the in-repo ``models/`` tree used by source checkouts.
    """
    override = os.environ.get("HAWAVOCLEAN_RESTORATION_CHECKPOINT")
    if override:
        return Path(override).resolve()
    packaged = models_dir() / "hawarestore-kd" / "hawarestore_kd.pt"
    if packaged.is_file():
        return packaged
    return _PACKAGE_ROOT.parents[1] / "models" / "hawarestore-kd" / "hawarestore_kd.pt"


def work_root() -> Path:
    """Root directory for per-job scratch workspaces."""
    override = os.environ.get("HAWAVOCLEAN_WORK_DIR")
    if override:
        return Path(override).resolve()
    return Path.home() / ".cache" / "hawavoclean" / "work"


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
