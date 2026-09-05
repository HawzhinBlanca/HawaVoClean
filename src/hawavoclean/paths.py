"""Resource path resolution independent of the current working directory.

Configs and model artifacts ship inside the package under
``hawavoclean/resources/``; the job workspace defaults to a per-user cache
directory. All three roots are overridable through environment variables so
tests and deployments can isolate state:

- ``HAWAVOCLEAN_CONFIG_DIR``: directory holding ``production.toml`` etc.
- ``HAWAVOCLEAN_MODEL_DIR``: directory holding lockfile and calibration.
- ``HAWAVOCLEAN_PROFILES_DIR``: directory holding per-speaker restore profiles.
- ``HAWAVOCLEAN_WORK_DIR``: root for per-job scratch workspaces.
"""

import os
import shutil
import sys
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


def profiles_root() -> Path:
    """Root directory holding per-speaker restoration profiles.

    Resolution must never depend on the working directory: a relative lookup
    silently misses when the engine is launched from the user's audio folder,
    and restore jobs would then fail preflight on a machine that has every
    profile installed. The env override wins, then the in-repo ``profiles/``
    tree used by source checkouts.
    """
    override = os.environ.get("HAWAVOCLEAN_PROFILES_DIR")
    if override:
        return Path(override).resolve()
    return _PACKAGE_ROOT.parents[1] / "profiles"


def work_root() -> Path:
    """Root directory for per-job scratch workspaces."""
    override = os.environ.get("HAWAVOCLEAN_WORK_DIR")
    if override:
        return Path(override).resolve()
    return Path.home() / ".cache" / "hawavoclean" / "work"


def app_data_root() -> Path:
    """Durable per-user application data, never the disposable work cache."""

    override = os.environ.get("HAWAVOCLEAN_STATE_DIR")
    if override:
        return Path(override).resolve()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "HawaVoClean"
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return base / "HawaVoClean"
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "hawavoclean"


def job_store_path() -> Path:
    """SQLite ledger used by the installed local engine broker."""

    return app_data_root() / "state" / "jobs.sqlite3"


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


def ffmpeg_bin_path() -> str | None:
    """Resolve the pinned or bundled FFmpeg binary path.

    Resolution order:
    1. Explicit environment variable: HAWAVOCLEAN_FFMPEG_PATH
    2. Pinned bundled binary inside package resources:
       _PACKAGE_ROOT / "resources" / "bin" / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    3. Pinned bundled binary alongside python prefix or engine root:
       Path(sys.prefix) / "bin" / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
       _PACKAGE_ROOT.parents[1] / "bin" / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    4. Host system PATH via shutil.which("ffmpeg")
    """
    env_override = os.environ.get("HAWAVOCLEAN_FFMPEG_PATH")
    if env_override:
        p = Path(env_override).resolve()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)

    exe_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"

    pkg_bin = _PACKAGE_ROOT / "resources" / "bin" / exe_name
    if pkg_bin.is_file() and os.access(pkg_bin, os.X_OK):
        return str(pkg_bin)

    prefix_bin = Path(sys.prefix) / "bin" / exe_name
    if prefix_bin.is_file() and os.access(prefix_bin, os.X_OK):
        return str(prefix_bin)

    engine_bin = _PACKAGE_ROOT.parents[1] / "bin" / exe_name
    if engine_bin.is_file() and os.access(engine_bin, os.X_OK):
        return str(engine_bin)

    return shutil.which("ffmpeg")


def ffprobe_bin_path() -> str | None:
    """Resolve the pinned or bundled ffprobe binary path.

    Resolution order:
    1. Explicit environment variable: HAWAVOCLEAN_FFPROBE_PATH
    2. Pinned bundled binary inside package resources:
       _PACKAGE_ROOT / "resources" / "bin" / ("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
    3. Pinned bundled binary alongside python prefix or engine root:
       Path(sys.prefix) / "bin" / ("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
       _PACKAGE_ROOT.parents[1] / "bin" / ("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
    4. Host system PATH via shutil.which("ffprobe")
    """
    env_override = os.environ.get("HAWAVOCLEAN_FFPROBE_PATH")
    if env_override:
        p = Path(env_override).resolve()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)

    exe_name = "ffprobe.exe" if sys.platform == "win32" else "ffprobe"

    pkg_bin = _PACKAGE_ROOT / "resources" / "bin" / exe_name
    if pkg_bin.is_file() and os.access(pkg_bin, os.X_OK):
        return str(pkg_bin)

    prefix_bin = Path(sys.prefix) / "bin" / exe_name
    if prefix_bin.is_file() and os.access(prefix_bin, os.X_OK):
        return str(prefix_bin)

    engine_bin = _PACKAGE_ROOT.parents[1] / "bin" / exe_name
    if engine_bin.is_file() and os.access(engine_bin, os.X_OK):
        return str(engine_bin)

    return shutil.which("ffprobe")
