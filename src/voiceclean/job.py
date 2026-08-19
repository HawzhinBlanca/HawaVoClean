"""Job workspace lifecycle, unit caching, and atomic publication."""

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from voiceclean.config import VoiceCleanConfig
from voiceclean.errors import PreflightError, PublicationError
from voiceclean.hashing import compute_job_id
from voiceclean.journal import JobJournal


class JobWorkspace:
    """Manages private directory structure, unit cache, and atomic output publication."""

    def __init__(
        self,
        input_path: Path,
        input_sha256: str,
        config: VoiceCleanConfig,
        core_id: str,
        guard_id: str,
        base_work_dir: Path | None = None,
        tool_version: str = "1.0.0",
    ) -> None:
        self.input_path = input_path.resolve()
        self.input_sha256 = input_sha256
        self.config = config
        self.config_hash = config.compute_hash()
        self.core_id = core_id
        self.guard_id = guard_id
        self.tool_version = tool_version

        self.job_id = compute_job_id(
            input_hash=self.input_sha256,
            config_hash=self.config_hash,
            core_hash=self.core_id,
            guard_hash=self.guard_id,
            tool_version=self.tool_version,
        )

        if base_work_dir is None:
            base_work_dir = Path(".voiceclean-work")

        self.root = (base_work_dir / self.job_id).resolve()
        self.units_dir = self.root / "units"
        self.cache_dir = self.root / "cache"
        self.reports_dir = self.root / "reports"
        self.journal_path = self.root / "journal.jsonl"
        self.job_meta_path = self.root / "job.json"

        self._init_workspace()
        self.journal = JobJournal(self.journal_path)

    def _init_workspace(self) -> None:
        """Create directories with restricted permissions (0o700)."""
        self.root.mkdir(parents=True, exist_ok=True)
        import contextlib

        with contextlib.suppress(Exception):
            self.root.chmod(0o700)

        self.units_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)

        if not self.job_meta_path.exists():
            meta = {
                "job_id": self.job_id,
                "input_path": str(self.input_path),
                "input_sha256": self.input_sha256,
                "config_hash": self.config_hash,
                "core_id": self.core_id,
                "guard_id": self.guard_id,
                "tool_version": self.tool_version,
            }
            with open(self.job_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

    def check_disk_space(self, required_bytes: int) -> None:
        """Verify sufficient free disk space before processing."""
        stat = shutil.disk_usage(self.root)
        safety_margin = 500 * 1024 * 1024  # 500 MB safety buffer
        if stat.free < (required_bytes + safety_margin):
            raise PreflightError(
                f"Insufficient disk space in {self.root}: available {stat.free / (1024 * 1024):.1f} MB, "
                f"required {(required_bytes + safety_margin) / (1024 * 1024):.1f} MB"
            )

    def save_unit_result(
        self,
        unit_id: int,
        channel_id: int,
        waveform: np.ndarray[Any, np.dtype[np.float32]],
    ) -> Path:
        """Persist committed unit waveform to private units directory."""
        unit_file = self.units_dir / f"unit_{unit_id:06d}_ch{channel_id}.npy"
        np.save(unit_file, waveform.astype(np.float32))
        return unit_file

    def load_unit_result(
        self,
        unit_id: int,
        channel_id: int,
    ) -> np.ndarray[Any, np.dtype[np.float32]] | None:
        """Load cached unit waveform if exists."""
        unit_file = self.units_dir / f"unit_{unit_id:06d}_ch{channel_id}.npy"
        if unit_file.exists():
            try:
                arr = np.load(unit_file)
                return np.ascontiguousarray(arr, dtype=np.float32)
            except Exception:
                return None
        return None

    def publish_atomically(
        self,
        temp_audio_path: Path,
        destination_audio_path: Path,
        json_report_str: str,
        txt_summary_str: str,
        overwrite: bool = False,
    ) -> tuple[Path, Path, Path]:
        """Atomically publish output audio, JSON report, and TXT summary."""
        dest_audio = Path(destination_audio_path).resolve()
        dest_audio.parent.mkdir(parents=True, exist_ok=True)

        dest_json = dest_audio.parent / f"{dest_audio.stem}.voiceclean.json"
        dest_txt = dest_audio.parent / f"{dest_audio.stem}.voiceclean.txt"

        if not overwrite and (dest_audio.exists() or dest_json.exists() or dest_txt.exists()):
            raise PublicationError(
                f"Destination output file already exists and overwrite=False: {dest_audio}"
            )

        if not temp_audio_path.exists():
            raise PublicationError(f"Temporary candidate audio file missing: {temp_audio_path}")

        # Write reports to workspace staging first
        tmp_json = self.reports_dir / "report.json.tmp"
        tmp_txt = self.reports_dir / "report.txt.tmp"

        with open(tmp_json, "w", encoding="utf-8") as f:
            f.write(json_report_str)
            f.flush()
            os.fsync(f.fileno())

        with open(tmp_txt, "w", encoding="utf-8") as f:
            f.write(txt_summary_str)
            f.flush()
            os.fsync(f.fileno())

        # Atomic replacements
        try:
            os.replace(temp_audio_path, dest_audio)
            os.replace(tmp_json, dest_json)
            os.replace(tmp_txt, dest_txt)
        except Exception as e:
            raise PublicationError(f"Atomic file publish failed: {e}") from e

        return dest_audio, dest_json, dest_txt

    def cleanup(self) -> None:
        """Remove workspace on clean completion if requested."""
        import contextlib

        with contextlib.suppress(Exception):
            shutil.rmtree(self.root)
