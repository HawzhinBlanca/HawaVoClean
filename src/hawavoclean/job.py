"""Job workspace lifecycle and atomic output publication.

The workspace is scratch space for exactly one run: it is created under
``hawavoclean.paths.work_root()``, holds the journal and staging files, and
is removed on successful publication. There is no resume cache — a repeated
run recomputes every unit, so the audit report always describes the run
that produced it. On a crash the workspace survives for forensics only.
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from hawavoclean.config import HawaVoCleanConfig
from hawavoclean.errors import PreflightError
from hawavoclean.hashing import compute_job_id
from hawavoclean.journal import JobJournal
from hawavoclean.paths import work_root
from hawavoclean.publication import publish_output_generation


class JobWorkspace:
    """Manages the private scratch directory, journal, and atomic publication."""

    def __init__(
        self,
        input_path: Path,
        input_sha256: str,
        config: HawaVoCleanConfig,
        core_id: str,
        guard_id: str,
        base_work_dir: Path | None = None,
        tool_version: str = "1.0.0",
        restore_context: str | None = None,
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
            restore_context=restore_context,
        )

        if base_work_dir is None:
            base_work_dir = work_root()

        # A fresh scratch directory per run: the job_id names the job, the
        # unique suffix guarantees no state is shared between runs.
        base_work_dir.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix=f"{self.job_id}-", dir=base_work_dir)).resolve()
        self.journal_path = self.root / "journal.jsonl"
        self.job_meta_path = self.root / "job.json"
        # Pipeline-owned resources that must be closed before Windows can
        # remove the scratch directory. Kept generic so JobWorkspace does not
        # import numpy or own DSP lifecycle policy.
        self.pipeline_disk_mappings: list[Any] = []

        self._init_workspace()
        self.journal = JobJournal(self.journal_path)

    def _init_workspace(self) -> None:
        """Restrict permissions and record job metadata."""
        import contextlib

        with contextlib.suppress(Exception):
            self.root.chmod(0o700)

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

    def check_disk_space(self, required_bytes: int, destination: Path | None = None) -> None:
        """Verify sufficient free space in the workspace and at the destination."""
        safety_margin = 500 * 1024 * 1024  # 500 MB safety buffer
        targets = [self.root]
        if destination is not None:
            dest_dir = destination if destination.is_dir() else destination.parent
            if dest_dir.exists():
                targets.append(dest_dir)
        for target in targets:
            stat = shutil.disk_usage(target)
            if stat.free < (required_bytes + safety_margin):
                raise PreflightError(
                    f"Insufficient disk space at {target}: available "
                    f"{stat.free / (1024 * 1024):.1f} MB, required "
                    f"{(required_bytes + safety_margin) / (1024 * 1024):.1f} MB"
                )

    @staticmethod
    def publish_atomically(
        temp_audio_path: Path,
        destination_audio_path: Path,
        json_report_str: str,
        txt_summary_str: str,
        overwrite: bool = False,
        clean_only: bool = False,
    ) -> tuple[Path, Path, Path]:
        """Publish output audio, JSON report, and TXT as one committed generation.

        The familiar public paths are stable relative aliases through one
        adjacent ``current`` pointer. Replacing that single pointer commits all
        three artifacts together; immutable prior generations remain available
        for recovery. See ADR 0005.

        If clean_only is True, emits only the destination .wav master file without
        creating public sidecars or hidden bundle directories.
        """
        return publish_output_generation(
            temp_audio_path=temp_audio_path,
            destination_audio_path=destination_audio_path,
            json_report_str=json_report_str,
            txt_summary_str=txt_summary_str,
            overwrite=overwrite,
            clean_only=clean_only,
        )

    def cleanup(self) -> None:
        """Remove the scratch workspace. Called on successful completion."""
        import contextlib

        from hawavoclean.source_pin import remove_source_snapshot_tree

        remove_source_snapshot_tree(self.root / "source-snapshot")
        with contextlib.suppress(Exception):
            shutil.rmtree(self.root)
