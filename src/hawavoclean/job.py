"""Job workspace lifecycle and atomic output publication.

The workspace is scratch space for exactly one run: it is created under
``hawavoclean.paths.work_root()``, holds the journal and staging files, and
is removed on successful publication. There is no resume cache — a repeated
run recomputes every unit, so the audit report always describes the run
that produced it. On a crash the workspace survives for forensics only.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

from hawavoclean.config import HawaVoCleanConfig
from hawavoclean.errors import PreflightError, PublicationError
from hawavoclean.hashing import compute_job_id
from hawavoclean.journal import JobJournal
from hawavoclean.paths import work_root


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
            base_work_dir = work_root()

        # A fresh scratch directory per run: the job_id names the job, the
        # unique suffix guarantees no state is shared between runs.
        base_work_dir.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix=f"{self.job_id}-", dir=base_work_dir)).resolve()
        self.journal_path = self.root / "journal.jsonl"
        self.job_meta_path = self.root / "job.json"

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
    ) -> tuple[Path, Path, Path]:
        """Atomically publish output audio, JSON report, and TXT summary.

        All three artifacts are staged in a temporary directory ON THE
        DESTINATION FILESYSTEM, so the final renames are always intra-device
        (no EXDEV) and effectively atomic. If any rename fails, the ones that
        already happened are rolled back and nothing partial is left behind.

        A staticmethod on purpose: it touches no workspace state, and the
        multi-pass orchestrator publishes its amended final report through
        this exact code path rather than a second implementation of the
        atomic-publish discipline.
        """
        dest_audio = Path(destination_audio_path).resolve()
        dest_audio.parent.mkdir(parents=True, exist_ok=True)

        dest_json = dest_audio.parent / f"{dest_audio.stem}.hawavoclean.json"
        dest_txt = dest_audio.parent / f"{dest_audio.stem}.hawavoclean.txt"

        if not overwrite and (dest_audio.exists() or dest_json.exists() or dest_txt.exists()):
            raise PublicationError(
                f"Destination output file already exists and overwrite=False: {dest_audio}"
            )

        if not temp_audio_path.exists():
            raise PublicationError(f"Temporary candidate audio file missing: {temp_audio_path}")

        staging = Path(tempfile.mkdtemp(prefix=".hawavoclean-publish-", dir=dest_audio.parent))
        try:
            staged_audio = staging / dest_audio.name
            staged_json = staging / dest_json.name
            staged_txt = staging / dest_txt.name

            shutil.copyfile(temp_audio_path, staged_audio)
            for staged, content in ((staged_json, json_report_str), (staged_txt, txt_summary_str)):
                with open(staged, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())

            renamed: list[tuple[Path, Path]] = []
            try:
                for staged, dest in (
                    (staged_audio, dest_audio),
                    (staged_json, dest_json),
                    (staged_txt, dest_txt),
                ):
                    os.replace(staged, dest)
                    renamed.append((staged, dest))
            except BaseException as e:
                # BaseException, not Exception: a Ctrl-C (or the SIGTERM the
                # CLI turns into one, or the parent-death watchdog's SIGINT)
                # can land between two of these renames, and `except
                # Exception` would let KeyboardInterrupt out of here with one
                # or two of the three artifacts already at the destination —
                # a master with no report beside it, which is exactly the
                # partial publication this method exists to prevent.
                for _, dest in renamed:
                    import contextlib

                    with contextlib.suppress(Exception):
                        dest.unlink()
                if isinstance(e, Exception):
                    raise PublicationError(f"Atomic file publish failed: {e}") from e
                raise  # an interrupt stays an interrupt: the run was cancelled
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        return dest_audio, dest_json, dest_txt

    def cleanup(self) -> None:
        """Remove the scratch workspace. Called on successful completion."""
        import contextlib

        with contextlib.suppress(Exception):
            shutil.rmtree(self.root)
