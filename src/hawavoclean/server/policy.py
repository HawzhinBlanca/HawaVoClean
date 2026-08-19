"""Request-side path policy for the engine API.

Every path a client sends (analyze, jobs, audio) must be absolute and must
resolve under the user's home directory, ``/Volumes`` (mounted media), or
the HawaVoClean work directory. Anything else is refused with 403 — the
engine never reads or writes arbitrary files on behalf of a web page.
"""

from pathlib import Path

from hawavoclean.paths import work_root


class PathPolicyError(Exception):
    """A client-supplied path was refused; ``status`` is the HTTP status."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def allowed_roots() -> list[Path]:
    """Roots a client path may resolve under (resolved, deduplicated)."""
    roots: list[Path] = []
    for candidate in (Path.home(), Path("/Volumes"), work_root()):
        try:
            resolved = candidate.resolve()
        except OSError:  # pragma: no cover - pathological filesystem state
            continue
        if resolved not in roots:
            roots.append(resolved)
    return roots


def resolve_client_path(raw: str, *, must_exist: bool = False) -> Path:
    """Validate ``raw`` against the policy and return its resolved ``Path``.

    Raises :class:`PathPolicyError` with 400 (not absolute), 403 (outside
    the allowed roots), or 404 (``must_exist`` and no such file).
    """
    if not raw or not raw.strip():
        raise PathPolicyError(400, "bad_request", "path is required")
    p = Path(raw)
    if not p.is_absolute():
        raise PathPolicyError(400, "bad_request", f"path must be absolute: {raw}")
    resolved = p.resolve()
    if not any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots()):
        raise PathPolicyError(
            403,
            "forbidden",
            f"path is outside the allowed locations (home, /Volumes, work dir): {raw}",
        )
    if must_exist and not resolved.is_file():
        raise PathPolicyError(404, "not_found", f"file not found: {raw}")
    return resolved
