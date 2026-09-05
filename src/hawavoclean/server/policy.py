"""Request-side path policy for the engine API.

Every path a client sends (analyze, jobs, audio) must be absolute and must
resolve under the user's home directory, ``/Volumes`` (mounted media), or
the HawaVoClean work directory. Anything else is refused with 403 — the
engine never reads or writes arbitrary files on behalf of a web page.

Refusals are *designed*: this module answers 400, 403 or 404 and never lets
a filesystem call raise through it. That matters because a client string is
not a path — ``"/x/a\\0b.wav"`` and a lone surrogate both make ``lstat``
raise ``ValueError`` deep inside ``Path.resolve()``, which the API would
otherwise publish as a 500 with a raw Python exception in the body. Text
that a POSIX filesystem *can* hold — newline, tab, ESC, DEL, non-UTF-8
bytes carried as surrogate escapes — is legal in a filename and is passed
through to the ordinary 403/404 answers.
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


def refuse_unusable_filename_text(raw: str, *, what: str = "path") -> None:
    """Raise 400 if ``raw`` is text no filesystem call can even accept.

    Two cases, both of which make ``os``/``pathlib`` raise ``ValueError``
    rather than return a refusal:

    * a NUL byte — it terminates the C string, so it can never be part of a
      name (and silently truncating it would open a policy bypass);
    * text that does not survive ``os.fsencode`` — an unpaired surrogate,
      which ``json.loads`` produces from a ``"\\ud800"`` escape and which
      the UTF-8 codec refuses. (Surrogate *escapes* in the U+DC80–DCFF
      range do encode: they are how Python carries non-UTF-8 filenames, and
      they stay legal here.)
    """
    nul = raw.find("\x00")
    if nul >= 0:
        raise PathPolicyError(400, "bad_request", f"{what} contains a NUL byte at position {nul}")
    try:
        raw.encode("utf-8", "surrogateescape")
    except (UnicodeEncodeError, ValueError) as e:
        detail = getattr(e, "reason", None) or str(e)
        raise PathPolicyError(
            400, "bad_request", f"{what} is not a usable filename: {detail}"
        ) from e


def resolve_client_path(raw: str, *, must_exist: bool = False) -> Path:
    """Validate ``raw`` against the policy and return its resolved ``Path``.

    Raises :class:`PathPolicyError` with 400 (empty, not absolute, or text
    that cannot be a filename at all), 403 (outside the allowed roots), or
    404 (``must_exist`` and no such file). It never raises anything else.
    """
    if not raw or not raw.strip():
        raise PathPolicyError(400, "bad_request", "path is required")
    refuse_unusable_filename_text(raw)
    p = Path(raw)
    if not p.is_absolute():
        raise PathPolicyError(400, "bad_request", f"path must be absolute: {raw}")
    try:
        resolved = p.resolve()
    except (OSError, ValueError) as e:  # pragma: no cover - defence in depth
        raise PathPolicyError(400, "bad_request", f"path cannot be resolved: {e}") from e
    if not any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots()):
        raise PathPolicyError(
            403,
            "forbidden",
            f"path is outside the allowed locations (home, /Volumes, work dir): {raw}",
        )
    if must_exist:
        try:
            exists = resolved.is_file()
        except (OSError, ValueError):  # pragma: no cover - defence in depth
            exists = False
        if not exists:
            raise PathPolicyError(404, "not_found", f"file not found: {raw}")
    return resolved


def resolve_client_output_path(raw: str) -> Path:
    """Validate a writable public path without following its final alias.

    A committed HawaVoClean output is a symlink into its adjacent generation
    bundle. Following that final symlink during an overwrite would silently
    retarget the job at the immutable old generation. The parent is still
    fully resolved and checked against policy; only the final public name is
    preserved lexically for the publisher.
    """
    if not raw or not raw.strip():
        raise PathPolicyError(400, "bad_request", "path is required")
    refuse_unusable_filename_text(raw)
    path = Path(raw)
    if not path.is_absolute():
        raise PathPolicyError(400, "bad_request", f"path must be absolute: {raw}")
    try:
        parent = path.parent.resolve()
    except (OSError, ValueError) as exc:  # pragma: no cover - defence in depth
        raise PathPolicyError(400, "bad_request", f"path cannot be resolved: {exc}") from exc
    if not any(parent == root or parent.is_relative_to(root) for root in allowed_roots()):
        raise PathPolicyError(
            403,
            "forbidden",
            f"path is outside the allowed locations (home, /Volumes, work dir): {raw}",
        )
    return parent / path.name
