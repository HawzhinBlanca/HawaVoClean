"""Canonical release identity shared by every shipped surface.

``release.json`` is the sole authored version source. Packaging manifests are
generated mirrors checked by ``scripts/sync_release_identity.py --check``;
runtime and report identities are read directly from the packaged JSON bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Final, Literal, TypedDict, cast

_EXPECTED_FIELDS = {
    "identity_schema_version",
    "product",
    "version",
    "report_schema_version",
}
_SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


class ReleaseIdentityError(RuntimeError):
    """The packaged release identity is missing or internally inconsistent."""


class ReleaseReportFields(TypedDict):
    """Fields copied verbatim into each current audit report."""

    product: str
    version: str
    report_schema_version: int
    identity_sha256: str


@dataclass(frozen=True, slots=True)
class ReleaseIdentity:
    """Validated identity of this HawaVoClean distribution."""

    identity_schema_version: int
    product: str
    version: str
    report_schema_version: int
    identity_sha256: str

    def report_fields(self) -> ReleaseReportFields:
        """Return the exact identity fields embedded in schema-v2 reports."""
        return {
            "product": self.product,
            "version": self.version,
            "report_schema_version": self.report_schema_version,
            "identity_sha256": self.identity_sha256,
        }


def _validated_identity(raw_bytes: bytes) -> ReleaseIdentity:
    try:
        value: Any = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseIdentityError(f"release.json is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != _EXPECTED_FIELDS:
        raise ReleaseIdentityError(
            f"release.json fields must be exactly {sorted(_EXPECTED_FIELDS)}"
        )
    if value["identity_schema_version"] != 1:
        raise ReleaseIdentityError("unsupported release identity schema")
    if value["product"] != "hawavoclean":
        raise ReleaseIdentityError("release identity names the wrong product")
    version = value["version"]
    if not isinstance(version, str) or _SEMVER.fullmatch(version) is None:
        raise ReleaseIdentityError("release version must be canonical MAJOR.MINOR.PATCH SemVer")
    report_schema = value["report_schema_version"]
    if report_schema != 2:
        raise ReleaseIdentityError("this build supports report_schema_version 2")
    return ReleaseIdentity(
        identity_schema_version=1,
        product="hawavoclean",
        version=version,
        report_schema_version=cast(Literal[2], report_schema),
        identity_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def load_release_identity() -> ReleaseIdentity:
    """Load and validate the identity bytes included in this package."""
    resource = files("hawavoclean").joinpath("release.json")
    try:
        raw_bytes = resource.read_bytes()
    except OSError as exc:
        raise ReleaseIdentityError(f"cannot read packaged release identity: {exc}") from exc
    return _validated_identity(raw_bytes)


RELEASE_IDENTITY = load_release_identity()
VERSION: Final[str] = RELEASE_IDENTITY.version
REPORT_SCHEMA_VERSION: Final[Literal[2]] = cast(Literal[2], RELEASE_IDENTITY.report_schema_version)
