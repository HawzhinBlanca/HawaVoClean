"""Strict, canonical schema for HawaVoClean Restore model packs.

The manifest is deliberately small and closed: unknown fields, duplicate JSON
keys, non-canonical JSON, ambiguous paths, and non-canonical versions or
timestamps are rejected.  Ed25519 signs the exact canonical bytes returned by
``canonical_manifest_bytes``; the signature itself lives in ``manifest.sig``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any, Final, Literal, cast

from hawavoclean.model_packs.errors import ModelPackManifestError

MANIFEST_FILENAME: Final = "manifest.json"
SIGNATURE_FILENAME: Final = "manifest.sig"
MANIFEST_SCHEMA_VERSION: Final = 1
PACK_PRODUCT: Final = "hawavoclean-restore"
MAX_MANIFEST_BYTES: Final = 1024 * 1024
MAX_PAYLOADS: Final = 256

CoreRole = Literal["model", "verifier", "preprocessing", "corpus", "runtime"]
AssetRole = Literal["license", "metadata", "auxiliary"]
PayloadRole = CoreRole | AssetRole
QualityTier = Literal["research", "candidate", "production"]
Maturity = Literal["experimental", "qualified", "blocked"]

_CORE_ROLES: Final[tuple[CoreRole, ...]] = (
    "model",
    "verifier",
    "preprocessing",
    "corpus",
    "runtime",
)
_ASSET_ROLES: Final[frozenset[str]] = frozenset({"license", "metadata", "auxiliary"})
_QUALITY_TIERS: Final[frozenset[str]] = frozenset({"research", "candidate", "production"})
_MATURITIES: Final[frozenset[str]] = frozenset({"experimental", "qualified", "blocked"})
_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$")
_KEY_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._:-]{0,126}[A-Za-z0-9])?$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SEMVER_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_UTC_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_WINDOWS_RESERVED: Final[frozenset[str]] = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)
_WINDOWS_FORBIDDEN = frozenset('<>:"|?*')


@dataclass(frozen=True, order=True, slots=True)
class SemanticVersion:
    """Canonical three-component SemVer used by packs and runtime ranges."""

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: object, *, field: str) -> SemanticVersion:
        if not isinstance(value, str):
            raise _manifest_error(f"{field} must be a string", "invalid_version")
        if len(value) > 32:
            raise _manifest_error(f"{field} is unreasonably long", "invalid_version")
        match = _SEMVER_RE.fullmatch(value)
        if match is None:
            raise _manifest_error(
                f"{field} must be canonical MAJOR.MINOR.PATCH SemVer",
                "invalid_version",
            )
        try:
            return cls(*(int(part) for part in match.groups()))
        except ValueError as exc:
            raise _manifest_error(f"{field} is not a supported version", "invalid_version") from exc

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True, slots=True)
class RuntimeCompatibility:
    """Inclusive lower and exclusive upper supported HawaVoClean versions."""

    min_version: str
    max_version_exclusive: str

    def supports(self, version: str) -> bool:
        candidate = SemanticVersion.parse(version, field="runtime version")
        return self.minimum <= candidate < self.maximum_exclusive

    @property
    def minimum(self) -> SemanticVersion:
        return SemanticVersion.parse(self.min_version, field="runtime_compatibility.min_version")

    @property
    def maximum_exclusive(self) -> SemanticVersion:
        return SemanticVersion.parse(
            self.max_version_exclusive,
            field="runtime_compatibility.max_version_exclusive",
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "min_version": self.min_version,
            "max_version_exclusive": self.max_version_exclusive,
        }


@dataclass(frozen=True, slots=True)
class PackPayload:
    """One regular file whose bytes, path, and size are signed by the manifest."""

    role: PayloadRole
    path: str
    sha256: str
    size_bytes: int

    def to_record(self, *, include_role: bool) -> dict[str, object]:
        record: dict[str, object] = {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }
        if include_role:
            record["role"] = self.role
        return record


@dataclass(frozen=True, slots=True)
class ModelPackManifest:
    """Validated v1 Restore model-pack manifest."""

    schema_version: int
    product: str
    pack_id: str
    version: str
    issued_at: str
    not_before: str
    expires_at: str
    signing_key_id: str
    quality_tier: QualityTier
    maturity: Maturity
    runtime_compatibility: RuntimeCompatibility
    components: tuple[PackPayload, ...]
    assets: tuple[PackPayload, ...]

    @property
    def semantic_version(self) -> SemanticVersion:
        return SemanticVersion.parse(self.version, field="version")

    @property
    def issued_datetime(self) -> datetime:
        return _parse_utc(self.issued_at, field="issued_at")

    @property
    def not_before_datetime(self) -> datetime:
        return _parse_utc(self.not_before, field="not_before")

    @property
    def expires_datetime(self) -> datetime:
        return _parse_utc(self.expires_at, field="expires_at")

    @property
    def payloads(self) -> tuple[PackPayload, ...]:
        return self.components + self.assets

    def component(self, role: CoreRole) -> PackPayload:
        for payload in self.components:
            if payload.role == role:
                return payload
        raise AssertionError(f"validated manifest is missing {role}")

    @property
    def model_sha256(self) -> str:
        return self.component("model").sha256

    @property
    def verifier_sha256(self) -> str:
        return self.component("verifier").sha256

    @property
    def preprocessing_sha256(self) -> str:
        return self.component("preprocessing").sha256

    @property
    def corpus_sha256(self) -> str:
        return self.component("corpus").sha256

    @property
    def runtime_sha256(self) -> str:
        return self.component("runtime").sha256

    def to_dict(self) -> dict[str, object]:
        component_map = {
            role: self.component(role).to_record(include_role=False) for role in _CORE_ROLES
        }
        return {
            "schema_version": self.schema_version,
            "product": self.product,
            "pack_id": self.pack_id,
            "version": self.version,
            "issued_at": self.issued_at,
            "not_before": self.not_before,
            "expires_at": self.expires_at,
            "signing_key_id": self.signing_key_id,
            "quality_tier": self.quality_tier,
            "maturity": self.maturity,
            "runtime_compatibility": self.runtime_compatibility.to_dict(),
            "components": component_map,
            "assets": [asset.to_record(include_role=True) for asset in self.assets],
        }


def canonical_json_bytes(value: object) -> bytes:
    """Encode signed JSON in its one accepted UTF-8 representation."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_manifest_bytes(manifest: ModelPackManifest) -> bytes:
    """Return the exact bytes that must be stored and signed."""
    return canonical_json_bytes(manifest.to_dict())


def parse_manifest_bytes(raw: bytes, *, require_canonical: bool = True) -> ModelPackManifest:
    """Parse a manifest with a closed schema and no ambiguous JSON constructs."""
    if len(raw) > MAX_MANIFEST_BYTES:
        raise _manifest_error("manifest exceeds the 1 MiB safety limit", "manifest_too_large")
    value = _load_strict_json(raw, subject="manifest")
    root = _expect_mapping(value, field="manifest")
    _expect_fields(
        root,
        {
            "schema_version",
            "product",
            "pack_id",
            "version",
            "issued_at",
            "not_before",
            "expires_at",
            "signing_key_id",
            "quality_tier",
            "maturity",
            "runtime_compatibility",
            "components",
            "assets",
        },
        field="manifest",
    )
    schema_version = root["schema_version"]
    if type(schema_version) is not int or schema_version != MANIFEST_SCHEMA_VERSION:
        raise _manifest_error(
            f"unsupported model-pack schema version: {schema_version!r}",
            "unsupported_schema",
        )
    if root["product"] != PACK_PRODUCT:
        raise _manifest_error("manifest names the wrong product", "wrong_product")

    pack_id = _validated_id(root["pack_id"], field="pack_id", pattern=_ID_RE)
    version = str(SemanticVersion.parse(root["version"], field="version"))
    issued_at = _validated_utc(root["issued_at"], field="issued_at")
    not_before = _validated_utc(root["not_before"], field="not_before")
    expires_at = _validated_utc(root["expires_at"], field="expires_at")
    if _parse_utc(issued_at, field="issued_at") > _parse_utc(not_before, field="not_before"):
        raise _manifest_error("issued_at must not be after not_before", "invalid_validity_window")
    if _parse_utc(not_before, field="not_before") >= _parse_utc(expires_at, field="expires_at"):
        raise _manifest_error(
            "not_before must be earlier than expires_at",
            "invalid_validity_window",
        )

    signing_key_id = _validated_id(
        root["signing_key_id"], field="signing_key_id", pattern=_KEY_ID_RE
    )
    quality = root["quality_tier"]
    if not isinstance(quality, str) or quality not in _QUALITY_TIERS:
        raise _manifest_error("unsupported quality_tier", "invalid_quality_tier")
    maturity = root["maturity"]
    if not isinstance(maturity, str) or maturity not in _MATURITIES:
        raise _manifest_error("unsupported maturity", "invalid_maturity")
    if maturity == "qualified" and quality != "production":
        raise _manifest_error(
            "qualified packs must use the production quality tier",
            "inconsistent_qualification",
        )
    if quality == "production" and maturity not in {"qualified", "blocked"}:
        raise _manifest_error(
            "production-tier packs must be qualified or blocked",
            "inconsistent_qualification",
        )

    compatibility_data = _expect_mapping(
        root["runtime_compatibility"], field="runtime_compatibility"
    )
    _expect_fields(
        compatibility_data,
        {"min_version", "max_version_exclusive"},
        field="runtime_compatibility",
    )
    compatibility = RuntimeCompatibility(
        min_version=str(
            SemanticVersion.parse(
                compatibility_data["min_version"],
                field="runtime_compatibility.min_version",
            )
        ),
        max_version_exclusive=str(
            SemanticVersion.parse(
                compatibility_data["max_version_exclusive"],
                field="runtime_compatibility.max_version_exclusive",
            )
        ),
    )
    if compatibility.minimum >= compatibility.maximum_exclusive:
        raise _manifest_error(
            "runtime compatibility range is empty",
            "invalid_runtime_range",
        )

    component_data = _expect_mapping(root["components"], field="components")
    _expect_fields(component_data, set(_CORE_ROLES), field="components")
    components = tuple(
        _parse_payload(component_data[role], role=role, include_role=False) for role in _CORE_ROLES
    )

    asset_data = root["assets"]
    if not isinstance(asset_data, list):
        raise _manifest_error("assets must be an array", "invalid_assets")
    assets: list[PackPayload] = []
    for index, raw_asset in enumerate(asset_data):
        asset_map = _expect_mapping(raw_asset, field=f"assets[{index}]")
        role_value = asset_map.get("role")
        if not isinstance(role_value, str) or role_value not in _ASSET_ROLES:
            raise _manifest_error(
                f"assets[{index}].role is unsupported",
                "invalid_payload_role",
            )
        assets.append(
            _parse_payload(
                asset_map,
                role=cast(AssetRole, role_value),
                include_role=True,
            )
        )
    if not any(asset.role == "license" for asset in assets):
        raise _manifest_error(
            "every model pack must bind at least one license asset",
            "missing_license_asset",
        )
    payloads = components + tuple(assets)
    if len(payloads) > MAX_PAYLOADS:
        raise _manifest_error(
            f"manifest declares more than {MAX_PAYLOADS} payloads",
            "too_many_payloads",
        )
    paths = [payload.path for payload in payloads]
    if len(set(paths)) != len(paths):
        raise _manifest_error("payload paths must be unique", "duplicate_payload_path")
    folded_paths = [path.casefold() for path in paths]
    if len(set(folded_paths)) != len(folded_paths):
        raise _manifest_error(
            "payload paths must remain unique on case-insensitive filesystems",
            "duplicate_payload_path",
        )
    path_parts = [tuple(part.casefold() for part in PurePosixPath(path).parts) for path in paths]
    for left_index, left in enumerate(path_parts):
        for right_index, right in enumerate(path_parts):
            if left_index != right_index and len(left) < len(right) and right[: len(left)] == left:
                raise _manifest_error(
                    "one payload path cannot be the parent of another",
                    "conflicting_payload_path",
                )

    manifest = ModelPackManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        product=PACK_PRODUCT,
        pack_id=pack_id,
        version=version,
        issued_at=issued_at,
        not_before=not_before,
        expires_at=expires_at,
        signing_key_id=signing_key_id,
        quality_tier=cast(QualityTier, quality),
        maturity=cast(Maturity, maturity),
        runtime_compatibility=compatibility,
        components=components,
        assets=tuple(assets),
    )
    if require_canonical and raw != canonical_manifest_bytes(manifest):
        raise _manifest_error(
            "manifest.json is not canonical JSON",
            "noncanonical_manifest",
        )
    return manifest


def _parse_payload(value: object, *, role: PayloadRole, include_role: bool) -> PackPayload:
    mapping = _expect_mapping(value, field=f"payload[{role}]")
    expected = {"path", "sha256", "size_bytes"}
    if include_role:
        expected.add("role")
    _expect_fields(mapping, expected, field=f"payload[{role}]")
    if include_role and mapping["role"] != role:
        raise _manifest_error(f"payload[{role}] role mismatch", "invalid_payload_role")
    path = _validated_payload_path(mapping["path"])
    sha256 = mapping["sha256"]
    if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
        raise _manifest_error(f"payload[{role}] has invalid SHA-256", "invalid_payload_hash")
    size = mapping["size_bytes"]
    if type(size) is not int or size <= 0:
        raise _manifest_error(
            f"payload[{role}] size_bytes must be a positive integer",
            "invalid_payload_size",
        )
    return PackPayload(role=role, path=path, sha256=sha256, size_bytes=size)


def _validated_payload_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise _manifest_error("payload path is empty or too long", "unsafe_payload_path")
    try:
        encoded_path = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _manifest_error("payload path is not valid Unicode", "unsafe_payload_path") from exc
    if len(encoded_path) > 512:
        raise _manifest_error("payload path is empty or too long", "unsafe_payload_path")
    if unicodedata.normalize("NFC", value) != value:
        raise _manifest_error("payload path must use Unicode NFC", "unsafe_payload_path")
    if "\\" in value or "\x00" in value or any(ord(character) < 32 for character in value):
        raise _manifest_error("payload path contains unsafe characters", "unsafe_payload_path")
    path = PurePosixPath(value)
    if path.is_absolute() or value.startswith("/"):
        raise _manifest_error("payload path must be relative", "unsafe_payload_path")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise _manifest_error("payload path must be normalized", "unsafe_payload_path")
    for part in path.parts:
        if any(character in _WINDOWS_FORBIDDEN for character in part):
            raise _manifest_error(
                "payload path is not portable to Windows",
                "unsafe_payload_path",
            )
        if part.endswith((" ", ".")) or _windows_reserved_name(part):
            raise _manifest_error(
                "payload path uses a Windows-reserved component",
                "unsafe_payload_path",
            )
    if value.casefold() in {MANIFEST_FILENAME.casefold(), SIGNATURE_FILENAME.casefold()}:
        raise _manifest_error("payload path collides with pack metadata", "unsafe_payload_path")
    return value


def _validated_id(value: object, *, field: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise _manifest_error(f"{field} has an invalid format", "invalid_identifier")
    if field == "pack_id" and _windows_reserved_name(value):
        raise _manifest_error("pack_id is reserved on Windows", "invalid_identifier")
    return value


def _windows_reserved_name(component: str) -> bool:
    return component.split(".", 1)[0].upper() in _WINDOWS_RESERVED


def _validated_utc(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        raise _manifest_error(
            f"{field} must use canonical UTC form YYYY-MM-DDTHH:MM:SSZ",
            "invalid_timestamp",
        )
    _parse_utc(value, field=field)
    return value


def _parse_utc(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise _manifest_error(f"{field} is not a real UTC timestamp", "invalid_timestamp") from exc
    return parsed


def _load_strict_json(raw: bytes, *, subject: str) -> object:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateKey(key)
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, _DuplicateKey) as exc:
        raise _manifest_error(f"{subject} is not strict UTF-8 JSON: {exc}", "invalid_json") from exc


def _expect_mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise _manifest_error(f"{field} must be an object", "invalid_manifest_shape")
    return cast(dict[str, Any], value)


def _expect_fields(value: dict[str, Any], expected: set[str], *, field: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise _manifest_error(
            f"{field} fields mismatch (missing={missing}, unknown={unknown})",
            "invalid_manifest_fields",
        )


class _DuplicateKey(ValueError):
    pass


def _manifest_error(message: str, code: str) -> ModelPackManifestError:
    return ModelPackManifestError(message, code=code)
