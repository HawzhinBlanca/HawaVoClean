"""Typed failures for the signed model-pack trust boundary."""

from __future__ import annotations


class ModelPackError(RuntimeError):
    """Base class carrying a stable machine-readable failure code."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class ModelPackManifestError(ModelPackError):
    """The manifest or signature envelope is malformed or unsupported."""


class ModelPackSignatureError(ModelPackError):
    """The signing key or Ed25519 signature is not trusted."""


class ModelPackPayloadError(ModelPackError):
    """A declared payload or the on-disk pack layout is unsafe or corrupt."""


class ModelPackCompatibilityError(ModelPackError):
    """The authentic pack is not valid for this runtime or point in time."""


class ModelPackRollbackError(ModelPackError):
    """Installing the pack would move below the persisted version floor."""


class ModelPackInstallError(ModelPackError):
    """The application-managed pack store could not commit a verified pack."""
