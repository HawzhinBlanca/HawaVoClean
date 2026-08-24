"""Error hierarchy and exit code definitions for HawaVoClean."""

from enum import IntEnum


class ExitCode(IntEnum):
    """Stable process exit codes published by the CLI contract."""

    SUCCESS = 0
    PREFLIGHT_FAILURE = 2
    PUBLICATION_FAILURE = 3
    INVALID_USER_INPUT = 4


class HawaVoCleanError(Exception):
    """Base exception for all HawaVoClean runtime errors."""

    def __init__(self, message: str, exit_code: ExitCode = ExitCode.PREFLIGHT_FAILURE) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


class PreflightError(HawaVoCleanError):
    """Raised when environment, disk space, or dependency validation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=ExitCode.PREFLIGHT_FAILURE)


class ConfigError(HawaVoCleanError):
    """Raised when configuration validation or schema checks fail."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=ExitCode.PREFLIGHT_FAILURE)


class ModelProvenanceError(HawaVoCleanError):
    """Raised when model weights, hashes, licenses, or lockfiles fail validation."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=ExitCode.PREFLIGHT_FAILURE)


class CalibrationError(HawaVoCleanError):
    """Raised when guard calibration artifact is missing, mismatched, or corrupted."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=ExitCode.PREFLIGHT_FAILURE)


class InvalidUserInputError(HawaVoCleanError):
    """Raised when input audio format, sample rate, or arguments are unsupported."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=ExitCode.INVALID_USER_INPUT)


class AmbiguousStereoError(HawaVoCleanError):
    """Raised when stereo audio has ambiguous channel relationships requiring user declaration."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=ExitCode.INVALID_USER_INPUT)


class PublicationError(HawaVoCleanError):
    """Raised when validation or committed-generation publication fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=ExitCode.PUBLICATION_FAILURE)


class OutputValidationError(PublicationError):
    """Raised when generated audio violates sample count, NaN/Inf, or structural invariants."""


class WorkerError(Exception):
    """Base exception for isolated enhancement worker failures."""


class WorkerCrashError(WorkerError):
    """Worker process terminated unexpectedly."""


class WorkerTimeoutError(WorkerError):
    """Worker process exceeded execution deadline."""


class WorkerOOMError(WorkerError):
    """Worker process encountered Out-Of-Memory (OOM)."""
