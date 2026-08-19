"""Error hierarchy and exit code definitions for Hawzhin VoiceClean."""

from enum import IntEnum


class ExitCode(IntEnum):
    """Standardized process exit codes defined in BLUEPRINT.md."""

    SUCCESS = 0
    PREFLIGHT_FAILURE = 2
    PUBLICATION_FAILURE = 3
    INVALID_USER_INPUT = 4


class VoiceCleanError(Exception):
    """Base exception for all VoiceClean runtime errors."""

    def __init__(self, message: str, exit_code: ExitCode = ExitCode.PREFLIGHT_FAILURE) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


class PreflightError(VoiceCleanError):
    """Raised when environment, disk space, or dependency validation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=ExitCode.PREFLIGHT_FAILURE)


class ConfigError(VoiceCleanError):
    """Raised when configuration validation or schema checks fail."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=ExitCode.PREFLIGHT_FAILURE)


class ModelProvenanceError(VoiceCleanError):
    """Raised when model weights, hashes, licenses, or lockfiles fail validation."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=ExitCode.PREFLIGHT_FAILURE)


class CalibrationError(VoiceCleanError):
    """Raised when guard calibration artifact is missing, mismatched, or corrupted."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=ExitCode.PREFLIGHT_FAILURE)


class InvalidUserInputError(VoiceCleanError):
    """Raised when input audio format, sample rate, or arguments are unsupported."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=ExitCode.INVALID_USER_INPUT)


class AmbiguousStereoError(VoiceCleanError):
    """Raised when stereo audio has ambiguous channel relationships requiring user declaration."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=ExitCode.INVALID_USER_INPUT)


class PublicationError(VoiceCleanError):
    """Raised when final validation or atomic file publication fails."""

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
