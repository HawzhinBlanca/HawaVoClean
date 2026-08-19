"""Unit tests for error hierarchy and structured logging."""

import pytest

from hawavoclean.errors import (
    AmbiguousStereoError,
    ConfigError,
    ExitCode,
    OutputValidationError,
    PreflightError,
)
from hawavoclean.logging import get_logger, setup_logging


@pytest.mark.unit
def test_error_exit_codes() -> None:
    assert PreflightError("err").exit_code == ExitCode.PREFLIGHT_FAILURE
    assert ConfigError("err").exit_code == ExitCode.PREFLIGHT_FAILURE
    assert OutputValidationError("err").exit_code == ExitCode.PUBLICATION_FAILURE
    assert AmbiguousStereoError("err").exit_code == ExitCode.INVALID_USER_INPUT


@pytest.mark.unit
def test_logging_setup() -> None:
    logger = setup_logging()
    assert logger.name == "hawavoclean"
    log = get_logger("test")
    assert log.name == "hawavoclean.test"
