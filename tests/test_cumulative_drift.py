"""Tests for B2 — cumulative drift guard in multipass."""

import numpy as np

from hawavoclean.multipass import (
    MAX_CUMULATIVE_DRIFT_DB,
    cumulative_spectral_drift,
)


class TestCumulativeSpectralDrift:
    """B2 · cumulative_spectral_drift function."""

    def test_identity_is_zero(self) -> None:
        """Identical signals should have zero drift."""
        x = np.sin(np.linspace(0, 10, 4096)).astype(np.float64)
        assert cumulative_spectral_drift(x, x) < 0.01

    def test_different_signals_positive(self) -> None:
        """Different signals should have positive drift."""
        rng = np.random.default_rng(42)
        x = np.sin(np.linspace(0, 10, 4096)).astype(np.float64)
        y = x + 0.3 * rng.standard_normal(4096)
        assert cumulative_spectral_drift(x, y) > 0.0

    def test_more_noise_means_more_drift(self) -> None:
        """More noise should produce greater drift."""
        rng = np.random.default_rng(42)
        x = np.sin(np.linspace(0, 10, 4096)).astype(np.float64)
        y_mild = x + 0.05 * rng.standard_normal(4096)
        y_loud = x + 0.5 * rng.standard_normal(4096)
        assert cumulative_spectral_drift(x, y_mild) < cumulative_spectral_drift(x, y_loud)

    def test_ceiling_is_positive(self) -> None:
        """The hardcoded ceiling should be a positive number."""
        assert MAX_CUMULATIVE_DRIFT_DB > 0.0

    def test_length_mismatch_handled(self) -> None:
        """Mismatched lengths should not crash."""
        x = np.sin(np.linspace(0, 10, 4096)).astype(np.float64)
        y = np.sin(np.linspace(0, 10, 3000)).astype(np.float64)
        d = cumulative_spectral_drift(x, y)
        assert d >= 0.0
