"""Unit tests for CTC posterior Jensen-Shannon divergence analysis."""

import numpy as np
import pytest

from hawavoclean.guard.posterior import compare_ctc_posteriors, compute_js_divergence


@pytest.mark.unit
def test_js_divergence_identical_posteriors() -> None:
    p = np.array([[0.7, 0.2, 0.1]], dtype=np.float32)
    q = np.array([[0.7, 0.2, 0.1]], dtype=np.float32)
    js = compute_js_divergence(p, q)
    assert float(js[0]) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.unit
def test_js_divergence_divergent_distributions() -> None:
    p = np.array([[0.9, 0.05, 0.05]], dtype=np.float32)
    q = np.array([[0.05, 0.9, 0.05]], dtype=np.float32)
    js = compute_js_divergence(p, q)
    assert float(js[0]) > 0.50


@pytest.mark.unit
def test_compare_ctc_posteriors_threshold_rejection() -> None:
    # 50 frames, 10 vocab
    p_orig = np.zeros((50, 10), dtype=np.float32)
    p_orig[:, 0] = 0.2  # voiced
    p_orig[:, 1] = 0.8

    p_cand = np.zeros((50, 10), dtype=np.float32)
    p_cand[:, 0] = 0.2
    p_cand[:, 2] = 0.8  # completely shifted to different phoneme

    res = compare_ctc_posteriors(p_orig, p_cand, max_mean_js_div=0.25)
    assert res.passed is False
    assert res.mean_js_divergence > 0.25
