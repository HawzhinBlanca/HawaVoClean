"""Unit tests for high-confidence token anchor verification."""

import pytest

from hawavoclean.guard.protocol import TokenInfo
from hawavoclean.guard.token_anchor import compare_token_anchors


@pytest.mark.unit
def test_anchor_perfect_match() -> None:
    orig = [
        TokenInfo(token_id=1, text="سڵاو", start_time_s=0.2, end_time_s=0.6, confidence=0.92),
        TokenInfo(token_id=2, text="لە", start_time_s=0.7, end_time_s=0.9, confidence=0.88),
        TokenInfo(token_id=3, text="هەمووان", start_time_s=1.0, end_time_s=1.5, confidence=0.95),
    ]
    cand = [
        TokenInfo(token_id=1, text="سڵاو", start_time_s=0.21, end_time_s=0.61, confidence=0.91),
        TokenInfo(token_id=2, text="لە", start_time_s=0.71, end_time_s=0.91, confidence=0.87),
        TokenInfo(token_id=3, text="هەمووان", start_time_s=1.01, end_time_s=1.51, confidence=0.94),
    ]
    res = compare_token_anchors(orig, cand, min_anchor_confidence=0.75)
    assert res.passed is True
    assert res.deleted_anchors_count == 0
    assert res.substituted_anchors_count == 0


@pytest.mark.unit
def test_anchor_rejects_substitution() -> None:
    orig = [
        TokenInfo(token_id=1, text="کتێب", start_time_s=0.2, end_time_s=0.6, confidence=0.92),
    ]
    cand = [
        TokenInfo(token_id=1, text="سێو", start_time_s=0.2, end_time_s=0.6, confidence=0.90),
    ]
    res = compare_token_anchors(orig, cand, min_anchor_confidence=0.75)
    assert res.passed is False
    assert res.substituted_anchors_count == 1
    assert any("substituted" in r for r in res.failure_reasons)


@pytest.mark.unit
def test_anchor_rejects_deletion() -> None:
    orig = [
        TokenInfo(token_id=1, text="هەولێر", start_time_s=0.2, end_time_s=0.6, confidence=0.95),
        TokenInfo(token_id=2, text="سلێمانی", start_time_s=0.7, end_time_s=1.2, confidence=0.94),
    ]
    cand = [
        TokenInfo(token_id=1, text="هەولێر", start_time_s=0.2, end_time_s=0.6, confidence=0.95),
    ]
    res = compare_token_anchors(orig, cand, min_anchor_confidence=0.75)
    assert res.passed is False
    assert res.deleted_anchors_count == 1


@pytest.mark.unit
def test_anchor_insufficient_anchors_returns_unverified() -> None:
    # Low confidence in original tokens
    orig = [
        TokenInfo(token_id=1, text="نادیار", start_time_s=0.2, end_time_s=0.6, confidence=0.45),
    ]
    cand = [
        TokenInfo(token_id=1, text="نادیار", start_time_s=0.2, end_time_s=0.6, confidence=0.50),
    ]
    res = compare_token_anchors(orig, cand, min_anchor_confidence=0.75)
    assert res.passed is False
    assert res.insufficient_anchors is True
