"""Timestamp-aware token anchor alignment and substitution/deletion detection."""

from dataclasses import dataclass, field

from hawavoclean.guard.protocol import TokenInfo


@dataclass
class AnchorComparisonResult:
    """Detailed token anchor comparison statistics."""

    passed: bool
    insufficient_anchors: bool
    total_original_tokens: int
    high_conf_anchors_count: int
    deleted_anchors_count: int
    substituted_anchors_count: int
    inserted_tokens_count: int
    max_timestamp_drift_ms: float
    mean_confidence_delta: float
    failure_reasons: list[str] = field(default_factory=list)


def compare_token_anchors(
    orig_tokens: list[TokenInfo],
    cand_tokens: list[TokenInfo],
    min_anchor_confidence: float = 0.75,
    max_timestamp_drift_ms: float = 40.0,
    max_confidence_drop: float = 0.50,
    min_required_anchors: int = 1,
) -> AnchorComparisonResult:
    """Compare candidate tokens against original high-confidence anchors.

    Fails immediately on any anchor substitution or deletion.
    """
    reasons: list[str] = []

    if not orig_tokens:
        # No original tokens
        if cand_tokens:
            # Candidate hallucinated tokens where original had none
            return AnchorComparisonResult(
                passed=False,
                insufficient_anchors=False,
                total_original_tokens=0,
                high_conf_anchors_count=0,
                deleted_anchors_count=0,
                substituted_anchors_count=0,
                inserted_tokens_count=len(cand_tokens),
                max_timestamp_drift_ms=0.0,
                mean_confidence_delta=0.0,
                failure_reasons=["Candidate hallucinated tokens in silence/no-token region."],
            )
        return AnchorComparisonResult(
            passed=True,
            insufficient_anchors=False,
            total_original_tokens=0,
            high_conf_anchors_count=0,
            deleted_anchors_count=0,
            substituted_anchors_count=0,
            inserted_tokens_count=0,
            max_timestamp_drift_ms=0.0,
            mean_confidence_delta=0.0,
        )

    # Identify high-confidence anchors in original
    anchors = [t for t in orig_tokens if t.confidence >= min_anchor_confidence]
    if len(anchors) < min_required_anchors:
        # Not enough anchors to guarantee verification
        return AnchorComparisonResult(
            passed=False,
            insufficient_anchors=True,
            total_original_tokens=len(orig_tokens),
            high_conf_anchors_count=len(anchors),
            deleted_anchors_count=0,
            substituted_anchors_count=0,
            inserted_tokens_count=0,
            max_timestamp_drift_ms=0.0,
            mean_confidence_delta=0.0,
            failure_reasons=["Insufficient high-confidence token anchors in speech unit."],
        )

    # Dynamic Programming Alignment between original and candidate tokens
    n = len(orig_tokens)
    m = len(cand_tokens)

    # Cost matrix for Levenshtein with timestamp penalty
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * 1.0
    for j in range(1, m + 1):
        dp[0][j] = j * 1.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            t_orig = orig_tokens[i - 1]
            t_cand = cand_tokens[j - 1]

            match = t_orig.text == t_cand.text
            cost_sub = 0.0 if match else 1.0
            # Timestamp drift penalty
            time_drift_s = abs(t_orig.start_time_s - t_cand.start_time_s)
            cost_sub += min(1.0, time_drift_s * 2.0)

            dp[i][j] = min(
                dp[i - 1][j] + 1.0,  # deletion
                dp[i][j - 1] + 1.0,  # insertion
                dp[i - 1][j - 1] + cost_sub,  # match / substitution
            )

    # Backtrack alignment
    i, j = n, m
    deleted_anchors = 0
    substituted_anchors = 0
    inserted_tokens = 0
    max_drift_ms = 0.0
    conf_deltas: list[float] = []

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            t_orig = orig_tokens[i - 1]
            t_cand = cand_tokens[j - 1]
            is_anchor = t_orig.confidence >= min_anchor_confidence

            # Check if this step came from diagonal
            time_drift_s = abs(t_orig.start_time_s - t_cand.start_time_s)
            diag_cost = (0.0 if t_orig.text == t_cand.text else 1.0) + min(1.0, time_drift_s * 2.0)

            if abs(dp[i][j] - (dp[i - 1][j - 1] + diag_cost)) < 1e-4:
                # Diagonal step
                if t_orig.text == t_cand.text:
                    # Match
                    if is_anchor:
                        drift_ms = time_drift_s * 1000.0
                        if drift_ms > max_drift_ms:
                            max_drift_ms = drift_ms
                        if drift_ms > max_timestamp_drift_ms:
                            reasons.append(
                                f"Anchor '{t_orig.text}' drifted {drift_ms:.1f}ms (threshold: {max_timestamp_drift_ms}ms)"
                            )

                        conf_drop = t_orig.confidence - t_cand.confidence
                        conf_deltas.append(conf_drop)
                        if conf_drop > max_confidence_drop:
                            reasons.append(
                                f"Anchor '{t_orig.text}' confidence dropped by {conf_drop:.2f} (from {t_orig.confidence:.2f} to {t_cand.confidence:.2f})"
                            )
                else:
                    # Substitution
                    if is_anchor:
                        substituted_anchors += 1
                        reasons.append(
                            f"High-confidence anchor '{t_orig.text}' substituted by '{t_cand.text}'"
                        )
                i -= 1
                j -= 1
                continue

        if i > 0 and (j == 0 or dp[i][j] == dp[i - 1][j] + 1.0):
            # Deletion
            t_orig = orig_tokens[i - 1]
            if t_orig.confidence >= min_anchor_confidence:
                deleted_anchors += 1
                reasons.append(f"High-confidence anchor '{t_orig.text}' was deleted.")
            i -= 1
        else:
            # Insertion
            inserted_tokens += 1
            j -= 1

    passed = deleted_anchors == 0 and substituted_anchors == 0 and len(reasons) == 0
    mean_delta = float(sum(conf_deltas) / len(conf_deltas)) if conf_deltas else 0.0

    return AnchorComparisonResult(
        passed=passed,
        insufficient_anchors=False,
        total_original_tokens=n,
        high_conf_anchors_count=len(anchors),
        deleted_anchors_count=deleted_anchors,
        substituted_anchors_count=substituted_anchors,
        inserted_tokens_count=inserted_tokens,
        max_timestamp_drift_ms=max_drift_ms,
        mean_confidence_delta=mean_delta,
        failure_reasons=reasons,
    )
