"""Calibration, confidence estimation, and route-regret evaluation for Smart Safe.

Implements Task sheet Phase I3.6:
"Calibrate confidence, ties and abstention; least intervention wins low-confidence or tied decisions.
ECE <= 0.05, route-regret upper 95% CI < 0.10 MOS, deterministic order/tie tests and explicit
abstention evidence pass."

Provides:
1. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) calculation.
2. Temperature scaling calibration for ranker and candidate confidence probabilities.
3. Route-regret evaluation measuring subjective regret across safe candidate sets and computing
   the Student-t 95% confidence interval upper bound.
4. Determinism and abstention verification auditing that enumeration order never alters selection
   and ties/low-confidence strictly default to minimal intervention.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.optimize
import scipy.stats

from hawavoclean.smart_safe.decision import (
    INTERVENTION_COST,
    CandidateEvidence,
    Route,
    SmartSafeDecision,
    SmartSafePolicy,
)


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    """Quantitative calibration error statistics."""

    ece: float
    mce: float
    bin_accuracies: tuple[float, ...]
    bin_confidences: tuple[float, ...]
    bin_counts: tuple[int, ...]
    num_samples: int


@dataclass(frozen=True, slots=True)
class RouteRegretSummary:
    """Route regret statistics with two-sided and upper 95% confidence bounds."""

    mean_regret_mos: float
    std_regret_mos: float
    standard_error_mos: float
    ci95_upper_mos: float
    ci95_lower_mos: float
    max_regret_mos: float
    abstention_rate: float
    num_evaluations: int


class TemperatureCalibrator:
    """Post-hoc confidence calibrator using optimal temperature scaling.

    Maps uncalibrated logits or decision scores z to calibrated probabilities
    p = sigmoid(z / T), optimizing temperature T > 0 by minimizing negative
    log-likelihood over validation observations.
    """

    def __init__(self, initial_temperature: float = 1.0) -> None:
        if initial_temperature <= 0.0 or not math.isfinite(initial_temperature):
            raise ValueError("initial_temperature must be a finite positive float")
        self._temperature: float = initial_temperature
        self._fitted: bool = False

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(
        self, logits: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray
    ) -> float:
        """Fit optimal scalar temperature T minimizing cross-entropy.

        Parameters
        ----------
        logits:
            Raw model logits or decision score margins.
        labels:
            Binary correctness / preference indicators (0 or 1).

        Returns
        -------
        Optimal positive temperature T.
        """
        z = np.asarray(logits, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if len(z) != len(y) or len(z) == 0:
            raise ValueError("logits and labels must be non-empty and have matching lengths")
        if not np.all((y == 0.0) | (y == 1.0)):
            raise ValueError("labels must be binary (0 or 1)")

        def _nll(temp: float) -> float:
            scaled_z = z / temp
            # Stable log-sum-exp based binary cross-entropy
            # log(1 + exp(-z)) for y=1, log(1 + exp(z)) for y=0
            p = 1.0 / (1.0 + np.exp(-np.clip(scaled_z, -60.0, 60.0)))
            eps = 1e-12
            p = np.clip(p, eps, 1.0 - eps)
            loss = -float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
            return loss

        res = scipy.optimize.minimize_scalar(_nll, bounds=(0.05, 20.0), method="bounded")
        if not res.success:
            raise RuntimeError(f"temperature calibration optimization failed: {res.message}")
        self._temperature = float(res.x)
        self._fitted = True
        return self._temperature

    def calibrate(self, logits: Sequence[float] | np.ndarray) -> np.ndarray:
        """Apply fitted temperature scaling and sigmoid activation."""
        z = np.asarray(logits, dtype=np.float64)
        scaled = z / self._temperature
        clipped = np.clip(scaled, -60.0, 60.0)
        calibrated: np.ndarray = 1.0 / (1.0 + np.exp(-clipped))
        return calibrated


def compute_expected_calibration_error(
    confidences: Sequence[float] | np.ndarray,
    accuracies: Sequence[int | bool | float] | np.ndarray,
    num_bins: int = 10,
) -> CalibrationMetrics:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Partitions confidence scores into ``num_bins`` uniform intervals over [0, 1].
    In each bin B_m, calculates average accuracy acc(B_m) and average confidence
    conf(B_m). ECE is the sample-weighted absolute difference:

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Parameters
    ----------
    confidences:
        Predicted confidence values in [0, 1].
    accuracies:
        Binary indicator of actual correctness (1 = correct/preferred, 0 = wrong/suboptimal).
    num_bins:
        Number of equal-width calibration bins (default 10).
    """
    conf = np.asarray(confidences, dtype=np.float64)
    acc = np.asarray(accuracies, dtype=np.float64)

    if len(conf) != len(acc) or len(conf) == 0:
        raise ValueError("confidences and accuracies must have equal non-zero lengths")
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")
    if np.any(conf < 0.0) or np.any(conf > 1.0):
        raise ValueError("confidence values must be bounded in [0, 1]")

    n_samples = len(conf)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_accs: list[float] = []
    bin_confs: list[float] = []
    bin_counts: list[int] = []

    total_weighted_diff = 0.0
    max_diff = 0.0

    for i in range(num_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]
        # Include upper boundary on the final bin
        if i == num_bins - 1:
            mask = (conf >= low) & (conf <= high)
        else:
            mask = (conf >= low) & (conf < high)

        count = int(np.sum(mask))
        bin_counts.append(count)

        if count > 0:
            bin_acc = float(np.mean(acc[mask]))
            bin_conf = float(np.mean(conf[mask]))
            diff = abs(bin_acc - bin_conf)
            total_weighted_diff += (count / n_samples) * diff
            max_diff = max(max_diff, diff)
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append((low + high) / 2.0)

    return CalibrationMetrics(
        ece=float(total_weighted_diff),
        mce=float(max_diff),
        bin_accuracies=tuple(bin_accs),
        bin_confidences=tuple(bin_confs),
        bin_counts=tuple(bin_counts),
        num_samples=n_samples,
    )


def compute_route_regret(
    decisions: Sequence[SmartSafeDecision],
    ground_truth_qualities: Sequence[dict[Route, float]],
) -> RouteRegretSummary:
    """Compute empirical route regret and the Student-t upper 95% confidence bound.

    Route regret for decision i with safe survivor set Safe_i is:
    R_i = max_{r in Safe_i} Q_true(r) - Q_true(r_selected)

    Parameters
    ----------
    decisions:
        Sequence of evaluated SmartSafeDecision instances.
    ground_truth_qualities:
        Corresponding sequence of ground-truth MOS quality ratings per route.

    Returns
    -------
    RouteRegretSummary containing mean, std, standard error, and Student-t
    95% CI upper bound.
    """
    if len(decisions) != len(ground_truth_qualities) or len(decisions) == 0:
        raise ValueError("decisions and ground_truth_qualities must have matching non-zero lengths")

    regrets: list[float] = []
    n_abstained = 0

    for decision, qualities in zip(decisions, ground_truth_qualities, strict=True):
        if decision.abstained:
            n_abstained += 1

        safe_routes = [c.route for c in decision.candidates if c.safe]
        if not safe_routes:
            raise ValueError("decision contains no safe candidate route")

        for r in safe_routes:
            if r not in qualities:
                raise ValueError(f"ground_truth_qualities is missing rating for safe route {r}")
        if decision.selected_route not in qualities:
            raise ValueError(
                f"ground_truth_qualities is missing rating for selected route {decision.selected_route}"
            )

        best_safe_quality = max(qualities[r] for r in safe_routes)
        selected_quality = qualities[decision.selected_route]
        # Regret cannot be negative
        regret = max(0.0, best_safe_quality - selected_quality)
        regrets.append(regret)

    n = len(regrets)
    arr = np.asarray(regrets, dtype=np.float64)
    mean_regret = float(np.mean(arr))
    std_regret = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = std_regret / math.sqrt(n) if n > 1 else 0.0

    if n > 1:
        # One-sided / two-sided 95% critical value from Student's t
        t_crit = float(scipy.stats.t.ppf(0.975, df=n - 1))
        ci_upper = mean_regret + t_crit * se
        ci_lower = max(0.0, mean_regret - t_crit * se)
    else:
        ci_upper = mean_regret
        ci_lower = mean_regret

    return RouteRegretSummary(
        mean_regret_mos=mean_regret,
        std_regret_mos=std_regret,
        standard_error_mos=se,
        ci95_upper_mos=ci_upper,
        ci95_lower_mos=ci_lower,
        max_regret_mos=float(np.max(arr)),
        abstention_rate=n_abstained / n,
        num_evaluations=n,
    )


def verify_abstention_and_tie_properties(
    candidates: Sequence[CandidateEvidence],
    decision: SmartSafeDecision,
    policy: SmartSafePolicy,
) -> tuple[bool, str]:
    """Verify that ties and low-confidence decisions obey the least-intervention contract.

    Invariants:
    1. If abstained is True, the selected route must have the strictly lowest intervention
       cost among all safe survivors.
    2. If the quality margin between top two safe candidates is <= policy.tie_margin_mos,
       the decision must abstain.
    3. If the top candidate's prediction_confidence < policy.decision_confidence_min,
       the decision must abstain.
    4. The decision reason must explicitly disclose the abstention cause.
    """
    safe_survivors = [
        c for c in candidates if c.route in [o.route for o in decision.candidates if o.safe]
    ]
    if not safe_survivors:
        return False, "no safe survivors found"

    # Compute scores as in decision engine
    scores = {
        c.route: c.predicted_quality_mos
        - INTERVENTION_COST[c.route] * policy.intervention_penalty_mos
        for c in safe_survivors
    }
    ranked = sorted(
        safe_survivors,
        key=lambda item: (-scores[item.route], INTERVENTION_COST[item.route], item.route),
    )
    best = ranked[0]
    low_confidence = best.prediction_confidence < policy.decision_confidence_min
    tied = (
        len(ranked) > 1 and (scores[best.route] - scores[ranked[1].route]) <= policy.tie_margin_mos
    )
    expected_abstained = low_confidence or tied

    if decision.abstained != expected_abstained:
        return False, (
            f"abstention mismatch: decision.abstained={decision.abstained}, "
            f"expected={expected_abstained} (low_conf={low_confidence}, tied={tied})"
        )

    if decision.abstained:
        min_cost = min(INTERVENTION_COST[c.route] for c in safe_survivors)
        selected_cost = INTERVENTION_COST[decision.selected_route]
        if selected_cost != min_cost:
            return False, (
                f"abstained decision did not select least intervention: "
                f"selected={decision.selected_route} (cost {selected_cost}), min_cost={min_cost}"
            )
        if low_confidence and "confidence is low" not in decision.reason:
            return False, f"low-confidence reason missing from decision: {decision.reason!r}"
        if tied and not low_confidence and "tied" not in decision.reason:
            return False, f"tied reason missing from decision: {decision.reason!r}"

    return True, "verified"
