"""Complexity and Runtime Penalty Module for Stratified Partitioning.

This module provides standalone evaluation of computational complexity and runtime
penalties for the iterative peeling process in the historical v5.1 design.

Mathematical Formulation:
-------------------------
For stage t >= 1 and remaining candidate dependent labels D:
    P(t, |D|) = (|D| / K) * exp(beta * (t - 1))
    Delta U(t) = Delta F1(t) - lambda_complexity * P(t, |D|)

Decision Rule:
--------------
Stop peeling if Delta U(t) <= 0 (marginal utility of peeling another stage
does not justify the computational and latency overhead).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np


@dataclass(frozen=True)
class ComplexityPenaltyConfig:
    """Configuration for the computational complexity penalty evaluator.

    Attributes:
        penalty_lambda: Multiplier for the complexity penalty (lambda_complexity).
        penalty_beta: Exponential growth rate of the penalty per depth level (beta).
        enabled: Whether the penalty check is active during selection.
        f1_gain_tolerance: Minimum positive F1 gain required to continue even if penalty is 0.
    """
    penalty_lambda: float = 0.01
    penalty_beta: float = 0.5
    enabled: bool = False
    f1_gain_tolerance: float = 1e-4

    def as_dict(self) -> Dict[str, Any]:
        return {
            "penalty_lambda": float(self.penalty_lambda),
            "penalty_beta": float(self.penalty_beta),
            "enabled": bool(self.enabled),
            "f1_gain_tolerance": float(self.f1_gain_tolerance),
        }


@dataclass(frozen=True)
class PenaltyEvaluationResult:
    """Detailed audit record for one stage's complexity penalty assessment."""
    stage: int
    remaining_dl_count: int
    total_labels: int
    f1_gain: float
    raw_penalty: float
    weighted_penalty: float
    penalized_utility: float
    should_stop: bool
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage": int(self.stage),
            "remaining_dl_count": int(self.remaining_dl_count),
            "total_labels": int(self.total_labels),
            "f1_gain": float(self.f1_gain),
            "raw_penalty": float(self.raw_penalty),
            "weighted_penalty": float(self.weighted_penalty),
            "penalized_utility": float(self.penalized_utility),
            "should_stop": bool(self.should_stop),
            "reason": self.reason,
        }


class ComplexityPenaltyEvaluator:
    """Standalone evaluator for computational complexity penalties in multi-stage selection."""

    def __init__(self, config: Optional[ComplexityPenaltyConfig] = None):
        self.config = config or ComplexityPenaltyConfig()

    def compute_raw_penalty(
        self, stage: int, remaining_dl_count: int, total_labels: int
    ) -> float:
        """Compute P(t, |D|) = (|D| / K) * exp(beta * (t - 1))."""
        if total_labels <= 0:
            return 0.0
        dl_ratio = float(remaining_dl_count) / float(total_labels)
        depth_exponent = self.config.penalty_beta * float(max(0, stage - 1))
        # Clip exponent to avoid numerical overflow
        exp_factor = float(np.exp(np.clip(depth_exponent, 0.0, 30.0)))
        return dl_ratio * exp_factor

    def evaluate(
        self,
        stage: int,
        remaining_dl_count: int,
        total_labels: int,
        f1_gain: float,
    ) -> PenaltyEvaluationResult:
        """Evaluate whether to stop peeling based on marginal F1 gain and complexity penalty.

        Parameters:
            stage: Current peeling stage index (1-based, t >= 1).
            remaining_dl_count: Number of candidate labels currently in D.
            total_labels: Total number of labels K in the dataset.
            f1_gain: Validation Macro-F1 improvement brought by newly qualified labels.

        Returns:
            PenaltyEvaluationResult with detailed metrics and stopping recommendation.
        """
        raw_penalty = self.compute_raw_penalty(stage, remaining_dl_count, total_labels)
        weighted_penalty = self.config.penalty_lambda * raw_penalty
        penalized_utility = float(f1_gain - weighted_penalty)

        if not self.config.enabled:
            # When penalty is disabled, only check if there is non-negative gain
            should_stop = f1_gain < self.config.f1_gain_tolerance
            reason = (
                "f1_gain_below_tolerance" if should_stop else "penalty_disabled_continue"
            )
            return PenaltyEvaluationResult(
                stage=stage,
                remaining_dl_count=remaining_dl_count,
                total_labels=total_labels,
                f1_gain=float(f1_gain),
                raw_penalty=raw_penalty,
                weighted_penalty=weighted_penalty,
                penalized_utility=penalized_utility,
                should_stop=should_stop,
                reason=reason,
            )

        # When penalty is active
        if penalized_utility <= 0.0:
            should_stop = True
            reason = f"negative_penalized_utility (utility={penalized_utility:.6f} <= 0)"
        else:
            should_stop = False
            reason = f"positive_penalized_utility (utility={penalized_utility:.6f} > 0)"

        return PenaltyEvaluationResult(
            stage=stage,
            remaining_dl_count=remaining_dl_count,
            total_labels=total_labels,
            f1_gain=float(f1_gain),
            raw_penalty=raw_penalty,
            weighted_penalty=weighted_penalty,
            penalized_utility=penalized_utility,
            should_stop=should_stop,
            reason=reason,
        )
