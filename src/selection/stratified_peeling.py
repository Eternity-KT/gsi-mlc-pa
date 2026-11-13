"""Stratified Data-Driven Peeling Partition Provider for GSI-MLC-PA v5.1.

This module implements the non-greedy, performance-based iterative peeling
partitioning described in the historical v5.1 design.

Core Principles:
1. Performance-based Independent Label (IL) Identification:
   A label is promoted to IL_1 if and only if a Binary Relevance classifier
   trained on X achieves validation F1 >= threshold (default tau = 0.75).
2. Iterative Multi-Stage Peeling:
   Labels that do not qualify in stage 1 are re-evaluated using augmented features
   X^(t) = [X, P_hat(Y_{IL_{1:t-1}} = 1)].
3. Ascending Correlation Ordering in Residual DL:
   Labels remaining in DL are ordered by ASCENDING total correlation, so least
   coupled labels appear first in the CC chain to prevent error accumulation.
4. Modular Complexity Penalty:
   The complexity penalty evaluator is kept pluggable and disabled by default.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import clone

from .complexity_penalty import ComplexityPenaltyConfig, ComplexityPenaltyEvaluator


def _safe_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute binary F1 score safely without zero-division warnings."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        # If no true positives and no predicted positives: perfect if no positives exist
        return 1.0 if np.sum(y_true == 1) == 0 else 0.0
    return float((2.0 * tp) / denominator)


def find_optimal_binary_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_candidate_thresholds: int = 50,
) -> Tuple[float, float]:
    """Scan candidate thresholds to maximize binary F1 on validation data.

    Returns:
        (best_threshold, best_f1)
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)

    unique_probs = np.unique(y_prob)
    if len(unique_probs) <= 1:
        # Trivial uniform probabilities
        f1_half = _safe_f1_score(y_true, (y_prob >= 0.5).astype(np.int32))
        return 0.5, f1_half

    if len(unique_probs) <= n_candidate_thresholds:
        candidate_thresholds = unique_probs
    else:
        candidate_thresholds = np.linspace(0.05, 0.95, n_candidate_thresholds)

    best_threshold = 0.5
    best_f1 = -1.0

    for th in candidate_thresholds:
        y_pred = (y_prob >= th).astype(np.int32)
        score = _safe_f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(th)

    return best_threshold, float(best_f1)


def compute_label_correlation_matrix(Y: np.ndarray) -> np.ndarray:
    """Return absolute Pearson/Phi correlation matrix for binary labels."""
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError("Y must be a 2D matrix.")
    n_labels = Y.shape[1]
    if n_labels <= 1:
        return np.ones((n_labels, n_labels), dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(Y, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    abs_corr = np.abs(np.clip(corr, -1.0, 1.0))
    np.fill_diagonal(abs_corr, 1.0)
    return abs_corr


def order_dl_by_correlation(
    correlation_matrix: np.ndarray,
    dl_labels: Sequence[int],
    direction: str = "ascending",
) -> List[int]:
    """Order residual dependent labels by cumulative absolute correlation.

    Parameters:
        correlation_matrix: Full K x K correlation matrix.
        dl_labels: List of label indices currently in DL.
        direction: "ascending" (least coupled first, recommended in v5.1)
                   or "descending" (legacy behavior).

    Returns:
        List of label indices ordered according to the specified direction.
    """
    dl_list = [int(l) for l in dl_labels]
    if len(dl_list) <= 1:
        return dl_list

    # Compute internal cumulative correlation with other DL labels
    scores = {}
    for label in dl_list:
        other_labels = [j for j in dl_list if j != label]
        if other_labels:
            total_corr = float(np.sum(correlation_matrix[label, other_labels]))
        else:
            total_corr = 0.0
        scores[label] = total_corr

    if direction == "ascending":
        # Smallest total correlation first; break ties by label index
        ordered = sorted(dl_list, key=lambda l: (scores[l], l))
    elif direction == "descending":
        # Largest total correlation first; break ties by label index
        ordered = sorted(dl_list, key=lambda l: (-scores[l], l))
    else:
        raise ValueError(
            f"Unknown direction '{direction}'. Available: ['ascending', 'descending']."
        )

    return ordered


def normalize_augmented_probabilities(
    probs: np.ndarray,
    reference_X: np.ndarray,
    strategy: str = "matching",
) -> np.ndarray:
    """Normalize augmented soft probability features to match the distribution/scale of reference_X.

    Strategies:
      - "matching" (default): Checks reference_X. If reference_X has negative values
        (e.g., scaled to [-1, 1] by MaxAbsScaler or standardized by StandardScaler),
        transforms probs from [0, 1] to [-1, 1] via 2*p - 1.0 (centered).
        If reference_X is strictly non-negative in [0, 1], leaves probs in [0, 1].
      - "centered": Linearly maps p in [0, 1] to 2*p - 1 in [-1, 1].
      - "standard": Z-score standardization (mean=0, std=1) with safety epsilon.
      - "logit": Log-odds transform log(p / (1-p)) clipped to [-10, 10], then standardized.
      - "none": Leaves raw probabilities in [0, 1].
    """
    probs = np.clip(np.asarray(probs, dtype=np.float32), 0.0, 1.0)
    if strategy == "none":
        return probs

    if strategy == "matching":
        has_negative = bool(np.nanmin(reference_X) < -1e-5)
        if has_negative:
            return 2.0 * probs - 1.0
        return probs

    if strategy == "centered":
        return 2.0 * probs - 1.0

    if strategy == "standard":
        mean = np.mean(probs, axis=0, keepdims=True)
        std = np.std(probs, axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return (probs - mean) / std

    if strategy == "logit":
        eps = 1e-4
        clipped = np.clip(probs, eps, 1.0 - eps)
        logit = np.log(clipped / (1.0 - clipped))
        mean = np.mean(logit, axis=0, keepdims=True)
        std = np.std(logit, axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return (logit - mean) / std

    raise ValueError(
        f"Unknown aug_normalization strategy '{strategy}'. "
        f"Available: ['matching', 'centered', 'standard', 'logit', 'none']."
    )


@dataclass(frozen=True)
class StratifiedPeelingConfig:
    """Configuration for Stratified Data-Driven Peeling Selector.

    Attributes:
        threshold: Performance threshold (F1 score) required to promote a label to IL.
        max_depth: Maximum number of peeling stages (t = 1 .. max_depth).
        dl_order_direction: "ascending" (recommended v5.1) or "descending".
        min_labels_residual: Minimum number of remaining DL labels to continue peeling.
        use_complexity_penalty: Whether to evaluate the complexity penalty module (default False).
        penalty_config: Optional ComplexityPenaltyConfig when penalty is enabled.
        aug_normalization: Strategy to normalize augmented probability features ("matching", "centered", "standard", "logit", "none").
        decaying_threshold: Whether to decay threshold across stages.
        threshold_decay_step: Amount to decrease threshold at each stage if decaying_threshold is True.
    """
    threshold: float = 0.75
    max_depth: int = 3
    dl_order_direction: str = "ascending"
    min_labels_residual: int = 1
    use_complexity_penalty: bool = False
    penalty_config: Optional[ComplexityPenaltyConfig] = None
    aug_normalization: str = "matching"
    decaying_threshold: bool = False
    threshold_decay_step: float = 0.05

    def as_dict(self) -> Dict[str, Any]:
        return {
            "threshold": float(self.threshold),
            "max_depth": int(self.max_depth),
            "dl_order_direction": self.dl_order_direction,
            "min_labels_residual": int(self.min_labels_residual),
            "use_complexity_penalty": bool(self.use_complexity_penalty),
            "penalty_config": (
                None if self.penalty_config is None else self.penalty_config.as_dict()
            ),
            "aug_normalization": self.aug_normalization,
            "decaying_threshold": bool(self.decaying_threshold),
            "threshold_decay_step": float(self.threshold_decay_step),
        }


@dataclass(frozen=True)
class StratifiedPeelingResult:
    """Immutable audit record and partition result produced by StratifiedPeelingSelector."""
    independent_layers: Tuple[Tuple[int, ...], ...]
    all_independent_labels: Tuple[int, ...]
    dependent_residual_labels: Tuple[int, ...]
    final_execution_order: Tuple[int, ...]
    stage_diagnostics: Tuple[Dict[str, Any], ...]
    stopping_reason: str
    num_stages_executed: int
    selection_time_seconds: float

    @property
    def n_il_stage_1(self) -> int:
        return len(self.independent_layers[0]) if len(self.independent_layers) > 0 else 0

    @property
    def n_il_stage_2(self) -> int:
        return len(self.independent_layers[1]) if len(self.independent_layers) > 1 else 0

    @property
    def n_il_stage_3(self) -> int:
        return len(self.independent_layers[2]) if len(self.independent_layers) > 2 else 0

    @property
    def labels_il_stage_1(self) -> Tuple[int, ...]:
        return self.independent_layers[0] if len(self.independent_layers) > 0 else ()

    @property
    def labels_il_stage_2(self) -> Tuple[int, ...]:
        return self.independent_layers[1] if len(self.independent_layers) > 1 else ()

    @property
    def labels_il_stage_3(self) -> Tuple[int, ...]:
        return self.independent_layers[2] if len(self.independent_layers) > 2 else ()

    @property
    def n_total_il(self) -> int:
        return len(self.all_independent_labels)

    @property
    def n_residual_dl(self) -> int:
        return len(self.dependent_residual_labels)

    @property
    def total_labels(self) -> int:
        return self.n_total_il + self.n_residual_dl

    @property
    def pct_total_il(self) -> float:
        total = self.total_labels
        return float(self.n_total_il / total * 100.0) if total > 0 else 0.0

    @property
    def pct_residual_dl(self) -> float:
        total = self.total_labels
        return float(self.n_residual_dl / total * 100.0) if total > 0 else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "independent_layers": [list(layer) for layer in self.independent_layers],
            "all_independent_labels": [int(l) for l in self.all_independent_labels],
            "dependent_residual_labels": [
                int(l) for l in self.dependent_residual_labels
            ],
            "final_execution_order": [int(l) for l in self.final_execution_order],
            "stage_diagnostics": list(self.stage_diagnostics),
            "stopping_reason": self.stopping_reason,
            "num_stages_executed": int(self.num_stages_executed),
            "selection_time_seconds": float(self.selection_time_seconds),
            "n_il_stage_1": self.n_il_stage_1,
            "labels_il_stage_1": list(self.labels_il_stage_1),
            "n_il_stage_2": self.n_il_stage_2,
            "labels_il_stage_2": list(self.labels_il_stage_2),
            "n_il_stage_3": self.n_il_stage_3,
            "labels_il_stage_3": list(self.labels_il_stage_3),
            "n_total_il": self.n_total_il,
            "pct_total_il": self.pct_total_il,
            "n_residual_dl": self.n_residual_dl,
            "pct_residual_dl": self.pct_residual_dl,
            "labels_residual_dl": list(self.dependent_residual_labels),
        }


def _fit_predict_prob_binary(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
) -> np.ndarray:
    """Fit a binary estimator and return predicted probabilities on X_eval."""
    unique = np.unique(y_train)
    if len(unique) <= 1:
        const_val = float(unique[0]) if len(unique) == 1 else 0.0
        return np.full(X_eval.shape[0], const_val, dtype=np.float64)

    model = clone(estimator)
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(X_eval), dtype=np.float64)
        if probs.ndim == 2:
            return np.clip(probs[:, 1], 0.0, 1.0)
        return np.clip(probs.ravel(), 0.0, 1.0)
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X_eval), dtype=np.float64)
        return np.clip(1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0))), 0.0, 1.0)
    else:
        preds = np.asarray(model.predict(X_eval), dtype=np.float64)
        return np.clip(preds.ravel(), 0.0, 1.0)


class StratifiedPeelingSelector:
    """Data-Driven Multi-Stage Stratified Peeling Selector for MLC."""

    def __init__(
        self,
        config: Optional[StratifiedPeelingConfig] = None,
        base_estimator_factory: Optional[Callable[[], Any]] = None,
    ):
        self.config = config or StratifiedPeelingConfig()
        self.base_estimator_factory = base_estimator_factory
        if self.config.use_complexity_penalty:
            penalty_cfg = self.config.penalty_config or ComplexityPenaltyConfig(
                enabled=True
            )
            self.penalty_evaluator = ComplexityPenaltyEvaluator(penalty_cfg)
        else:
            self.penalty_evaluator = None

    def fit_partition(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        base_estimator=None,
    ) -> StratifiedPeelingResult:
        """Execute the multi-stage peeling algorithm and produce a frozen partition.

        Parameters:
            X_train: Training feature matrix of shape (n_train, n_features).
            Y_train: Training label matrix of shape (n_train, n_labels).
            X_val: Validation feature matrix of shape (n_val, n_features).
            Y_val: Validation label matrix of shape (n_val, n_labels).
            base_estimator: Optional scikit-learn binary estimator template.
        """
        start_time = perf_counter()
        X_train = np.asarray(X_train, dtype=np.float32)
        Y_train = np.asarray(Y_train, dtype=np.int32)
        X_val = np.asarray(X_val, dtype=np.float32)
        Y_val = np.asarray(Y_val, dtype=np.int32)

        n_labels = Y_train.shape[1]
        correlation_matrix = compute_label_correlation_matrix(Y_train)

        estimator_template = base_estimator
        if estimator_template is None and self.base_estimator_factory is not None:
            estimator_template = self.base_estimator_factory()

        independent_layers: List[List[int]] = []
        accumulated_il: List[int] = []
        candidate_dl: List[int] = list(range(n_labels))
        stage_diagnostics: List[Dict[str, Any]] = []
        stopping_reason = "max_depth_reached"

        # Pre-allocate feature augmentation buffers
        current_X_train = X_train
        current_X_val = X_val

        # Best F1 recorded for each label across stages to compute delta gain
        prev_scores = {k: 0.0 for k in range(n_labels)}
        all_train_probs = {}
        all_val_probs = {}

        for stage in range(1, self.config.max_depth + 1):
            stage_f1_scores = {}
            stage_thresholds = {}
            new_promoted_labels = []

            # Determine effective threshold for this stage
            if self.config.decaying_threshold:
                effective_threshold = max(
                    0.50,
                    float(self.config.threshold - (stage - 1) * self.config.threshold_decay_step),
                )
            else:
                effective_threshold = float(self.config.threshold)

            # Step 1: Train & evaluate binary estimators for all current candidate DL labels
            stage_val_probs = {}
            stage_train_probs = {}

            for label in candidate_dl:
                y_tr = Y_train[:, label]
                y_vl = Y_val[:, label]

                val_probs = _fit_predict_prob_binary(
                    estimator_template, current_X_train, y_tr, current_X_val
                )
                train_probs = _fit_predict_prob_binary(
                    estimator_template, current_X_train, y_tr, current_X_train
                )

                best_th, best_f1 = find_optimal_binary_threshold(y_vl, val_probs)
                stage_f1_scores[label] = best_f1
                stage_thresholds[label] = best_th
                stage_val_probs[label] = val_probs
                stage_train_probs[label] = train_probs
                all_train_probs[label] = train_probs
                all_val_probs[label] = val_probs

                # Promotion condition: Validation F1 >= effective threshold
                if best_f1 >= effective_threshold:
                    new_promoted_labels.append(label)

            # Calculate stage F1 gain across all candidate labels
            stage_f1_gain = 0.0
            if new_promoted_labels:
                gain_sum = sum(
                    stage_f1_scores[l] - prev_scores[l] for l in new_promoted_labels
                )
                stage_f1_gain = float(gain_sum / max(1, n_labels))

            # Record diagnostic record for this stage
            diag = {
                "stage": int(stage),
                "effective_threshold": float(effective_threshold),
                "num_candidate_dl": int(len(candidate_dl)),
                "num_promoted": int(len(new_promoted_labels)),
                "promoted_labels": list(new_promoted_labels),
                "stage_f1_scores": {int(k): float(v) for k, v in stage_f1_scores.items()},
                "stage_thresholds": {
                    int(k): float(v) for k, v in stage_thresholds.items()
                },
                "stage_f1_gain": float(stage_f1_gain),
            }

            # Update best recorded scores
            for l, score in stage_f1_scores.items():
                prev_scores[l] = max(prev_scores[l], score)

            # If no new labels reached threshold, terminate peeling
            if not new_promoted_labels:
                stopping_reason = "no_promotion_in_stage"
                stage_diagnostics.append(diag)
                break

            # Check complexity penalty if evaluator is active
            if self.penalty_evaluator is not None and self.config.use_complexity_penalty:
                penalty_res = self.penalty_evaluator.evaluate(
                    stage=stage,
                    remaining_dl_count=len(candidate_dl),
                    total_labels=n_labels,
                    f1_gain=stage_f1_gain,
                )
                diag["penalty_audit"] = penalty_res.as_dict()
                if penalty_res.should_stop:
                    stage_diagnostics.append(diag)
                    stopping_reason = f"complexity_penalty ({penalty_res.reason})"
                    break

            stage_diagnostics.append(diag)

            # Accept promoted labels into a new IL layer
            independent_layers.append(list(new_promoted_labels))
            accumulated_il.extend(new_promoted_labels)
            candidate_dl = [l for l in candidate_dl if l not in new_promoted_labels]

            # If DL is now empty or minimal, stop peeling
            if len(candidate_dl) <= self.config.min_labels_residual:
                stopping_reason = (
                    "all_labels_independent"
                    if len(candidate_dl) == 0
                    else "minimal_dl_residual_reached"
                )
                break

            # If we reached max depth, stop peeling
            if stage >= self.config.max_depth:
                stopping_reason = "max_depth_reached"
                break

            # Prepare augmented features for next stage: X^(t) = [X, Normalize(P_hat(Y_{IL}))]
            aug_train_raw = np.column_stack([all_train_probs[l] for l in accumulated_il])
            aug_val_raw = np.column_stack([all_val_probs[l] for l in accumulated_il])

            aug_train_norm = normalize_augmented_probabilities(
                aug_train_raw, reference_X=X_train, strategy=self.config.aug_normalization
            )
            aug_val_norm = normalize_augmented_probabilities(
                aug_val_raw, reference_X=X_val, strategy=self.config.aug_normalization
            )

            current_X_train = np.hstack([X_train, aug_train_norm])
            current_X_val = np.hstack([X_val, aug_val_norm])

        # Post-peeling ordering:
        # 1. Independent labels preserve layer order [IL_1, IL_2, ...]
        ordered_il = [l for layer in independent_layers for l in layer]

        # 2. Residual dependent labels ordered by correlation (ascending by default)
        ordered_dl = order_dl_by_correlation(
            correlation_matrix,
            candidate_dl,
            direction=self.config.dl_order_direction,
        )

        # 3. Final unified chain execution order: [IL_layers, ordered_DL]
        final_execution_order = ordered_il + ordered_dl

        total_time = float(perf_counter() - start_time)

        return StratifiedPeelingResult(
            independent_layers=tuple(tuple(layer) for layer in independent_layers),
            all_independent_labels=tuple(ordered_il),
            dependent_residual_labels=tuple(ordered_dl),
            final_execution_order=tuple(final_execution_order),
            stage_diagnostics=tuple(stage_diagnostics),
            stopping_reason=stopping_reason,
            num_stages_executed=len(stage_diagnostics),
            selection_time_seconds=total_time,
        )
