"""Multi-Stage 5-Fold Cross-Validation Peeling Selector for Anchor & Contextual Labels.

Partitions the label space into two semantically grounded tiers:
1. Anchor Labels (AL, L_anc):
   Labels with high standalone out-of-fold predictability (Selective-F1 >= threshold)
   that are predicted first and provide reliable probability context features.
2. Contextual Labels (CL, L_ctx):
   Remaining harder/ambiguous labels that leverage augmented context features
   X_context = [X, Normalize(P_hat_OOF(L_anc))] in the second stage.
"""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

from .stratified_peeling import (
    _safe_f1_score,
    compute_label_correlation_matrix,
    find_optimal_binary_threshold,
    normalize_augmented_probabilities,
    order_dl_by_correlation,
)


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
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X_eval), dtype=np.float64)
        return np.clip(1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0))), 0.0, 1.0)
    preds = np.asarray(model.predict(X_eval), dtype=np.float64)
    return np.clip(preds.ravel(), 0.0, 1.0)


def evaluate_label_5fold_cv(
    X: np.ndarray,
    y: np.ndarray,
    estimator_factory: Callable[[], Any],
    n_folds: int = 5,
    cost: float = 0.30,
    metric: str = "selective_f1",
    random_state: int = 42,
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """Perform K-fold cross-validation for one label and compute out-of-fold metrics."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    n_samples = len(y)

    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_count = int(np.min(class_counts)) if len(unique_classes) > 1 else 0

    if len(unique_classes) <= 1 or min_count < n_folds:
        cv_splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    else:
        cv_splitter = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )

    oof_probs = np.zeros(n_samples, dtype=np.float64)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        estimator = estimator_factory()
        probs_val = _fit_predict_prob_binary(estimator, X_tr, y_tr, X_val)
        oof_probs[val_idx] = probs_val

        decided_fold = np.minimum(probs_val, 1.0 - probs_val) <= cost
        if np.any(decided_fold):
            pred_fold = (probs_val[decided_fold] >= 0.5).astype(np.int32)
            f1_fold = _safe_f1_score(y_val[decided_fold], pred_fold)
        else:
            f1_fold = 0.0

        fold_metrics.append({
            "fold": fold_idx,
            "val_size": len(val_idx),
            "selective_f1": float(f1_fold),
            "coverage": float(np.mean(decided_fold)),
        })

    decided_all = np.minimum(oof_probs, 1.0 - oof_probs) <= cost
    coverage_all = float(np.mean(decided_all))

    if np.any(decided_all):
        y_pred_decided = (oof_probs[decided_all] >= 0.5).astype(np.int32)
        y_true_decided = y[decided_all]
        selective_f1 = _safe_f1_score(y_true_decided, y_pred_decided)
        sel_tp = np.sum((y_true_decided == 1) & (y_pred_decided == 1))
        sel_fp = np.sum((y_true_decided == 0) & (y_pred_decided == 1))
        selective_prec = float(sel_tp / (sel_tp + sel_fp)) if (sel_tp + sel_fp) > 0 else 0.0
        selective_acc = float(np.mean(y_true_decided == y_pred_decided))
    else:
        selective_f1 = 0.0
        selective_prec = 0.0
        selective_acc = 0.0

    optimal_th, optimal_f1 = find_optimal_binary_threshold(y, oof_probs)
    standard_f1 = _safe_f1_score(y, (oof_probs >= 0.5).astype(np.int32))

    if metric == "selective_f1":
        primary_score = selective_f1
    elif metric == "optimal_f1":
        primary_score = optimal_f1
    elif metric == "standard_f1":
        primary_score = standard_f1
    else:
        raise ValueError(
            f"Unknown metric '{metric}'. Available: ['selective_f1', 'optimal_f1', 'standard_f1']."
        )

    diagnostic = {
        "primary_score": float(primary_score),
        "selective_f1": float(selective_f1),
        "selective_precision": float(selective_prec),
        "selective_accuracy": float(selective_acc),
        "coverage": float(coverage_all),
        "optimal_threshold": float(optimal_th),
        "optimal_f1": float(optimal_f1),
        "standard_f1": float(standard_f1),
        "positive_rate": float(np.mean(y == 1)),
        "n_samples": n_samples,
        "fold_metrics": fold_metrics,
    }

    return float(primary_score), oof_probs, diagnostic


@dataclass(frozen=True)
class CVPeelingConfig:
    """Configuration for Multi-Stage K-Fold CV Anchor/Contextual Peeling Selector.

    Attributes:
        threshold: OOF metric threshold required to promote a candidate to Anchor Labels (AL).
        n_folds: Number of cross-validation folds (default 5).
        max_depth: Maximum number of anchor peeling stages (t = 1 .. max_depth).
        cost: Rejection cost c for partial abstention during peeling (default 0.30).
        metric: Promotion metric: 'selective_f1' (default), 'optimal_f1', or 'standard_f1'.
        decaying_threshold: Whether to decay promotion threshold across stages.
        threshold_decay_step: Threshold decrement per stage when decaying_threshold is True.
        min_threshold: Lower bound on decayed promotion threshold.
        aug_normalization: Normalization strategy for augmented Anchor probabilities.
        min_contextual_labels: Stop peeling when remaining Contextual Labels <= this count.
        dl_order_direction: Optional ordering for legacy chain callers ('ascending' or 'descending').
        random_state: Random seed for CV splits and base estimators.
    """

    threshold: float = 0.75
    n_folds: int = 5
    max_depth: int = 3
    cost: float = 0.30
    metric: str = "selective_f1"
    decaying_threshold: bool = False
    threshold_decay_step: float = 0.05
    min_threshold: float = 0.50
    aug_normalization: str = "matching"
    min_contextual_labels: int = 0
    dl_order_direction: str = "ascending"
    random_state: int = 42

    def __post_init__(self):
        if not 0.0 <= float(self.threshold) <= 2.0:
            raise ValueError(f"threshold must lie in [0.0, 2.0], got {self.threshold}.")
        if int(self.n_folds) < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}.")
        if int(self.max_depth) < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}.")
        if not 0.0 <= float(self.cost) <= 1.0:
            raise ValueError(f"cost must lie in [0.0, 1.0], got {self.cost}.")
        if self.metric not in ("selective_f1", "optimal_f1", "standard_f1"):
            raise ValueError(
                f"Unknown metric '{self.metric}'. Available: ['selective_f1', 'optimal_f1', 'standard_f1']."
            )
        if self.dl_order_direction not in ("ascending", "descending"):
            raise ValueError(
                f"Unknown dl_order_direction '{self.dl_order_direction}'. Available: ['ascending', 'descending']."
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "threshold": float(self.threshold),
            "n_folds": int(self.n_folds),
            "max_depth": int(self.max_depth),
            "cost": float(self.cost),
            "metric": self.metric,
            "decaying_threshold": bool(self.decaying_threshold),
            "threshold_decay_step": float(self.threshold_decay_step),
            "min_threshold": float(self.min_threshold),
            "aug_normalization": self.aug_normalization,
            "min_contextual_labels": int(self.min_contextual_labels),
            "dl_order_direction": self.dl_order_direction,
            "random_state": int(self.random_state),
        }


@dataclass(frozen=True)
class CVPeelingResult:
    """Immutable audit record and Anchor/Contextual partition result."""

    anchor_layers: Tuple[Tuple[int, ...], ...]
    anchor_labels: Tuple[int, ...]
    contextual_labels: Tuple[int, ...]
    final_execution_order: Tuple[int, ...]
    stage_diagnostics: Tuple[Dict[str, Any], ...]
    anchor_br_metrics: Dict[str, Any]
    stopping_reason: str
    num_stages_executed: int
    selection_time_seconds: float
    oof_probabilities: Dict[int, Any] = field(default_factory=dict)

    @property
    def n_al_stage_1(self) -> int:
        return len(self.anchor_layers[0]) if len(self.anchor_layers) > 0 else 0

    @property
    def n_al_stage_2(self) -> int:
        return len(self.anchor_layers[1]) if len(self.anchor_layers) > 1 else 0

    @property
    def n_al_stage_3(self) -> int:
        return len(self.anchor_layers[2]) if len(self.anchor_layers) > 2 else 0

    @property
    def labels_al_stage_1(self) -> Tuple[int, ...]:
        return self.anchor_layers[0] if len(self.anchor_layers) > 0 else ()

    @property
    def labels_al_stage_2(self) -> Tuple[int, ...]:
        return self.anchor_layers[1] if len(self.anchor_layers) > 1 else ()

    @property
    def labels_al_stage_3(self) -> Tuple[int, ...]:
        return self.anchor_layers[2] if len(self.anchor_layers) > 2 else ()

    @property
    def n_anchor_labels(self) -> int:
        return len(self.anchor_labels)

    @property
    def n_contextual_labels(self) -> int:
        return len(self.contextual_labels)

    @property
    def total_labels(self) -> int:
        return self.n_anchor_labels + self.n_contextual_labels

    @property
    def pct_anchor_labels(self) -> float:
        total = self.total_labels
        return float(self.n_anchor_labels / total * 100.0) if total > 0 else 0.0

    @property
    def pct_contextual_labels(self) -> float:
        total = self.total_labels
        return float(self.n_contextual_labels / total * 100.0) if total > 0 else 0.0

    # Backward-compatibility aliases for legacy v5/v6 callers (gsi_mlc_pa.py)
    @property
    def independent_layers(self) -> Tuple[Tuple[int, ...], ...]:
        return self.anchor_layers

    @property
    def all_independent_labels(self) -> Tuple[int, ...]:
        return self.anchor_labels

    @property
    def dependent_residual_labels(self) -> Tuple[int, ...]:
        return self.contextual_labels

    @property
    def br_independent_metrics(self) -> Dict[str, Any]:
        return self.anchor_br_metrics

    def as_dict(self) -> Dict[str, Any]:
        return {
            "anchor_layers": [list(layer) for layer in self.anchor_layers],
            "anchor_labels": [int(l) for l in self.anchor_labels],
            "contextual_labels": [int(l) for l in self.contextual_labels],
            "final_execution_order": [int(l) for l in self.final_execution_order],
            "stage_diagnostics": list(self.stage_diagnostics),
            "anchor_br_metrics": dict(self.anchor_br_metrics),
            "stopping_reason": self.stopping_reason,
            "num_stages_executed": int(self.num_stages_executed),
            "selection_time_seconds": float(self.selection_time_seconds),
            "n_al_stage_1": self.n_al_stage_1,
            "labels_al_stage_1": list(self.labels_al_stage_1),
            "n_al_stage_2": self.n_al_stage_2,
            "labels_al_stage_2": list(self.labels_al_stage_2),
            "n_al_stage_3": self.n_al_stage_3,
            "labels_al_stage_3": list(self.labels_al_stage_3),
            "n_anchor_labels": self.n_anchor_labels,
            "pct_anchor_labels": self.pct_anchor_labels,
            "n_contextual_labels": self.n_contextual_labels,
            "pct_contextual_labels": self.pct_contextual_labels,
        }


class CVStratifiedPeelingSelector:
    """Multi-Stage 5-Fold CV Peeling Selector discovering Anchor (AL) and Contextual (CL) labels."""

    def __init__(
        self,
        config: Optional[CVPeelingConfig] = None,
        base_estimator_factory: Optional[Callable[[], Any]] = None,
    ):
        self.config = config or CVPeelingConfig()
        self.base_estimator_factory = base_estimator_factory

    def fit_partition(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        base_estimator_factory: Optional[Callable[[], Any]] = None,
    ) -> CVPeelingResult:
        """Execute the 5-fold CV peeling algorithm to partition labels into AL and CL."""
        start_time = perf_counter()
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got ndim={X.ndim}.")
        if Y.ndim != 2:
            raise ValueError(f"Y must be a 2D array, got ndim={Y.ndim}.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Sample count mismatch: X has {X.shape[0]} samples, Y has {Y.shape[0]} samples."
            )
        if X.shape[0] == 0:
            raise ValueError("Dataset cannot be empty (0 samples).")
        if Y.shape[1] == 0:
            raise ValueError("Y must contain at least one label.")

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        _, n_labels = Y.shape

        factory = base_estimator_factory or self.base_estimator_factory
        if factory is None:
            from ..models.base_learners import create_binary_estimator
            factory = lambda: create_binary_estimator("logistic", random_state=self.config.random_state)

        anchor_layers: List[List[int]] = []
        accumulated_al: List[int] = []
        candidate_cl: List[int] = list(range(n_labels))
        stage_diagnostics: List[Dict[str, Any]] = []
        stopping_reason = "max_depth_reached"

        all_oof_probs: Dict[int, np.ndarray] = {}
        all_label_diagnostics: Dict[int, Dict[str, Any]] = {}
        current_X = X

        for stage in range(1, self.config.max_depth + 1):
            if self.config.decaying_threshold:
                effective_threshold = max(
                    self.config.min_threshold,
                    float(self.config.threshold - (stage - 1) * self.config.threshold_decay_step),
                )
            else:
                effective_threshold = float(self.config.threshold)

            stage_scores: Dict[int, float] = {}
            stage_diagnostics_per_label: Dict[int, Dict[str, Any]] = {}
            promoted_anchors: List[int] = []
            stage_oof_probs: Dict[int, np.ndarray] = {}

            for label in candidate_cl:
                score, oof_p, diag = evaluate_label_5fold_cv(
                    X=current_X,
                    y=Y[:, label],
                    estimator_factory=factory,
                    n_folds=self.config.n_folds,
                    cost=self.config.cost,
                    metric=self.config.metric,
                    random_state=self.config.random_state + stage,
                )
                stage_scores[label] = score
                stage_diagnostics_per_label[label] = diag
                stage_oof_probs[label] = oof_p

                if score >= effective_threshold:
                    promoted_anchors.append(label)

            stage_diagnostics.append({
                "stage": int(stage),
                "effective_threshold": float(effective_threshold),
                "num_candidate_cl": int(len(candidate_cl)),
                "num_promoted": int(len(promoted_anchors)),
                "promoted_anchor_labels": list(promoted_anchors),
                "stage_scores": {int(k): float(v) for k, v in stage_scores.items()},
                "label_diagnostics": {
                    int(k): v for k, v in stage_diagnostics_per_label.items()
                },
            })

            if not promoted_anchors:
                stopping_reason = "no_promotion_in_stage"
                break

            for lbl in promoted_anchors:
                all_oof_probs[lbl] = stage_oof_probs[lbl]
                all_label_diagnostics[lbl] = stage_diagnostics_per_label[lbl]

            anchor_layers.append(list(promoted_anchors))
            accumulated_al.extend(promoted_anchors)
            candidate_cl = [lbl for lbl in candidate_cl if lbl not in promoted_anchors]

            if len(candidate_cl) <= self.config.min_contextual_labels:
                stopping_reason = (
                    "all_labels_anchor"
                    if len(candidate_cl) == 0
                    else "minimal_contextual_labels_reached"
                )
                break

            if stage >= self.config.max_depth:
                stopping_reason = "max_depth_reached"
                break

            aug_raw = np.column_stack([all_oof_probs[lbl] for lbl in accumulated_al])
            aug_norm = normalize_augmented_probabilities(
                aug_raw, reference_X=X, strategy=self.config.aug_normalization
            )
            current_X = np.hstack([X, aug_norm])

        ordered_al = [lbl for layer in anchor_layers for lbl in layer]
        if len(candidate_cl) > 1:
            correlation_matrix = compute_label_correlation_matrix(Y)
            ordered_cl = order_dl_by_correlation(
                correlation_matrix,
                candidate_cl,
                direction=self.config.dl_order_direction,
            )
        else:
            ordered_cl = list(candidate_cl)

        final_execution_order = ordered_al + ordered_cl

        anchor_br_metrics: Dict[str, Any] = {}
        if ordered_al:
            al_selective_f1s = [
                all_label_diagnostics[l]["selective_f1"]
                for l in ordered_al
                if l in all_label_diagnostics
            ]
            al_coverages = [
                all_label_diagnostics[l]["coverage"]
                for l in ordered_al
                if l in all_label_diagnostics
            ]
            al_precisions = [
                all_label_diagnostics[l]["selective_precision"]
                for l in ordered_al
                if l in all_label_diagnostics
            ]
            anchor_br_metrics = {
                "num_anchor_labels": len(ordered_al),
                "mean_selective_f1": float(np.mean(al_selective_f1s)) if al_selective_f1s else 0.0,
                "mean_coverage": float(np.mean(al_coverages)) if al_coverages else 0.0,
                "mean_selective_precision": float(np.mean(al_precisions)) if al_precisions else 0.0,
                "per_label_selective_f1": {
                    int(l): float(all_label_diagnostics[l]["selective_f1"])
                    for l in ordered_al
                    if l in all_label_diagnostics
                },
            }

        total_time = float(perf_counter() - start_time)

        return CVPeelingResult(
            anchor_layers=tuple(tuple(layer) for layer in anchor_layers),
            anchor_labels=tuple(ordered_al),
            contextual_labels=tuple(ordered_cl),
            final_execution_order=tuple(final_execution_order),
            stage_diagnostics=tuple(stage_diagnostics),
            anchor_br_metrics=anchor_br_metrics,
            stopping_reason=stopping_reason,
            num_stages_executed=len(stage_diagnostics),
            selection_time_seconds=total_time,
            oof_probabilities=dict(all_oof_probs),
        )
