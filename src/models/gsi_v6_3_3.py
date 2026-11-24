"""GSI-MLC-PA Classifier with Minimal 3-Hyperparameter Architecture.

Architecture (3 core hyperparameters: cost, anchor_threshold, gamma_min):
1. Phase 1 (Single-Stage Anchor Discovery):
   Evaluates all labels via K-fold out-of-fold Selective-F1 at rejection cost c.
   Labels with Selective-F1_OOF >= anchor_threshold form Anchor Labels (AL);
   remaining labels form Contextual Labels (CL).
2. Phase 2 (Zero-Centered Contextual Inference):
   If AL is non-empty, predicts Contextual Labels on X_context = [X, 2*P_hat(AL) - 1].
   If AL is empty, reuses Phase-1 OOF probabilities directly without redundant CV fits.
3. Phase 3 (Unified Identity-Regularized Balanced-Root Calibration):
   Calibrates OOF and test probabilities smoothly across all class-imbalance regimes.
4. Phase 4 (Closed-Form Prior-Adjusted Bayes Decision & OOF Coverage Guard):
   Computes parameter-free per-label Bayes thresholds [tau_0, tau_1] in {-1, 0, 1}
   with OOF coverage floor gamma_min.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .base_learners import create_binary_estimator
from .binary_relevance import _ConstantClassifier
from ..selection.cv_peeling import evaluate_label_5fold_cv
from ..selection.tri_regime_decision import (
    apply_tri_regime_abstention,
    compute_tri_regime_adaptive_thresholds_table,
)
from ..calibration.adaptive_calibrator import MultiLabelAdaptiveCalibrator


def _fit_binary_model_safe(estimator_factory, X: np.ndarray, y: np.ndarray) -> Any:
    """Safely fit a binary estimator, including single-class targets."""
    unique = np.unique(y)
    if len(unique) <= 1:
        const_val = int(unique[0]) if len(unique) == 1 else 0
        return _ConstantClassifier(const_val)
    est = estimator_factory()
    est.fit(X, y)
    return est


class GSIMLCPAv6_3_3Classifier(BaseEstimator, ClassifierMixin):
    """Minimal 3-hyperparameter GSI-MLC-PA with Anchor/Contextual cascade, unified calibration, and closed-form Bayes abstention."""

    def __init__(
        self,
        base_learner: str = "logistic",
        cost: float = 0.30,
        anchor_threshold: float = 0.75,
        gamma_min: float = 0.70,
        cv_folds: int = 5,
        abstain_value: int = -1,
        random_state: int = 42,
    ):
        self.base_learner = base_learner
        self.cost = float(cost)
        self.anchor_threshold = float(anchor_threshold)
        self.gamma_min = float(gamma_min)
        self.cv_folds = int(cv_folds)
        self.abstain_value = int(abstain_value)
        self.random_state = int(random_state)
        self.is_fitted_ = False

    def _create_binary_model(self):
        """Create the configured binary estimator."""
        return create_binary_estimator(self.base_learner, random_state=self.random_state)

    @staticmethod
    def _predict_prob_single(estimator: Any, X_eval: np.ndarray) -> np.ndarray:
        """Extract positive-class probabilities from a fitted binary estimator."""
        if hasattr(estimator, "predict_proba"):
            probs = np.asarray(estimator.predict_proba(X_eval), dtype=np.float64)
            if probs.ndim == 2:
                return np.clip(probs[:, 1], 0.0, 1.0).astype(np.float32)
            return np.clip(probs.ravel(), 0.0, 1.0).astype(np.float32)
        if hasattr(estimator, "decision_function"):
            scores = np.asarray(estimator.decision_function(X_eval), dtype=np.float64)
            return np.clip(1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0))), 0.0, 1.0).astype(np.float32)
        preds = np.asarray(estimator.predict(X_eval), dtype=np.float64)
        return np.clip(preds.ravel(), 0.0, 1.0).astype(np.float32)

    # Read-only compatibility aliases for audit/legacy callers
    @property
    def stratified_threshold(self) -> float:
        return self.anchor_threshold

    @property
    def independent_layers_(self) -> List[List[int]]:
        return self.anchor_layers_

    @property
    def independent_labels_(self) -> List[int]:
        return self.anchor_labels_

    @property
    def dependent_labels_(self) -> List[int]:
        return self.contextual_labels_

    def predict_proba(self, X) -> np.ndarray:
        """Predict calibrated probabilities using the Anchor-to-Contextual cascade."""
        if not self.is_fitted_:
            raise ValueError("GSIMLCPAv6_3_3Classifier is not fitted yet.")

        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        P_final = np.zeros((n_samples, self.n_labels_), dtype=np.float32)

        # STAGE 1: Predict Anchor Labels (AL) directly from X
        al_test_probs: List[np.ndarray] = []
        for lbl in self.anchor_labels_:
            p_l = self._predict_prob_single(self.anchor_models_[lbl], X)
            P_final[:, lbl] = p_l
            al_test_probs.append(p_l)

        if al_test_probs:
            p_centered = 2.0 * np.column_stack(al_test_probs) - 1.0
            X_context_test = np.hstack([X, p_centered.astype(np.float32)])
        else:
            X_context_test = X

        # STAGE 2: Predict Contextual Labels (CL) on [X, 2*P_hat(AL) - 1]
        for lbl in self.contextual_labels_:
            P_final[:, lbl] = self._predict_prob_single(
                self.contextual_models_[lbl], X_context_test
            )

        # STAGE 3: Apply unified tail probability calibration
        if self.calibrator_ is not None:
            P_final = self.calibrator_.transform(P_final)

        return np.clip(P_final, 0.0, 1.0).astype(np.float32)

    def _compute_thresholds_for_cost(
        self, Y_train: np.ndarray, oof_probs: np.ndarray, cost: float
    ) -> Dict[int, Dict[str, Any]]:
        return compute_tri_regime_adaptive_thresholds_table(
            Y_train=Y_train,
            OOF_probs=oof_probs,
            cost=float(cost),
            gamma_min=self.gamma_min,
        )

    def fit(self, X, Y):
        """Fit Single-Stage Anchor/Contextual cascade, unified calibrator, and Bayes thresholds."""
        if not (0.0 < float(self.cost) < 0.5):
            raise ValueError(f"cost must lie in (0, 0.5), got {self.cost}")
        if not (0.0 <= float(self.anchor_threshold) <= 2.0):
            raise ValueError(f"anchor_threshold must lie in [0.0, 2.0], got {self.anchor_threshold}")
        if not (0.0 <= float(self.gamma_min) <= 1.0):
            raise ValueError(f"gamma_min must lie in [0.0, 1.0], got {self.gamma_min}")
        if int(self.cv_folds) < 2:
            raise ValueError(f"cv_folds must be >= 2, got {self.cv_folds}")

        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        Y = np.asarray(Y, dtype=np.int32)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D feature matrix, got shape {X.shape}")
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D binary matrix, got shape {Y.shape}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Sample count mismatch: X has {X.shape[0]}, Y has {Y.shape[0]}"
            )

        n_samples, n_labels = Y.shape
        self.n_samples_ = n_samples
        self.n_labels_ = n_labels
        self.n_features_in_ = X.shape[1]

        # -------------------------------------------------------------------------
        # PHASE 1: Single-Stage K-Fold OOF Anchor Label Discovery
        # -------------------------------------------------------------------------
        stage1_oof: Dict[int, np.ndarray] = {}
        self.anchor_labels_: List[int] = []
        self.contextual_labels_: List[int] = []

        for lbl in range(n_labels):
            score, oof_p, _ = evaluate_label_5fold_cv(
                X=X,
                y=Y[:, lbl],
                estimator_factory=self._create_binary_model,
                n_folds=self.cv_folds,
                cost=self.cost,
                metric="selective_f1",
                random_state=self.random_state + 1,
            )
            stage1_oof[lbl] = oof_p
            if score >= self.anchor_threshold:
                self.anchor_labels_.append(lbl)
            else:
                self.contextual_labels_.append(lbl)

        self.anchor_layers_: List[List[int]] = (
            [list(self.anchor_labels_)] if self.anchor_labels_ else []
        )

        self.anchor_models_: Dict[int, Any] = {}
        P_OOF_all = np.zeros((n_samples, n_labels), dtype=np.float64)

        for lbl in self.anchor_labels_:
            self.anchor_models_[lbl] = _fit_binary_model_safe(
                self._create_binary_model, X, Y[:, lbl]
            )
            P_OOF_all[:, lbl] = stage1_oof[lbl]

        self.anchor_layer_models_ = (
            [dict(self.anchor_models_)] if self.anchor_models_ else []
        )

        # -------------------------------------------------------------------------
        # PHASE 2: Zero-Centered Contextual Label Inference
        # -------------------------------------------------------------------------
        self.contextual_models_: Dict[int, Any] = {}
        if self.anchor_labels_:
            oof_al = np.column_stack([stage1_oof[lbl] for lbl in self.anchor_labels_])
            oof_centered = (2.0 * oof_al - 1.0).astype(np.float32)
            X_context = np.hstack([X, oof_centered])

            for lbl in self.contextual_labels_:
                _, oof_p, _ = evaluate_label_5fold_cv(
                    X=X_context,
                    y=Y[:, lbl],
                    estimator_factory=self._create_binary_model,
                    n_folds=self.cv_folds,
                    cost=self.cost,
                    metric="selective_f1",
                    random_state=self.random_state + lbl,
                )
                P_OOF_all[:, lbl] = oof_p
                self.contextual_models_[lbl] = _fit_binary_model_safe(
                    self._create_binary_model, X_context, Y[:, lbl]
                )
        else:
            # No Anchor Labels promoted: X_context == X, reuse Stage-1 OOF directly!
            for lbl in self.contextual_labels_:
                P_OOF_all[:, lbl] = stage1_oof[lbl]
                self.contextual_models_[lbl] = _fit_binary_model_safe(
                    self._create_binary_model, X, Y[:, lbl]
                )

        # -------------------------------------------------------------------------
        # PHASE 3: Unified Smooth Balanced-Root Probability Calibration
        # -------------------------------------------------------------------------
        self.calibrator_ = MultiLabelAdaptiveCalibrator(random_state=self.random_state)
        self.calibrator_.fit(P_OOF_all, Y)
        P_OOF_all = self.calibrator_.transform(P_OOF_all)

        self.oof_probabilities_all_ = P_OOF_all
        self._y_train_cache_ = Y.copy()

        # -------------------------------------------------------------------------
        # PHASE 4: Closed-Form Prior-Adjusted Bayes Thresholds + OOF Coverage Guard
        # -------------------------------------------------------------------------
        self.adaptive_thresholds_ = self._compute_thresholds_for_cost(
            Y_train=Y,
            oof_probs=P_OOF_all,
            cost=self.cost,
        )

        self.is_fitted_ = True
        return self

    def predict(self, X, cost: Optional[float] = None) -> np.ndarray:
        """Predict multilabel targets with partial abstention in {-1, 0, 1}."""
        P = self.predict_proba(X)
        c = float(cost if cost is not None else self.cost)
        if cost is not None and c >= 0.5:
            return self.predict_full(X)
        if cost is not None and not np.isclose(c, self.cost) and 0.0 < c < 0.5:
            thresholds = self._compute_thresholds_for_cost(
                Y_train=self._y_train_cache_,
                oof_probs=self.oof_probabilities_all_,
                cost=c,
            )
        else:
            thresholds = self.adaptive_thresholds_
        return apply_tri_regime_abstention(
            P, thresholds, abstain_value=self.abstain_value
        )

    def predict_full(self, X, threshold: Optional[float] = None) -> np.ndarray:
        """Predict complete binary multilabel decisions without abstention."""
        P = self.predict_proba(X)
        th = 0.50 if threshold is None else float(threshold)
        return (P >= th).astype(np.int32)
