"""GSI-MLC-PA Classifier with Anchor/Contextual Cascade and Unified Calibration.

Architecture:
1. Phase 1 (Anchor Label Discovery & Cascade):
   Partitions labels via 5-fold CV peeling into high-confidence Anchor Labels (AL)
   predicted directly/sequentially from features, and Contextual Labels (CL).
2. Phase 2 (Contextual Label Inference):
   Predicts Contextual Labels conditioned on [X, Normalize(P_hat(AL))] without
   One-Step Normalized Mean-Field (OS-NMF) or residual correlation heuristics.
3. Phase 3 (Unified Identity-Regularized Balanced-Root Calibration):
   Calibrates OOF and test probabilities smoothly across all class-imbalance regimes.
4. Phase 4 (Adaptive Tri-Regime Decision & Dual Coverage Guard):
   Computes per-label abstention thresholds [tau_0, tau_1] in {-1, 0, 1}.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .base_learners import create_binary_estimator
from .binary_relevance import _ConstantClassifier
from ..selection.cv_peeling import (
    CVPeelingConfig,
    CVStratifiedPeelingSelector,
    evaluate_label_5fold_cv,
)
from ..selection.stratified_peeling import normalize_augmented_probabilities
from ..selection.tri_regime_decision import (
    apply_tri_regime_abstention,
    compute_label_balance_factor,
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
    """GSI-MLC-PA with Anchor/Contextual cascade, unified tail calibration, and Tri-Regime inference."""

    def __init__(
        self,
        base_learner: str = "logistic",
        stratified_threshold: float = 0.75,
        cv_folds: int = 5,
        max_peeling_depth: int = 3,
        peeling_metric: str = "selective_f1",
        decaying_threshold: bool = True,
        threshold_decay_step: float = 0.05,
        aug_normalization: str = "matching",
        cost: float = 0.30,
        gamma_min: float = 0.70,
        use_prior_adaptive: bool = True,
        calibrate_tail: bool = True,
        calibration_method: str = "sqrt_platt",
        precision_guard: bool = True,
        min_tau_1: float = 0.50,
        abstain_value: int = -1,
        random_state: int = 42,
        tau_balance: float = 0.85,
        blend_kappa: float = 20.0,
        tau_1_cap: float = 0.55,
        tau_0_floor: float = 0.45,
        separation_aware: bool = False,
    ):
        self.base_learner = base_learner
        self.stratified_threshold = float(stratified_threshold)
        self.cv_folds = int(cv_folds)
        self.max_peeling_depth = int(max_peeling_depth)
        self.peeling_metric = str(peeling_metric)
        self.decaying_threshold = bool(decaying_threshold)
        self.threshold_decay_step = float(threshold_decay_step)
        self.aug_normalization = str(aug_normalization)
        self.cost = float(cost)
        self.gamma_min = float(gamma_min)
        self.use_prior_adaptive = bool(use_prior_adaptive)
        self.calibrate_tail = bool(calibrate_tail)
        self.calibration_method = str(calibration_method)
        self.precision_guard = bool(precision_guard)
        self.min_tau_1 = float(min_tau_1)
        self.abstain_value = int(abstain_value)
        self.random_state = int(random_state)
        self.tau_balance = float(tau_balance)
        self.blend_kappa = float(blend_kappa)
        self.tau_1_cap = float(tau_1_cap)
        self.tau_0_floor = float(tau_0_floor)
        self.separation_aware = bool(separation_aware)
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

    def _apply_symmetric_chow_rejection(self, probabilities, cost):
        predictions = np.full(probabilities.shape, self.abstain_value, dtype=np.int32)
        predictions[probabilities <= cost] = 0
        predictions[probabilities >= 1.0 - cost] = 1
        return predictions

    # Read-only compatibility aliases for legacy callers
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
        """Predict calibrated probabilities using Anchor-to-Contextual cascade."""
        if not self.is_fitted_:
            raise ValueError("GSIMLCPAv6_3_3Classifier is not fitted yet.")

        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        P_final = np.zeros((n_samples, self.n_labels_), dtype=np.float32)

        # STAGE 1: Predict Anchor Labels (AL) across discovered layers
        current_test_X = X
        accumulated_al_test_probs: List[np.ndarray] = []

        for layer_idx, layer in enumerate(self.anchor_layers_):
            layer_models = self.anchor_layer_models_[layer_idx]
            for lbl in layer:
                model = layer_models[lbl]
                p_l = self._predict_prob_single(model, current_test_X)
                P_final[:, lbl] = p_l
                accumulated_al_test_probs.append(p_l)

            if accumulated_al_test_probs:
                p_raw = np.column_stack(accumulated_al_test_probs)
                p_norm = normalize_augmented_probabilities(
                    p_raw, reference_X=X, strategy=self.aug_normalization
                )
                current_test_X = np.hstack([X, p_norm])

        X_context_test = current_test_X

        # STAGE 2: Predict Contextual Labels (CL) directly on [X, Normalize(P_AL)]
        for lbl in self.contextual_labels_:
            P_final[:, lbl] = self._predict_prob_single(
                self.contextual_models_[lbl], X_context_test
            )

        # STAGE 3: Apply unified tail probability calibration
        if self.calibrate_tail and self.calibrator_ is not None:
            P_final = self.calibrator_.transform(P_final)

        return np.clip(P_final, 0.0, 1.0).astype(np.float32)

    def _compute_thresholds_for_cost(
        self, Y_train: np.ndarray, oof_probs: np.ndarray, cost: float, in_sample_probs=None
    ) -> Dict[int, Dict[str, Any]]:
        return compute_tri_regime_adaptive_thresholds_table(
            Y_train=Y_train,
            OOF_probs=oof_probs,
            cost=float(cost),
            tau_balance=self.tau_balance,
            blend_kappa=self.blend_kappa,
            gamma_min=self.gamma_min,
            precision_guard=self.precision_guard,
            tau_1_cap=self.tau_1_cap,
            tau_0_floor=self.tau_0_floor,
            in_sample_probs=in_sample_probs,
            separation_aware=self.separation_aware,
        )

    def fit(self, X, Y):
        """Fit Anchor/Contextual cascade, unified tail calibrator, and Tri-Regime thresholds."""
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)

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
        # PHASE 1: Layered CV Peeling to discover Anchor Labels (AL)
        # -------------------------------------------------------------------------
        peeling_cfg = CVPeelingConfig(
            threshold=self.stratified_threshold,
            n_folds=self.cv_folds,
            max_depth=self.max_peeling_depth,
            cost=self.cost,
            metric=self.peeling_metric,
            decaying_threshold=self.decaying_threshold,
            threshold_decay_step=self.threshold_decay_step,
            aug_normalization=self.aug_normalization,
            min_contextual_labels=0,
            random_state=self.random_state,
        )

        selector = CVStratifiedPeelingSelector(
            config=peeling_cfg,
            base_estimator_factory=self._create_binary_model,
        )

        self.peeling_result_ = selector.fit_partition(X, Y)
        self.anchor_layers_ = [list(layer) for layer in self.peeling_result_.anchor_layers]
        self.anchor_labels_ = list(self.peeling_result_.anchor_labels)
        self.contextual_labels_ = list(self.peeling_result_.contextual_labels)

        # Fit dedicated binary models for each Anchor Label layer
        self.anchor_layer_models_ = []
        current_train_X = X
        accumulated_al_labels: List[int] = []
        P_OOF_all = np.zeros((n_samples, n_labels), dtype=np.float64)

        for layer in self.anchor_layers_:
            layer_dict: Dict[int, Any] = {}
            for lbl in layer:
                est = _fit_binary_model_safe(self._create_binary_model, current_train_X, Y[:, lbl])
                layer_dict[lbl] = est
                if self.peeling_result_.oof_probabilities and lbl in self.peeling_result_.oof_probabilities:
                    P_OOF_all[:, lbl] = self.peeling_result_.oof_probabilities[lbl]
            self.anchor_layer_models_.append(layer_dict)

            accumulated_al_labels.extend(layer)
            if accumulated_al_labels and self.peeling_result_.oof_probabilities:
                oof_cols = [
                    self.peeling_result_.oof_probabilities[l]
                    for l in accumulated_al_labels
                    if l in self.peeling_result_.oof_probabilities
                ]
                if oof_cols:
                    oof_raw = np.column_stack(oof_cols)
                    oof_norm = normalize_augmented_probabilities(
                        oof_raw, reference_X=X, strategy=self.aug_normalization
                    )
                    current_train_X = np.hstack([X, oof_norm])

        X_context = current_train_X

        # -------------------------------------------------------------------------
        # PHASE 2: Fit Contextual Labels (CL) on [X, Normalize(P_OOF_AL)]
        # -------------------------------------------------------------------------
        self.contextual_models_ = {}
        for lbl in self.contextual_labels_:
            _, oof_p, _ = evaluate_label_5fold_cv(
                X=X_context,
                y=Y[:, lbl],
                estimator_factory=self._create_binary_model,
                n_folds=self.cv_folds,
                cost=self.cost,
                metric=self.peeling_metric,
                random_state=self.random_state + lbl,
            )
            P_OOF_all[:, lbl] = oof_p
            self.contextual_models_[lbl] = _fit_binary_model_safe(
                self._create_binary_model, X_context, Y[:, lbl]
            )

        # -------------------------------------------------------------------------
        # PHASE 3: Unified Smooth Balanced-Root Probability Calibration
        # -------------------------------------------------------------------------
        self.calibrator_: Optional[MultiLabelAdaptiveCalibrator] = None
        if self.calibrate_tail:
            self.calibrator_ = MultiLabelAdaptiveCalibrator(
                tau_balance=self.tau_balance,
                method=self.calibration_method,
                random_state=self.random_state,
            )
            self.calibrator_.fit(P_OOF_all, Y)
            P_OOF_all = self.calibrator_.transform(P_OOF_all)

        self.oof_probabilities_all_ = P_OOF_all
        self._y_train_cache_ = Y.copy()

        # -------------------------------------------------------------------------
        # PHASE 4: Adaptive Tri-Regime Thresholds Table with Dual Coverage Guard
        # -------------------------------------------------------------------------
        if self.use_prior_adaptive:
            self.is_fitted_ = True
            P_fitted = self.predict_proba(X) if self.separation_aware else None
            self.adaptive_thresholds_ = self._compute_thresholds_for_cost(
                Y_train=Y,
                oof_probs=P_OOF_all,
                cost=self.cost,
                in_sample_probs=P_fitted,
            )
        else:
            self.adaptive_thresholds_ = {
                j: {
                    "tau_0": float(self.cost),
                    "tau_1": float(1.0 - self.cost),
                    "prior": float(np.mean(Y[:, j])),
                    "beta": float(compute_label_balance_factor(np.mean(Y[:, j]))),
                    "regime": "Symmetric",
                    "blend_weight": 1.0,
                    "coverage_guard": False,
                    "empirical_coverage": 1.0,
                }
                for j in range(n_labels)
            }

        self.is_fitted_ = True
        return self

    def predict(self, X, cost: Optional[float] = None) -> np.ndarray:
        """Predict multilabel targets with partial abstention in {-1, 0, 1}."""
        P = self.predict_proba(X)
        c = float(cost if cost is not None else self.cost)
        if self.use_prior_adaptive and hasattr(self, "adaptive_thresholds_"):
            if cost is not None and not np.isclose(c, self.cost) and 0.0 < c < 0.5:
                thresholds = self._compute_thresholds_for_cost(
                    Y_train=self._y_train_cache_,
                    oof_probs=self.oof_probabilities_all_,
                    cost=c,
                )
            elif cost is not None and c >= 0.5:
                return self.predict_full(X)
            else:
                thresholds = self.adaptive_thresholds_
            return apply_tri_regime_abstention(
                P, thresholds, abstain_value=self.abstain_value
            )
        return self._apply_symmetric_chow_rejection(P, cost=c)

    def predict_full(self, X, threshold: Optional[float] = None) -> np.ndarray:
        """Predict complete binary multilabel decisions without abstention."""
        P = self.predict_proba(X)
        if threshold is not None:
            return (P >= float(threshold)).astype(np.int32)
        if not hasattr(self, "adaptive_thresholds_"):
            return (P >= 0.5).astype(np.int32)

        N, K = P.shape
        preds = np.zeros((N, K), dtype=np.int32)
        for j in range(K):
            th = self.adaptive_thresholds_[j]
            pi = th.get("prior", 0.5)
            if pi <= 0.20:
                t = th["tau_1"]
            elif pi >= 0.80:
                t = th["tau_0"]
            else:
                t = 0.50
            preds[:, j] = (P[:, j] >= t).astype(np.int32)
        return preds
