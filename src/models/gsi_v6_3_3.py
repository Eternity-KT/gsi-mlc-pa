"""GSI-MLC-PA v6.3.3 final classifier."""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .base_learners import create_binary_estimator
from .binary_relevance import _ConstantClassifier
from ..selection.cv_peeling import CVPeelingConfig, CVStratifiedPeelingSelector, evaluate_label_5fold_cv
from ..selection.stratified_peeling import normalize_augmented_probabilities
from ..selection.br_residual_correlation import (
    build_residual_dependency_graph, compute_br_residual_matrix,
    compute_residual_pcc_matrix, export_residual_correlation_audit_table,
    one_step_normalized_mean_field,
)
from ..selection.tri_regime_decision import (
    apply_tri_regime_abstention, compute_label_balance_factor,
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
    """GSI-MLC-PA v6.3.3 with adaptive tri-regime inference and OS-NMF."""

    def __init__(
        self, base_learner: str = "logistic", stratified_threshold: float = 0.75,
        residual_corr_threshold: float = 0.25, cv_folds: int = 5,
        max_peeling_depth: int = 3, peeling_metric: str = "selective_f1",
        decaying_threshold: bool = True, threshold_decay_step: float = 0.05,
        aug_normalization: str = "matching", error_format: str = "residual",
        cost: float = 0.30, gamma_min: float = 0.70,
        use_prior_adaptive: bool = True, calibrate_tail: bool = True,
        calibration_method: str = "sqrt_platt", precision_guard: bool = True,
        min_tau_1: float = 0.50, abstain_value: int = -1,
        random_state: int = 42, mf_alpha: float = 0.25, mf_z_max: float = 0.5,
        tau_balance: float = 0.85, blend_kappa: float = 20.0,
        tau_1_cap: float = 0.55, tau_0_floor: float = 0.45,
        separation_aware: bool = False,
    ):
        self.base_learner = base_learner
        self.stratified_threshold = float(stratified_threshold)
        self.residual_corr_threshold = float(residual_corr_threshold)
        self.cv_folds = int(cv_folds)
        self.max_peeling_depth = int(max_peeling_depth)
        self.peeling_metric = str(peeling_metric)
        self.decaying_threshold = bool(decaying_threshold)
        self.threshold_decay_step = float(threshold_decay_step)
        self.aug_normalization = str(aug_normalization)
        self.error_format = str(error_format)
        self.cost = float(cost)
        self.gamma_min = float(gamma_min)
        self.use_prior_adaptive = bool(use_prior_adaptive)
        self.calibrate_tail = bool(calibrate_tail)
        self.calibration_method = str(calibration_method)
        self.precision_guard = bool(precision_guard)
        self.min_tau_1 = float(min_tau_1)
        self.abstain_value = int(abstain_value)
        self.random_state = int(random_state)
        self.mf_alpha = float(mf_alpha)
        self.mf_z_max = float(mf_z_max)
        self.tau_balance = float(tau_balance)
        self.blend_kappa = float(blend_kappa)
        self.tau_1_cap = float(tau_1_cap)
        self.tau_0_floor = float(tau_0_floor)
        self.separation_aware = bool(separation_aware)
        self.is_fitted_ = False
        self.signed_residual_corr_matrix_ = None

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



    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities on test samples using One-Step Mean-Field on DL."""
        if not self.is_fitted_:
            raise ValueError("GSIMLCPAv6_3_3Classifier is not fitted yet.")

        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        P_final = np.zeros((n_samples, self.n_labels_), dtype=np.float32)

        # STAGE 1: Predict Independent Labels across layers
        current_test_X = X
        accumulated_il_test_probs: List[np.ndarray] = []

        for layer_idx, layer in enumerate(self.independent_layers_):
            layer_models = self.br_il_layer_models_[layer_idx]
            for lbl in layer:
                model = layer_models[lbl]
                p_l = self._predict_prob_single(model, current_test_X)
                P_final[:, lbl] = p_l
                accumulated_il_test_probs.append(p_l)

            if accumulated_il_test_probs:
                p_raw = np.column_stack(accumulated_il_test_probs)
                p_norm = normalize_augmented_probabilities(
                    p_raw, reference_X=X, strategy=self.aug_normalization
                )
                current_test_X = np.hstack([X, p_norm])

        FS_context_test = current_test_X

        # STAGE 2: Predict Dependent Labels via Base BR + Mean-Field
        dl = self.dependent_labels_
        if len(dl) > 0:
            P_base_DL = np.column_stack([
                self._predict_prob_single(self.br_base_dl_models_[lbl], FS_context_test)
                for lbl in dl
            ])

            if len(dl) >= 2 and self.signed_residual_corr_matrix_ is not None:
                P_refined_DL = one_step_normalized_mean_field(
                    P_base=P_base_DL,
                    corr_matrix=self.signed_residual_corr_matrix_,
                    alpha=self.mf_alpha,
                    threshold=self.residual_corr_threshold,
                    z_max=self.mf_z_max,
                )
                P_final[:, dl] = P_refined_DL
            else:
                P_final[:, dl] = P_base_DL

        # Apply tail calibration
        if self.calibrate_tail and self.calibrator_ is not None:
            P_final = self.calibrator_.transform(P_final)

        return np.clip(P_final, 0.0, 1.0).astype(np.float32)

    def fit(self, X, Y):
        """Fit GSI-MLC-PA v6.3.3 using CV peeling, OS-NMF coupling, and Tri-Regime calibration."""
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
        # PHASE 1: Layered Peeling to discover Independent Labels (IL)
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
            promote_singleton_dl=True,
            random_state=self.random_state,
        )

        selector = CVStratifiedPeelingSelector(
            config=peeling_cfg,
            base_estimator_factory=self._create_binary_model,
        )

        self.peeling_result_ = selector.fit_partition(X, Y)
        self.independent_layers_ = self.peeling_result_.independent_layers
        self.independent_labels_ = list(self.peeling_result_.all_independent_labels)
        self.dependent_labels_ = list(self.peeling_result_.dependent_residual_labels)

        # Train dedicated BR models for each layer in IL
        self.br_il_layer_models_ = []
        current_train_X = X
        accumulated_il_labels: List[int] = []
        P_OOF_all = np.zeros((n_samples, n_labels), dtype=np.float64)

        for layer_idx, layer in enumerate(self.independent_layers_):
            layer_dict: Dict[int, Any] = {}
            for lbl in layer:
                est = _fit_binary_model_safe(self._create_binary_model, current_train_X, Y[:, lbl])
                layer_dict[lbl] = est
                if self.peeling_result_.oof_probabilities and lbl in self.peeling_result_.oof_probabilities:
                    P_OOF_all[:, lbl] = self.peeling_result_.oof_probabilities[lbl]
            self.br_il_layer_models_.append(layer_dict)

            accumulated_il_labels.extend(layer)
            if accumulated_il_labels and self.peeling_result_.oof_probabilities:
                oof_cols = [
                    self.peeling_result_.oof_probabilities[l]
                    for l in accumulated_il_labels
                    if l in self.peeling_result_.oof_probabilities
                ]
                if oof_cols:
                    oof_raw = np.column_stack(oof_cols)
                    oof_norm = normalize_augmented_probabilities(
                        oof_raw, reference_X=X, strategy=self.aug_normalization
                    )
                    current_train_X = np.hstack([X, oof_norm])

        FS_context = current_train_X

        # -------------------------------------------------------------------------
        # PHASE 2: Handle Residual Dependent Labels (DL) via One-Step Mean-Field
        # -------------------------------------------------------------------------
        self.br_base_dl_models_ = {}
        self.br_cond_dl_models_ = {}
        self.dependency_graph_ = {lbl: [] for lbl in self.dependent_labels_}
        self.dl_error_rates_ = {}
        self.residual_corr_matrix_ = None
        self.signed_residual_corr_matrix_ = None

        n_dl = len(self.dependent_labels_)
        if n_dl >= 2:
            P_OOF_DL = np.zeros((n_samples, n_dl), dtype=np.float64)

            for idx_dl, lbl in enumerate(self.dependent_labels_):
                _, oof_p, diag = evaluate_label_5fold_cv(
                    X=FS_context,
                    y=Y[:, lbl],
                    estimator_factory=self._create_binary_model,
                    n_folds=self.cv_folds,
                    cost=self.cost,
                    metric=self.peeling_metric,
                    random_state=self.random_state + lbl,
                )
                P_OOF_DL[:, idx_dl] = oof_p
                P_OOF_all[:, lbl] = oof_p

                base_est = _fit_binary_model_safe(self._create_binary_model, FS_context, Y[:, lbl])
                self.br_base_dl_models_[lbl] = base_est
                self.br_cond_dl_models_[lbl] = base_est
                self.dl_error_rates_[lbl] = float(np.mean(np.abs(Y[:, lbl] - oof_p)))

            residuals_dl = compute_br_residual_matrix(
                Y_true=Y[:, self.dependent_labels_],
                P_prob=P_OOF_DL,
                error_format=self.error_format,
            )

            self.signed_residual_corr_matrix_ = compute_residual_pcc_matrix(
                residual_matrix=residuals_dl,
                use_absolute=False,
            )
            self.residual_corr_matrix_ = np.abs(self.signed_residual_corr_matrix_)

            self.dependency_graph_ = build_residual_dependency_graph(
                corr_matrix=self.residual_corr_matrix_,
                dl_labels=self.dependent_labels_,
                threshold=self.residual_corr_threshold,
            )

            # Refine OOF predictions directly via One-Step Normalized Mean-Field
            refined_oof_dl = one_step_normalized_mean_field(
                P_base=P_OOF_DL,
                corr_matrix=self.signed_residual_corr_matrix_,
                alpha=self.mf_alpha,
                threshold=self.residual_corr_threshold,
                z_max=self.mf_z_max,
            )
            P_OOF_all[:, self.dependent_labels_] = refined_oof_dl

            self.audit_table_ = export_residual_correlation_audit_table(
                corr_matrix=self.residual_corr_matrix_,
                dl_labels=self.dependent_labels_,
                dependency_graph=self.dependency_graph_,
                error_rates=self.dl_error_rates_,
            )

        elif n_dl == 1:
            single_lbl = self.dependent_labels_[0]
            est = _fit_binary_model_safe(self._create_binary_model, FS_context, Y[:, single_lbl])
            self.br_base_dl_models_[single_lbl] = est
            self.br_cond_dl_models_[single_lbl] = est
            self.dependency_graph_[single_lbl] = []

            _, oof_p_single, _ = evaluate_label_5fold_cv(
                X=FS_context,
                y=Y[:, single_lbl],
                estimator_factory=self._create_binary_model,
                n_folds=self.cv_folds,
                cost=self.cost,
                metric=self.peeling_metric,
                random_state=self.random_state + single_lbl,
            )
            P_OOF_all[:, single_lbl] = oof_p_single

        # -------------------------------------------------------------------------
        # PHASE 2.5: Adaptive Tri-Regime Probability Calibration
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

        # -------------------------------------------------------------------------
        # PHASE 3: Adaptive Tri-Regime Thresholds Table with Smooth Blending
        # -------------------------------------------------------------------------
        if self.use_prior_adaptive:
            self.is_fitted_ = True
            P_fitted = self.predict_proba(X) if self.separation_aware else None
            self.adaptive_thresholds_ = compute_tri_regime_adaptive_thresholds_table(
                Y_train=Y,
                OOF_probs=P_OOF_all,
                cost=self.cost,
                tau_balance=self.tau_balance,
                blend_kappa=self.blend_kappa,
                gamma_min=self.gamma_min,
                precision_guard=self.precision_guard,
                tau_1_cap=self.tau_1_cap,
                tau_0_floor=self.tau_0_floor,
                in_sample_probs=P_fitted,
                separation_aware=self.separation_aware,
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
        """
        Predict multilabel targets with partial abstention using Tri-Regime thresholds.
        """
        P = self.predict_proba(X)
        if self.use_prior_adaptive and hasattr(self, "adaptive_thresholds_"):
            return apply_tri_regime_abstention(
                P, self.adaptive_thresholds_, abstain_value=self.abstain_value
            )
        c = float(cost if cost is not None else self.cost)
        return self._apply_symmetric_chow_rejection(P, cost=c)

    def predict_full(self, X, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict binary multilabel decisions without abstention.
        If threshold is specified as float, applies static threshold across all labels.
        If threshold is None (default), applies label-specific adaptive decision thresholds.
        """
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
