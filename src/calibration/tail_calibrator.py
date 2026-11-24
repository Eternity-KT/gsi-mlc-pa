"""Unified Smooth Identity-Regularized Balanced-Root Platt Calibrator.

Fixes all five calibration defects identified in legacy v6.3.3:
1. Identity-centered regularization: penalizes (a - 1)^2 and b^2 instead of
   liblinear's shrinkage of (a, b) toward (0, 0), preserving logit separation.
2. Mean-normalized sample weights (mean(w) == 1): keeps regularization scale
   invariant to dataset size and class-imbalance ratio.
3. Rare-label overfitting guard (min_minority_samples = 3): locks slope a = 1.0
   and estimates only a regularized intercept shift when minority support < 3.
4. Smooth unified weighting w_pos / w_neg = sqrt(N_0 / N_1) across all priors
   pi in (0, 1): reduces smoothly to unweighted Platt at pi = 0.50 (calibrating
   SVM/MLP in balanced regimes) and satisfies exact dual symmetry
   f(1 - p; 1 - y) = 1 - f(p; y) without separate Inverted/Identity classes.
5. Strictly convex L-BFGS-B optimization with analytical gradients and positive
   slope bounds, guaranteeing monotonicity by construction.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


VALID_CALIBRATION_METHODS = (
    "sqrt_platt",
    "balanced_root",
    "weighted_platt",
    "linear",
    "standard_platt",
    "unweighted",
)


class TailCalibrator:
    """Monotonic 1D probability calibrator with smooth balanced-root weighting."""

    def __init__(
        self,
        method: str = "sqrt_platt",
        clip_eps: float = 1e-5,
        reg_slope: float = 0.2,
        reg_intercept: float = 0.02,
        min_minority_samples: int = 3,
        slope_bounds: Tuple[float, float] = (0.05, 10.0),
        intercept_bound: float = 15.0,
        random_state: int = 42,
    ):
        if method not in VALID_CALIBRATION_METHODS:
            raise ValueError(
                f"Unknown calibration method '{method}'. "
                f"Available: {VALID_CALIBRATION_METHODS}."
            )
        if not (0.0 < float(clip_eps) < 0.5):
            raise ValueError(f"clip_eps must lie in (0, 0.5), got {clip_eps}.")
        if float(reg_slope) < 0.0 or float(reg_intercept) < 0.0:
            raise ValueError("Regularization strengths must be non-negative.")
        if int(min_minority_samples) < 1:
            raise ValueError("min_minority_samples must be >= 1.")
        min_a, max_a = float(slope_bounds[0]), float(slope_bounds[1])
        if min_a <= 0.0 or min_a >= max_a:
            raise ValueError(
                f"slope_bounds must satisfy 0 < min_slope < max_slope, got {slope_bounds}."
            )

        self.method = str(method)
        self.clip_eps = float(clip_eps)
        self.reg_slope = float(reg_slope)
        self.reg_intercept = float(reg_intercept)
        self.min_minority_samples = int(min_minority_samples)
        self.slope_bounds = (min_a, max_a)
        self.intercept_bound = float(intercept_bound)
        self.random_state = int(random_state)

        self.is_fitted_: bool = False
        self.is_degenerate_: bool = False
        self.degenerate_value_: float = 0.0
        self.is_low_support_: bool = False
        self.a_: float = 1.0
        self.b_: float = 0.0

    def _compute_normalized_weights(
        self, labels: np.ndarray, n_pos: int, n_neg: int
    ) -> np.ndarray:
        """Compute mean-normalized class weights satisfying exact 0/1 symmetry."""
        if self.method in ("standard_platt", "unweighted"):
            weights = np.ones(len(labels), dtype=np.float64)
        elif self.method in ("sqrt_platt", "balanced_root"):
            pos_ratio = np.sqrt(float(n_neg) / float(n_pos))
            weights = np.where(labels == 1, pos_ratio, 1.0).astype(np.float64)
        else:
            pos_ratio = float(n_neg) / float(n_pos)
            weights = np.where(labels == 1, pos_ratio, 1.0).astype(np.float64)

        return weights / np.mean(weights)

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "TailCalibrator":
        p = np.asarray(probs, dtype=np.float64).ravel()
        labels = np.asarray(y, dtype=np.int32).ravel()

        if len(p) != len(labels):
            raise ValueError(f"Shape mismatch: {len(p)} vs {len(labels)}")
        if len(labels) == 0:
            raise ValueError("Cannot fit TailCalibrator on an empty array.")
        if not np.all(np.isin(labels, (0, 1))):
            raise ValueError("Target y must contain only binary values {0, 1}.")

        n_pos = int(np.sum(labels == 1))
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            self.is_fitted_ = True
            self.is_degenerate_ = True
            self.is_low_support_ = False
            self.degenerate_value_ = 1.0 if n_pos > 0 else 0.0
            self.a_ = 1.0
            self.b_ = 0.0
            return self

        self.is_degenerate_ = False
        p_clipped = np.clip(p, self.clip_eps, 1.0 - self.clip_eps)
        z = np.log(p_clipped / (1.0 - p_clipped))
        y_float = labels.astype(np.float64)
        weights = self._compute_normalized_weights(labels, n_pos, n_neg)
        b_lim = self.intercept_bound

        if min(n_pos, n_neg) < self.min_minority_samples:
            # Ultra-rare support (< 3 minority samples): lock slope a = 1.0 to
            # prevent slope overfitting/inversion and fit intercept shift only.
            self.is_low_support_ = True
            reg_b = self.reg_intercept

            def intercept_obj(b_vec: np.ndarray):
                b = float(b_vec[0])
                s = z + b
                nll = float(np.mean(weights * (np.logaddexp(0.0, s) - y_float * s)))
                loss = nll + 0.5 * reg_b * (b * b)
                grad = float(np.mean(weights * (expit(s) - y_float)) + reg_b * b)
                return loss, np.array([grad], dtype=np.float64)

            res = minimize(
                intercept_obj,
                x0=np.array([0.0], dtype=np.float64),
                jac=True,
                method="L-BFGS-B",
                bounds=[(-b_lim, b_lim)],
            )
            self.a_ = 1.0
            self.b_ = float(res.x[0]) if res.success or np.isfinite(res.x[0]) else 0.0
            self.is_fitted_ = True
            return self

        self.is_low_support_ = False
        reg_a = self.reg_slope
        reg_b = self.reg_intercept
        min_a, max_a = self.slope_bounds

        def platt_obj(params: np.ndarray):
            a, b = float(params[0]), float(params[1])
            s = a * z + b
            nll = float(np.mean(weights * (np.logaddexp(0.0, s) - y_float * s)))
            reg = 0.5 * reg_a * ((a - 1.0) ** 2) + 0.5 * reg_b * (b * b)
            diff = weights * (expit(s) - y_float)
            grad_a = float(np.mean(diff * z) + reg_a * (a - 1.0))
            grad_b = float(np.mean(diff) + reg_b * b)
            return nll + reg, np.array([grad_a, grad_b], dtype=np.float64)

        res = minimize(
            platt_obj,
            x0=np.array([1.0, 0.0], dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            bounds=[(min_a, max_a), (-b_lim, b_lim)],
        )
        if np.all(np.isfinite(res.x)):
            self.a_ = float(res.x[0])
            self.b_ = float(res.x[1])
        else:
            self.a_ = 1.0
            self.b_ = 0.0

        self.is_fitted_ = True
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        p = np.asarray(probs, dtype=np.float64).ravel()
        if not self.is_fitted_:
            return np.clip(p, 0.0, 1.0)
        if self.is_degenerate_:
            return np.full_like(p, self.degenerate_value_)

        p_clipped = np.clip(p, self.clip_eps, 1.0 - self.clip_eps)
        z = np.log(p_clipped / (1.0 - p_clipped))
        return np.clip(expit(self.a_ * z + self.b_), 0.0, 1.0)


class MultiLabelTailCalibrator:
    """Multilabel container applying independent TailCalibrator per label column."""

    def __init__(
        self,
        method: str = "sqrt_platt",
        clip_eps: float = 1e-5,
        reg_slope: float = 0.2,
        reg_intercept: float = 0.02,
        min_minority_samples: int = 3,
        random_state: int = 42,
    ):
        self.method = method
        self.clip_eps = clip_eps
        self.reg_slope = reg_slope
        self.reg_intercept = reg_intercept
        self.min_minority_samples = min_minority_samples
        self.random_state = random_state
        self.calibrators_: Dict[int, TailCalibrator] = {}
        self.n_labels_: int = 0

    def fit(self, probs: np.ndarray, Y: np.ndarray) -> "MultiLabelTailCalibrator":
        P = np.asarray(probs, dtype=np.float64)
        Y_arr = np.asarray(Y, dtype=np.int32)
        if P.ndim != 2 or Y_arr.shape != P.shape:
            raise ValueError(f"Shape mismatch: probs {P.shape} vs Y {Y_arr.shape}")

        self.n_labels_ = P.shape[1]
        self.calibrators_ = {}
        for j in range(self.n_labels_):
            cal = TailCalibrator(
                method=self.method,
                clip_eps=self.clip_eps,
                reg_slope=self.reg_slope,
                reg_intercept=self.reg_intercept,
                min_minority_samples=self.min_minority_samples,
                random_state=self.random_state,
            )
            cal.fit(P[:, j], Y_arr[:, j])
            self.calibrators_[j] = cal
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        P = np.asarray(probs, dtype=np.float64)
        out = np.zeros_like(P)
        for j in range(P.shape[1]):
            if j in self.calibrators_:
                out[:, j] = self.calibrators_[j].predict_proba(P[:, j])
            else:
                out[:, j] = np.clip(P[:, j], 0.0, 1.0)
        return np.clip(out, 0.0, 1.0)
