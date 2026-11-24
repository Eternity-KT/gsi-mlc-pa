"""Unified Multi-Label Adaptive Calibrator.

Applies smooth, dual-symmetric, identity-regularized Balanced-Root Platt
calibration across all label regimes without regime boundary cliffs:
- Minority-Positive (pi_l < (1 - tau_balance)/2): w_pos / w_neg = sqrt(N_0 / N_1) > 1.
- Balanced (beta_l >= tau_balance): smooth transition around w_pos / w_neg ~ 1
  (calibrating SVM/MLP probabilities instead of leaving them uncalibrated).
- Majority-Positive (pi_l > (1 + tau_balance)/2): w_pos / w_neg = sqrt(N_0 / N_1) < 1
  (mathematically identical to weighting negatives by sqrt(N_1 / N_0)).
"""

from typing import Dict
import numpy as np
from .tail_calibrator import TailCalibrator


class MultiLabelAdaptiveCalibrator:
    """Unified smooth multi-label calibrator with diagnostic regime tracking."""

    def __init__(
        self,
        tau_balance: float = 0.85,
        method: str = "sqrt_platt",
        clip_eps: float = 1e-5,
        reg_slope: float = 0.5,
        reg_intercept: float = 0.05,
        min_minority_samples: int = 3,
        random_state: int = 42,
    ):
        self.tau_balance = float(tau_balance)
        self.method = str(method)
        self.clip_eps = float(clip_eps)
        self.reg_slope = float(reg_slope)
        self.reg_intercept = float(reg_intercept)
        self.min_minority_samples = int(min_minority_samples)
        self.random_state = int(random_state)
        self.calibrators_: Dict[int, TailCalibrator] = {}
        self.regimes_: Dict[int, str] = {}
        self.n_labels_: int = 0

    @staticmethod
    def _classify_regime(prior: float, tau_balance: float) -> str:
        beta = 1.0 - 2.0 * abs(float(prior) - 0.50)
        if beta >= tau_balance:
            return "Balanced"
        if prior <= 0.50:
            return "Minority_Positive"
        return "Majority_Positive"

    def fit(self, probs: np.ndarray, Y: np.ndarray) -> "MultiLabelAdaptiveCalibrator":
        P = np.asarray(probs, dtype=np.float64)
        Y_arr = np.asarray(Y, dtype=np.int32)
        if P.ndim != 2 or Y_arr.shape != P.shape:
            raise ValueError(f"Shape mismatch: probs {P.shape} vs Y {Y_arr.shape}")

        _, n_labels = P.shape
        self.n_labels_ = n_labels
        self.calibrators_ = {}
        self.regimes_ = {}

        for j in range(n_labels):
            prior_j = float(np.mean(Y_arr[:, j]))
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
            self.regimes_[j] = self._classify_regime(prior_j, self.tau_balance)

        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        P = np.asarray(probs, dtype=np.float64)
        if P.ndim != 2:
            raise ValueError(f"Expected 2D probability matrix, got shape {P.shape}")
        out = np.zeros_like(P)

        for j in range(P.shape[1]):
            if j in self.calibrators_:
                out[:, j] = self.calibrators_[j].predict_proba(P[:, j])
            else:
                out[:, j] = np.clip(P[:, j], 0.0, 1.0)

        return np.clip(out, 0.0, 1.0)
