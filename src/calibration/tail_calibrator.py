"""
Tail Probability Calibration for Extreme Label Imbalance.

Reference:
    - spec/spec_v6_3_3.md
    - Kull et al. (2017) "Beta calibration: a well-founded and easily implemented alternative on two-class problems"
    - Platt (1999) Probabilistic Outputs for Support Vector Machines

Provides:
- TailCalibrator: Monotonic 1D probability calibrator with class-weight compensation for rare labels.
- MultiLabelTailCalibrator: Multilabel container for independent per-label probability calibration.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.linear_model import LogisticRegression


class TailCalibrator:
    """
    Calibrates 1D probabilities from classifiers under extreme class imbalance.

    Fits a weighted Platt sigmoid over logit-transformed probabilities:
        logit(p) = log(p / (1 - p))
        P_cal(y=1 | p) = 1 / (1 + exp(-(A * logit(p) + B)))
    with sample weights balancing positive and negative gradient contributions.
    """

    def __init__(
        self,
        method: str = "weighted_platt",
        clip_eps: float = 1e-5,
        random_state: int = 42,
    ):
        self.method = method
        self.clip_eps = clip_eps
        self.random_state = random_state
        self.is_fitted_ = False
        self.is_degenerate_ = False
        self.degenerate_value_ = 0.0
        self.model_: Optional[LogisticRegression] = None

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "TailCalibrator":
        p = np.asarray(probs, dtype=np.float64).ravel()
        labels = np.asarray(y, dtype=np.int32).ravel()

        if len(p) != len(labels):
            raise ValueError(f"Shape mismatch: {len(p)} vs {len(labels)}")

        unique = np.unique(labels)
        if len(unique) <= 1:
            self.is_fitted_ = True
            self.is_degenerate_ = True
            self.degenerate_value_ = float(unique[0]) if len(unique) == 1 else 0.0
            return self

        # Avoid zero or one for logit transform
        p_clipped = np.clip(p, self.clip_eps, 1.0 - self.clip_eps)
        logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

        # Compute imbalance weights based on method
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        
        if self.method in ("standard_platt", "unweighted"):
            sample_weight = None
        elif self.method in ("sqrt_platt", "balanced_root"):
            pos_weight = float(n_neg) / float(max(n_pos, 1))
            sample_weight = np.where(labels == 1, np.sqrt(pos_weight), 1.0).astype(np.float64)
        else:
            # Default: weighted_platt (full linear class weight)
            pos_weight = float(n_neg) / float(max(n_pos, 1))
            sample_weight = np.where(labels == 1, pos_weight, 1.0).astype(np.float64)

        try:
            lr = LogisticRegression(
                C=1.0,
                solver="liblinear",
                max_iter=1000,
                random_state=self.random_state,
            )
            lr.fit(logits, labels, sample_weight=sample_weight)

            # Sanity check: slope must be strictly positive (monotonic relationship)
            if lr.coef_[0, 0] <= 0.0:
                # Non-monotonic or inverted slope -> fallback to identity
                self.model_ = None
            else:
                self.model_ = lr
        except Exception:
            self.model_ = None

        self.is_fitted_ = True
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        p = np.asarray(probs, dtype=np.float64).ravel()
        if not self.is_fitted_:
            return p

        if self.is_degenerate_:
            return np.full_like(p, self.degenerate_value_)

        if self.model_ is None:
            # Fallback to uncalibrated probability
            return np.clip(p, 0.0, 1.0)

        p_clipped = np.clip(p, self.clip_eps, 1.0 - self.clip_eps)
        logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

        cal_probs = self.model_.predict_proba(logits)[:, 1]
        return np.clip(cal_probs, 0.0, 1.0)


class MultiLabelTailCalibrator:
    """Multilabel calibrator applying independent TailCalibrator per label column."""

    def __init__(
        self,
        method: str = "weighted_platt",
        clip_eps: float = 1e-5,
        random_state: int = 42,
    ):
        self.method = method
        self.clip_eps = clip_eps
        self.random_state = random_state
        self.calibrators_: Dict[int, TailCalibrator] = {}
        self.n_labels_: int = 0

    def fit(self, probs: np.ndarray, Y: np.ndarray) -> "MultiLabelTailCalibrator":
        P = np.asarray(probs, dtype=np.float64)
        Y_arr = np.asarray(Y, dtype=np.int32)
        n_samples, n_labels = P.shape

        self.n_labels_ = n_labels
        self.calibrators_ = {}

        for j in range(n_labels):
            cal = TailCalibrator(
                method=self.method,
                clip_eps=self.clip_eps,
                random_state=self.random_state,
            )
            cal.fit(P[:, j], Y_arr[:, j])
            self.calibrators_[j] = cal

        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        P = np.asarray(probs, dtype=np.float64)
        n_samples, n_labels = P.shape
        out = np.zeros_like(P)

        for j in range(n_labels):
            if j in self.calibrators_:
                out[:, j] = self.calibrators_[j].predict_proba(P[:, j])
            else:
                out[:, j] = np.clip(P[:, j], 0.0, 1.0)

        return np.clip(out, 0.0, 1.0)
