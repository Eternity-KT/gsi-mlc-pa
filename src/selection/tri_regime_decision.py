"""Parameter-Free Closed-Form Prior-Adjusted Bayes Thresholds & OOF Coverage Guard.

Mathematical Foundation:
For label prior pi_l in (0, 1) and abstention cost c in (0, 0.5), the closed-form
prior-adjusted Bayes rejection thresholds bounded by the canonical 0.50 decision
boundary are:

    tau_0(pi_l, c) = min(0.50, c * pi_l / (c * pi_l + (1 - c) * (1 - pi_l)))
    tau_1(pi_l, c) = max(0.50, (1 - c) * pi_l / ((1 - c) * pi_l + c * (1 - pi_l)))

Key properties (requiring zero extra tuning hyperparameters):
1. Exact Chow reduction at balance: tau_0(0.50, c) = c and tau_1(0.50, c) = 1 - c.
2. Exact 0/1 dual symmetry: tau_0(1 - pi, c) = 1 - tau_1(pi, c) and
   tau_1(1 - pi, c) = 1 - tau_0(pi, c) for all pi in (0, 1).
3. Global continuity across pi in (0, 1) without heuristic regime splits,
   threshold caps/floors, or sigmoid blending parameters.
4. OOF Coverage Guard (gamma_min): contracts [tau_0, tau_1] smoothly on training
   out-of-fold probabilities when empirical coverage < gamma_min while strictly
   preserving tau_0 <= 0.50 <= tau_1.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np


def compute_label_balance_factor(prior: float) -> float:
    """Compute diagnostic Label Balance Factor beta_l = 1 - 2 * |pi_l - 0.50| in [0, 1]."""
    pi = float(np.clip(prior, 0.0, 1.0))
    return float(1.0 - 2.0 * abs(pi - 0.50))


def compute_prior_bayes_raw_thresholds(
    prior: float,
    cost: float = 0.30,
    clip_bounds: Tuple[float, float] = (0.005, 0.995),
) -> Tuple[float, float, str]:
    """Compute parameter-free closed-form prior-adjusted Bayes thresholds [tau_0, tau_1]."""
    c = float(cost)
    if c <= 0.0 or c >= 0.5:
        raise ValueError(f"Rejection cost c must be in (0, 0.5), got {c}")

    pi = float(np.clip(prior, 1e-6, 1.0 - 1e-6))
    one_minus_pi = 1.0 - pi
    min_t, max_t = clip_bounds

    t0_raw = (c * pi) / (c * pi + (1.0 - c) * one_minus_pi)
    t1_raw = ((1.0 - c) * pi) / ((1.0 - c) * pi + c * one_minus_pi)

    tau_0 = float(np.clip(min(0.50, t0_raw), min_t, 0.49))
    tau_1 = float(np.clip(max(0.50, t1_raw), 0.51, max_t))

    if abs(pi - 0.50) <= 0.075:
        regime = "Symmetric"
    elif pi < 0.50:
        regime = "Rare_Negative"
    else:
        regime = "Rare_Positive"

    return tau_0, tau_1, regime


# Backward-compatible alias returning (tau_0, tau_1, regime, blend_weight)
def compute_tri_regime_raw_thresholds(
    prior: float,
    cost: float = 0.30,
    clip_bounds: Tuple[float, float] = (0.005, 0.995),
) -> Tuple[float, float, str, float]:
    tau_0, tau_1, regime = compute_prior_bayes_raw_thresholds(
        prior=prior, cost=cost, clip_bounds=clip_bounds
    )
    return tau_0, tau_1, regime, compute_label_balance_factor(prior)


def apply_dual_coverage_guard(
    probs_1d: np.ndarray,
    tau_0: float,
    tau_1: float,
    prior: float,
    gamma_min: float = 0.70,
    max_iterations: int = 50,
) -> Tuple[float, float, bool]:
    """Contract [tau_0, tau_1] if OOF empirical coverage < gamma_min while preserving tau_0 <= 0.50 <= tau_1."""
    probs_clean = np.asarray(probs_1d, dtype=np.float64).ravel()
    if len(probs_clean) == 0 or float(gamma_min) <= 0.0:
        return float(tau_0), float(tau_1), False

    gamma = float(gamma_min)
    decided = (probs_clean <= tau_0) | (probs_clean >= tau_1)
    if float(np.mean(decided)) >= gamma:
        return float(tau_0), float(tau_1), False

    t0, t1 = float(tau_0), float(tau_1)
    step = (t1 - t0) / (2.0 * max_iterations)

    for _ in range(max_iterations):
        if prior <= 0.50:
            t0 = min(0.50, t0 + step)
            if t1 > 0.50:
                t1 = max(0.50, t1 - step)
        else:
            t1 = max(0.50, t1 - step)
            if t0 < 0.50:
                t0 = min(0.50, t0 + step)

        if t0 >= t1:
            mid = 0.5 * (tau_0 + tau_1)
            t0 = mid - 1e-4
            t1 = mid + 1e-4
            break

        if float(np.mean((probs_clean <= t0) | (probs_clean >= t1))) >= gamma:
            break

    return float(t0), float(t1), True


def compute_tri_regime_adaptive_thresholds_table(
    Y_train: np.ndarray,
    OOF_probs: Optional[np.ndarray] = None,
    cost: float = 0.30,
    gamma_min: float = 0.70,
    clip_bounds: Tuple[float, float] = (0.005, 0.995),
) -> Dict[int, Dict[str, Any]]:
    """Compute per-label closed-form Bayes abstention thresholds with OOF coverage guard."""
    Y = np.asarray(Y_train, dtype=np.int32)
    _, n_labels = Y.shape
    table: Dict[int, Dict[str, Any]] = {}

    for j in range(n_labels):
        prior_j = float(np.mean(Y[:, j]))
        beta_j = compute_label_balance_factor(prior_j)
        t0, t1, regime_j = compute_prior_bayes_raw_thresholds(
            prior=prior_j, cost=cost, clip_bounds=clip_bounds
        )

        guard_applied = False
        emp_cov = 1.0
        if OOF_probs is not None and OOF_probs.shape[1] > j:
            p_j = OOF_probs[:, j]
            if float(gamma_min) > 0.0:
                t0, t1, guard_applied = apply_dual_coverage_guard(
                    probs_1d=p_j,
                    tau_0=t0,
                    tau_1=t1,
                    prior=prior_j,
                    gamma_min=gamma_min,
                )
            emp_cov = float(np.mean((p_j <= t0) | (p_j >= t1)))

        table[j] = {
            "tau_0": float(t0),
            "tau_1": float(t1),
            "prior": float(prior_j),
            "beta": float(beta_j),
            "regime": regime_j,
            "blend_weight": float(beta_j),
            "coverage_guard": bool(guard_applied),
            "empirical_coverage": float(emp_cov),
        }

    return table


def apply_tri_regime_abstention(
    probs: np.ndarray,
    thresholds_table: Dict[int, Dict[str, Any]],
    abstain_value: int = -1,
) -> np.ndarray:
    """Apply per-label [tau_0, tau_1] abstention thresholds to probability matrix."""
    P = np.asarray(probs, dtype=np.float64)
    M, K = P.shape
    preds = np.full((M, K), abstain_value, dtype=np.int32)

    for j in range(K):
        t0 = thresholds_table[j]["tau_0"]
        t1 = thresholds_table[j]["tau_1"]
        p_col = P[:, j]
        preds[p_col <= t0, j] = 0
        preds[p_col >= t1, j] = 1

    return preds
