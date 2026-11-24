"""Unit Tests for GSI-MLC-PA Core Components (Anchor/Contextual Cascade & Unified Calibration).

Tests:
1. Label balance factor beta_l computation.
2. Tri-Regime threshold computation & smooth boundary blending.
3. Unified TailCalibrator:
   - Exact dual 0/1 symmetry: cal(1 - p; 1 - y) == 1 - cal(p; y).
   - Strict monotonicity (a_ > 0) even on anti-correlated inputs.
   - Rare-label support guard (min_minority_samples = 3): locks a_ == 1.0.
   - Smooth calibration across Minority_Positive, Balanced, and Majority_Positive regimes.
4. GSIMLCPAv6_3_3Classifier:
   - Anchor Labels (AL) & Contextual Labels (CL) partitioning without OS-NMF.
   - End-to-end fit, predict_proba, predict, predict_full, and dynamic cost sweep.
   - Separation-Aware Margin Guard (opt-in synthetic test).
"""

import numpy as np
import pytest

from src.selection.tri_regime_decision import (
    compute_label_balance_factor,
    compute_tri_regime_raw_thresholds,
)
from src.calibration.adaptive_calibrator import MultiLabelAdaptiveCalibrator
from src.calibration.tail_calibrator import TailCalibrator
from src.models.gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier


def test_label_balance_factor():
    """Verify beta_l equals 1.0 at balance and 0.0 at extremes."""
    assert compute_label_balance_factor(0.50) == pytest.approx(1.0, abs=1e-5)
    assert compute_label_balance_factor(0.0) == pytest.approx(0.0, abs=1e-5)
    assert compute_label_balance_factor(1.0) == pytest.approx(0.0, abs=1e-5)
    assert compute_label_balance_factor(0.30) == pytest.approx(0.60, abs=1e-5)
    assert compute_label_balance_factor(0.70) == pytest.approx(0.60, abs=1e-5)
    assert compute_label_balance_factor(0.10) == pytest.approx(0.20, abs=1e-5)
    assert compute_label_balance_factor(0.90) == pytest.approx(0.20, abs=1e-5)


def test_tri_regime_thresholds_regimes():
    """Verify threshold behavior across the 3 regimes."""
    cost = 0.30

    t0_mid, t1_mid, regime_mid, blend_mid = compute_tri_regime_raw_thresholds(
        prior=0.50, cost=cost, tau_balance=0.40
    )
    assert regime_mid == "Symmetric"
    assert blend_mid > 0.99
    assert t0_mid == pytest.approx(0.30, abs=0.01)
    assert t1_mid == pytest.approx(0.70, abs=0.01)

    t0_low, t1_low, regime_low, blend_low = compute_tri_regime_raw_thresholds(
        prior=0.02, cost=cost, tau_balance=0.40
    )
    assert regime_low == "Rare_Negative"
    assert blend_low < 0.05
    assert t0_low < 0.15
    assert t1_low >= 0.50

    t0_high, t1_high, regime_high, blend_high = compute_tri_regime_raw_thresholds(
        prior=0.95, cost=cost, tau_balance=0.40
    )
    assert regime_high == "Rare_Positive"
    assert blend_high < 0.05
    assert t0_high <= 0.50
    assert t1_high > 0.85


def test_smooth_blending_continuity():
    """Verify smooth sigmoid blending has no discontinuous jumps across prior sweep."""
    priors = np.linspace(0.01, 0.99, 100)
    t0_list = []
    t1_list = []

    for p in priors:
        t0, t1, _, _ = compute_tri_regime_raw_thresholds(prior=p, cost=0.30, tau_balance=0.40)
        t0_list.append(t0)
        t1_list.append(t1)

    assert np.max(np.abs(np.diff(t0_list))) < 0.08
    assert np.max(np.abs(np.diff(t1_list))) < 0.08


def test_tail_calibrator_exact_dual_symmetry_and_monotonicity():
    """Verify unified TailCalibrator satisfies f(1 - p; 1 - y) == 1 - f(p; y) and a_ > 0."""
    rng = np.random.RandomState(42)
    probs = rng.uniform(0.02, 0.98, size=180)
    y = (probs + 0.25 * rng.randn(180) > 0.72).astype(np.int32)

    for method in ("sqrt_platt", "weighted_platt", "standard_platt"):
        cal_pos = TailCalibrator(method=method).fit(probs, y)
        cal_inv = TailCalibrator(method=method).fit(1.0 - probs, 1 - y)

        assert cal_pos.a_ > 0.0
        assert cal_pos.a_ == pytest.approx(cal_inv.a_, abs=1e-6)
        assert cal_pos.b_ == pytest.approx(-cal_inv.b_, abs=1e-6)

        out_pos = cal_pos.predict_proba(probs)
        out_inv = cal_inv.predict_proba(1.0 - probs)
        np.testing.assert_allclose(out_inv, 1.0 - out_pos, atol=1e-7)

    # Anti-correlated noisy input must remain strictly monotonic (a_ >= min_slope > 0)
    cal_anti = TailCalibrator().fit(probs, 1 - y)
    grid = np.linspace(0.01, 0.99, 50)
    pred_grid = cal_anti.predict_proba(grid)
    assert cal_anti.a_ >= 0.05
    assert np.all(np.diff(pred_grid) > 0.0)


def test_tail_calibrator_low_support_and_degenerate_guards():
    """Verify ultra-rare labels (< 3 minority samples) lock slope a_ = 1.0 without overfitting."""
    probs = np.linspace(0.05, 0.95, 100)
    y_one_pos = np.zeros(100, dtype=np.int32)
    y_one_pos[-1] = 1  # single positive sample (e.g., Genbase rare motif)

    cal_rare = TailCalibrator(min_minority_samples=3).fit(probs, y_one_pos)
    assert cal_rare.is_low_support_ is True
    assert cal_rare.a_ == 1.0
    assert np.isfinite(cal_rare.b_)

    # Degenerate constant targets
    cal_zero = TailCalibrator().fit(probs, np.zeros(100, dtype=np.int32))
    assert cal_zero.is_degenerate_ is True
    np.testing.assert_allclose(cal_zero.predict_proba(probs), 0.0)

    cal_one = TailCalibrator().fit(probs, np.ones(100, dtype=np.int32))
    assert cal_one.is_degenerate_ is True
    np.testing.assert_allclose(cal_one.predict_proba(probs), 1.0)


def test_multilabel_adaptive_calibrator():
    """Verify unified calibrator calibrates all regimes including Balanced without Identity hole."""
    rng = np.random.RandomState(42)
    N = 200
    raw_logits = rng.randn(N, 3)
    # Stochastic ground-truth with true logit slope = 1.0, while probs has overconfident slope = 4.0
    probs = 1.0 / (1.0 + np.exp(-4.0 * raw_logits))

    y0 = (rng.rand(N) < 1.0 / (1.0 + np.exp(-(raw_logits[:, 0] - 2.2)))).astype(int)  # Minority_Positive
    y1 = (rng.rand(N) < 1.0 / (1.0 + np.exp(-raw_logits[:, 1]))).astype(int)          # Balanced (~50%)
    y2 = (rng.rand(N) < 1.0 / (1.0 + np.exp(-(raw_logits[:, 2] + 2.2)))).astype(int)  # Majority_Positive
    Y = np.column_stack([y0, y1, y2])

    calibrator = MultiLabelAdaptiveCalibrator(tau_balance=0.80)
    calibrator.fit(probs, Y)

    assert calibrator.regimes_[0] == "Minority_Positive"
    assert calibrator.regimes_[1] == "Balanced"
    assert calibrator.regimes_[2] == "Majority_Positive"

    # All regimes use unified TailCalibrator (including Balanced!)
    for j in range(3):
        assert isinstance(calibrator.calibrators_[j], TailCalibrator)

    cal_probs = calibrator.transform(probs)
    assert cal_probs.shape == probs.shape
    assert np.all((cal_probs >= 0.0) & (cal_probs <= 1.0))
    # Overconfident balanced logits (3.5x) should have slope compressed below 1.0
    assert calibrator.calibrators_[1].a_ < 1.0


def test_gsi_classifier_anchor_contextual_no_nmf_and_cost_sweep():
    """Verify AL/CL partitioning, OS-NMF removal, and dynamic cost sensitivity."""
    rng = np.random.RandomState(42)
    N, d = 160, 8
    X = rng.randn(N, d).astype(np.float32)

    # Label 0: strongly predictable -> Anchor Label (AL)
    y0 = (2.8 * X[:, 0] - 1.5 * X[:, 1] > 0.0).astype(int)
    # Label 1: balanced contextual
    y1 = (0.8 * X[:, 2] + 1.5 * y0 + 0.8 * rng.randn(N) > 0.7).astype(int)
    # Label 2: rare positive contextual
    y2 = (rng.rand(N) < 0.08).astype(int)
    # Label 3: majority positive contextual
    y3 = (rng.rand(N) < 0.88).astype(int)
    Y = np.column_stack([y0, y1, y2, y3])

    clf = GSIMLCPAv6_3_3Classifier(
        base_learner="logistic",
        stratified_threshold=0.75,
        cv_folds=3,
        cost=0.30,
        tau_balance=0.85,
        random_state=42,
    ).fit(X, Y)

    assert clf.is_fitted_
    # Verify standardized Anchor (AL) and Contextual (CL) attributes
    assert 0 in clf.anchor_labels_
    assert sorted(clf.anchor_labels_ + clf.contextual_labels_) == [0, 1, 2, 3]
    assert not set(clf.anchor_labels_).intersection(clf.contextual_labels_)

    # Verify OS-NMF artifacts are completely absent
    assert not hasattr(clf, "signed_residual_corr_matrix_")
    assert not hasattr(clf, "mf_alpha")
    assert not hasattr(clf, "dependency_graph_")

    probs = clf.predict_proba(X)
    assert probs.shape == (N, 4)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    preds_default = clf.predict(X)
    assert set(np.unique(preds_default)).issubset({-1, 0, 1})

    # Dynamic cost sweep: higher abstention cost must yield >= coverage
    clf_free_guard = GSIMLCPAv6_3_3Classifier(
        base_learner="logistic", cv_folds=3, cost=0.30, gamma_min=0.0, random_state=42
    ).fit(X, Y)
    cov_low_cost = np.mean(clf_free_guard.predict(X, cost=0.10) != -1)
    cov_high_cost = np.mean(clf_free_guard.predict(X, cost=0.45) != -1)
    cov_full_cost = np.mean(clf_free_guard.predict(X, cost=0.50) != -1)
    assert cov_high_cost >= cov_low_cost
    assert cov_full_cost == 1.0

    full_preds = clf.predict_full(X)
    assert set(np.unique(full_preds)).issubset({0, 1})


def test_separation_aware_margin_guard_extreme_imbalance():
    """Verify opt-in Separation-Aware Margin Guard adapts tau_1 for cleanly separated rare labels."""
    X = np.zeros((100, 2), dtype=np.float32)
    X[2:, 0] = -1.0
    X[:2, 0] = 0.8
    y_rare = np.zeros(100, dtype=int)
    y_rare[:2] = 1
    y_bal = (np.arange(100) % 2).astype(int)
    Y = np.column_stack([y_rare, y_bal])

    clf = GSIMLCPAv6_3_3Classifier(
        base_learner="logistic",
        cv_folds=3,
        separation_aware=True,
        random_state=42,
    ).fit(X, Y)

    th_rare = clf.adaptive_thresholds_[0]
    assert th_rare["tau_0"] < th_rare["tau_1"]
    preds = clf.predict(X)
    assert preds[0, 0] == 1 and preds[1, 0] == 1
    assert np.mean(preds[2:, 0] == 0) > 0.90
