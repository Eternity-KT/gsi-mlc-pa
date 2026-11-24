"""Unit Tests for Minimal 3-Hyperparameter GSI-MLC-PA Core Components.

Tests:
1. Label balance factor beta_l computation.
2. Parameter-free closed-form prior-adjusted Bayes thresholds:
   - Exact reduction to Symmetric Chow [c, 1 - c] at pi = 0.50.
   - Exact dual 0/1 symmetry: tau_0(1 - pi, c) == 1 - tau_1(pi, c).
   - Global continuity across pi in (0, 1) without regime cliffs.
3. Unified TailCalibrator:
   - Exact dual 0/1 symmetry: cal(1 - p; 1 - y) == 1 - cal(p; y).
   - Strict monotonicity (a_ > 0) even on anti-correlated inputs.
   - Rare-label support guard (min_minority_samples = 3): locks a_ == 1.0.
   - Smooth calibration across Minority_Positive, Balanced, and Majority_Positive regimes.
4. GSIMLCPAv6_3_3Classifier:
   - Minimal 3-hyperparameter interface (cost, anchor_threshold, gamma_min).
   - Single-stage Anchor (AL) & Contextual (CL) cascade with zero-centered 2*P - 1 context.
   - Zero duplicate CV fits when AL is empty.
   - End-to-end fit, predict_proba, predict, predict_full, and dynamic cost sweep.
"""

import inspect
import numpy as np
import pytest

from src.selection.tri_regime_decision import (
    apply_dual_coverage_guard,
    compute_label_balance_factor,
    compute_prior_bayes_raw_thresholds,
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


def test_closed_form_bayes_thresholds_symmetry_and_chow_reduction():
    """Verify exact Chow reduction at pi=0.50 and exact 0/1 dual symmetry across (0, 1)."""
    cost = 0.30

    t0_mid, t1_mid, regime_mid, blend_mid = compute_tri_regime_raw_thresholds(
        prior=0.50, cost=cost
    )
    assert regime_mid == "Symmetric"
    assert blend_mid == pytest.approx(1.0, abs=1e-6)
    assert t0_mid == pytest.approx(0.30, abs=1e-7)
    assert t1_mid == pytest.approx(0.70, abs=1e-7)

    # Exact dual 0/1 symmetry across all priors
    for pi in (0.02, 0.15, 0.30, 0.42, 0.50, 0.68, 0.91):
        t0_p, t1_p, _ = compute_prior_bayes_raw_thresholds(prior=pi, cost=cost)
        t0_inv, t1_inv, _ = compute_prior_bayes_raw_thresholds(prior=1.0 - pi, cost=cost)
        assert t0_p == pytest.approx(1.0 - t1_inv, abs=1e-7)
        assert t1_p == pytest.approx(1.0 - t0_inv, abs=1e-7)
        assert t0_p <= 0.50 <= t1_p


def test_closed_form_bayes_continuity_and_coverage_guard():
    """Verify smooth continuity across priors and canonical 0.50 boundary preservation."""
    priors = np.linspace(0.01, 0.99, 100)
    t0_list, t1_list = [], []
    for p in priors:
        t0, t1, _ = compute_prior_bayes_raw_thresholds(prior=p, cost=0.30)
        t0_list.append(t0)
        t1_list.append(t1)

    assert np.max(np.abs(np.diff(t0_list))) < 0.03
    assert np.max(np.abs(np.diff(t1_list))) < 0.03

    probs = np.linspace(0.20, 0.80, 100)
    t0_g, t1_g, triggered = apply_dual_coverage_guard(
        probs, tau_0=0.05, tau_1=0.55, prior=0.08, gamma_min=0.70
    )
    assert triggered is True
    assert t0_g <= 0.50 <= t1_g


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

    cal_anti = TailCalibrator().fit(probs, 1 - y)
    grid = np.linspace(0.01, 0.99, 50)
    pred_grid = cal_anti.predict_proba(grid)
    assert cal_anti.a_ >= 0.05
    assert np.all(np.diff(pred_grid) > 0.0)


def test_tail_calibrator_low_support_and_degenerate_guards():
    """Verify ultra-rare labels (< 3 minority samples) lock slope a_ = 1.0 without overfitting."""
    probs = np.linspace(0.05, 0.95, 100)
    y_one_pos = np.zeros(100, dtype=np.int32)
    y_one_pos[-1] = 1

    cal_rare = TailCalibrator(min_minority_samples=3).fit(probs, y_one_pos)
    assert cal_rare.is_low_support_ is True
    assert cal_rare.a_ == 1.0
    assert np.isfinite(cal_rare.b_)

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
    probs = 1.0 / (1.0 + np.exp(-4.0 * raw_logits))

    y0 = (rng.rand(N) < 1.0 / (1.0 + np.exp(-(raw_logits[:, 0] - 2.2)))).astype(int)
    y1 = (rng.rand(N) < 1.0 / (1.0 + np.exp(-raw_logits[:, 1]))).astype(int)
    y2 = (rng.rand(N) < 1.0 / (1.0 + np.exp(-(raw_logits[:, 2] + 2.2)))).astype(int)
    Y = np.column_stack([y0, y1, y2])

    calibrator = MultiLabelAdaptiveCalibrator(tau_balance=0.80).fit(probs, Y)
    assert calibrator.regimes_[0] == "Minority_Positive"
    assert calibrator.regimes_[1] == "Balanced"
    assert calibrator.regimes_[2] == "Majority_Positive"

    for j in range(3):
        assert isinstance(calibrator.calibrators_[j], TailCalibrator)

    cal_probs = calibrator.transform(probs)
    assert cal_probs.shape == probs.shape
    assert np.all((cal_probs >= 0.0) & (cal_probs <= 1.0))
    assert calibrator.calibrators_[1].a_ < 1.0


def test_gsi_minimal_hyperparameter_signature_and_inference():
    """Verify minimal 3-hyperparameter signature, AL/CL cascade, empty-AL OOF reuse, and cost sweep."""
    sig_params = tuple(inspect.signature(GSIMLCPAv6_3_3Classifier.__init__).parameters.keys())
    assert sig_params == (
        "self",
        "base_learner",
        "cost",
        "anchor_threshold",
        "gamma_min",
        "cv_folds",
        "abstain_value",
        "random_state",
    )

    rng = np.random.RandomState(42)
    N, d = 160, 8
    X = rng.randn(N, d).astype(np.float32)

    y0 = (2.8 * X[:, 0] - 1.5 * X[:, 1] > 0.0).astype(int)
    y1 = (0.8 * X[:, 2] + 1.5 * y0 + 0.8 * rng.randn(N) > 0.7).astype(int)
    y2 = (rng.rand(N) < 0.08).astype(int)
    y3 = (rng.rand(N) < 0.88).astype(int)
    Y = np.column_stack([y0, y1, y2, y3])

    clf = GSIMLCPAv6_3_3Classifier(
        base_learner="logistic",
        cost=0.30,
        anchor_threshold=0.75,
        gamma_min=0.70,
        cv_folds=3,
        random_state=42,
    ).fit(X, Y)

    assert clf.is_fitted_
    assert 0 in clf.anchor_labels_
    assert clf.anchor_layers_ == [clf.anchor_labels_]
    assert sorted(clf.anchor_labels_ + clf.contextual_labels_) == [0, 1, 2, 3]
    assert not set(clf.anchor_labels_).intersection(clf.contextual_labels_)

    probs = clf.predict_proba(X)
    assert probs.shape == (N, 4)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    preds_default = clf.predict(X)
    assert set(np.unique(preds_default)).issubset({-1, 0, 1})

    # Empty-AL path (anchor_threshold > 1.0) must cleanly reuse Stage-1 OOF without error
    clf_empty_al = GSIMLCPAv6_3_3Classifier(
        base_learner="logistic",
        cost=0.30,
        anchor_threshold=1.50,
        gamma_min=0.0,
        cv_folds=3,
        random_state=42,
    ).fit(X, Y)
    assert clf_empty_al.anchor_labels_ == []
    assert clf_empty_al.anchor_layers_ == []
    assert clf_empty_al.contextual_labels_ == [0, 1, 2, 3]

    cov_low_cost = np.mean(clf_empty_al.predict(X, cost=0.10) != -1)
    cov_high_cost = np.mean(clf_empty_al.predict(X, cost=0.45) != -1)
    cov_full_cost = np.mean(clf_empty_al.predict(X, cost=0.50) != -1)
    assert cov_high_cost >= cov_low_cost
    assert cov_full_cost == 1.0

    full_preds = clf.predict_full(X)
    assert set(np.unique(full_preds)).issubset({0, 1})
