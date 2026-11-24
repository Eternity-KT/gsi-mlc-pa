"""Unit and Integration Tests for Multi-Stage 5-Fold CV Anchor/Contextual Peeling."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.base_learners import create_binary_estimator
from src.models.gsi_mlc_pa import GSIMLCPartialAbstentionClassifier
from src.selection.cv_peeling import (
    CVPeelingConfig,
    CVPeelingResult,
    CVStratifiedPeelingSelector,
    evaluate_label_5fold_cv,
)
from src.selection.partition import canonical_partition_mode


def _make_hierarchical_data(n_samples: int = 300, random_state: int = 42):
    """Generate synthetic multilabel dataset with hierarchical predictability:
    - y0: strongly predictable from X -> Stage-1 Anchor Label (AL_1)
    - y1: depends heavily on y0 and moderately on X -> Stage-2 Anchor Label (AL_2)
    - y2: random low-signal label -> Contextual Label (CL)
    """
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, 6).astype(np.float32)

    logits_0 = 3.0 * X[:, 0] + 2.0 * X[:, 1]
    y0 = (1.0 / (1.0 + np.exp(-logits_0)) > 0.5).astype(np.int32)

    logits_1 = 3.5 * y0 + 1.5 * X[:, 2] - 2.0
    y1 = (1.0 / (1.0 + np.exp(-logits_1)) > 0.5).astype(np.int32)

    y2 = (rng.rand(n_samples) > 0.6).astype(np.int32)
    return X, np.column_stack([y0, y1, y2])


def test_cv_peeling_config():
    config = CVPeelingConfig(
        threshold=0.75,
        n_folds=5,
        max_depth=3,
        cost=0.30,
        decaying_threshold=True,
        threshold_decay_step=0.05,
    )
    d = config.as_dict()
    assert d["threshold"] == 0.75
    assert d["n_folds"] == 5
    assert d["max_depth"] == 3
    assert d["cost"] == 0.30
    assert d["decaying_threshold"] is True
    assert d["threshold_decay_step"] == 0.05


def test_evaluate_label_5fold_cv():
    X, Y = _make_hierarchical_data(n_samples=200, random_state=42)
    factory = lambda: LogisticRegression(solver="liblinear", random_state=42)

    score0, oof_probs0, diag0 = evaluate_label_5fold_cv(
        X, Y[:, 0], factory, n_folds=5, cost=0.30, metric="selective_f1"
    )
    assert len(oof_probs0) == 200
    assert 0.0 <= np.min(oof_probs0) <= np.max(oof_probs0) <= 1.0
    assert score0 > 0.80
    assert diag0["coverage"] > 0.50
    assert len(diag0["fold_metrics"]) == 5

    score2, oof_probs2, _ = evaluate_label_5fold_cv(
        X, Y[:, 2], factory, n_folds=5, cost=0.30, metric="selective_f1"
    )
    assert len(oof_probs2) == 200
    assert score2 < score0


def test_cv_peeling_anchor_contextual_promotion():
    """Verify Stage 1 discovers AL_1 and low-predictability label remains in CL."""
    X, Y = _make_hierarchical_data(n_samples=400, random_state=42)
    selector = CVStratifiedPeelingSelector(
        config=CVPeelingConfig(threshold=0.75, n_folds=5, max_depth=3, cost=0.30, random_state=42),
        base_estimator_factory=lambda: LogisticRegression(solver="liblinear", random_state=42),
    )
    result = selector.fit_partition(X, Y)

    assert isinstance(result, CVPeelingResult)
    assert 0 in result.labels_al_stage_1
    assert 0 in result.anchor_labels
    assert result.n_anchor_labels >= 1
    assert 2 in result.contextual_labels
    assert len(result.final_execution_order) == 3


def test_cv_peeling_stopping_reasons():
    X, Y = _make_hierarchical_data(n_samples=150, random_state=42)

    res_high = CVStratifiedPeelingSelector(
        config=CVPeelingConfig(threshold=1.05, n_folds=3, random_state=42),
        base_estimator_factory=lambda: LogisticRegression(solver="liblinear", random_state=42),
    ).fit_partition(X, Y)
    assert res_high.stopping_reason == "no_promotion_in_stage"
    assert res_high.n_anchor_labels == 0
    assert res_high.n_contextual_labels == 3

    res_low = CVStratifiedPeelingSelector(
        config=CVPeelingConfig(threshold=0.01, n_folds=3, random_state=42),
        base_estimator_factory=lambda: LogisticRegression(solver="liblinear", random_state=42),
    ).fit_partition(X, Y)
    assert res_low.stopping_reason == "all_labels_anchor"
    assert res_low.n_anchor_labels == 3
    assert res_low.n_contextual_labels == 0


def test_cv_peeling_decaying_threshold():
    X, Y = _make_hierarchical_data(n_samples=200, random_state=42)
    res = CVStratifiedPeelingSelector(
        config=CVPeelingConfig(
            threshold=0.85, decaying_threshold=True, threshold_decay_step=0.10, n_folds=3, random_state=42
        ),
        base_estimator_factory=lambda: LogisticRegression(solver="liblinear", random_state=42),
    ).fit_partition(X, Y)
    diags = res.stage_diagnostics
    assert diags[0]["effective_threshold"] == 0.85
    if len(diags) > 1:
        assert diags[1]["effective_threshold"] == 0.75


def test_partition_mode_aliases():
    assert canonical_partition_mode("v6") == "cv_stratified_peeling"
    assert canonical_partition_mode("v6_core") == "cv_stratified_peeling"
    assert canonical_partition_mode("cv_peeling") == "cv_stratified_peeling"
    assert canonical_partition_mode("cv_stratified_peeling") == "cv_stratified_peeling"


def test_gsi_classifier_v6_legacy_integration():
    """Verify backward-compatible properties keep legacy GSIMLCPartialAbstentionClassifier working."""
    X, Y = _make_hierarchical_data(n_samples=200, random_state=42)
    model = GSIMLCPartialAbstentionClassifier(
        partition_mode="v6",
        base_learner="logistic",
        stratified_threshold=0.75,
        cv_folds=3,
        random_state=42,
    ).fit(X, Y)

    assert hasattr(model, "stratified_peeling_audit_")
    assert model.stratified_peeling_audit_ is not None
    assert "anchor_layers" in model.stratified_peeling_audit_

    preds = model.predict(X, cost=0.30)
    assert preds.shape == Y.shape
    assert set(np.unique(preds)).issubset({-1, 0, 1})


def test_cv_peeling_base_learners_svm_mlp():
    X, Y = _make_hierarchical_data(n_samples=150, random_state=42)
    for learner in ("svm_calibrated", "mlp_sklearn"):
        res = CVStratifiedPeelingSelector(
            config=CVPeelingConfig(threshold=0.75, n_folds=3, random_state=42),
            base_estimator_factory=lambda l=learner: create_binary_estimator(l, random_state=42),
        ).fit_partition(X, Y)
        assert isinstance(res, CVPeelingResult)
        assert len(res.final_execution_order) == 3


def test_cv_peeling_defensive_validation():
    with pytest.raises(ValueError, match="threshold"):
        CVPeelingConfig(threshold=-0.1)
    with pytest.raises(ValueError, match="n_folds"):
        CVPeelingConfig(n_folds=1)
    with pytest.raises(ValueError, match="cost"):
        CVPeelingConfig(cost=1.5)
    with pytest.raises(ValueError, match="metric"):
        CVPeelingConfig(metric="non_existent_metric")
    with pytest.raises(ValueError, match="dl_order_direction"):
        CVPeelingConfig(dl_order_direction="diagonal")

    selector = CVStratifiedPeelingSelector(
        config=CVPeelingConfig(threshold=0.75, n_folds=3, random_state=42),
        base_estimator_factory=lambda: LogisticRegression(solver="liblinear", random_state=42),
    )
    X, Y = _make_hierarchical_data(n_samples=50, random_state=42)
    with pytest.raises(ValueError, match="2D array"):
        selector.fit_partition(X[:, 0], Y)
    with pytest.raises(ValueError, match="2D array"):
        selector.fit_partition(X, Y[:, 0])
    with pytest.raises(ValueError, match="Sample count mismatch"):
        selector.fit_partition(X[:30], Y)
    with pytest.raises(ValueError, match="cannot be empty"):
        selector.fit_partition(np.empty((0, 4)), np.empty((0, 2)))

    X_with_nans = X.copy()
    X_with_nans[0, 0] = np.nan
    X_with_nans[1, 1] = np.inf
    assert isinstance(selector.fit_partition(X_with_nans, Y), CVPeelingResult)
