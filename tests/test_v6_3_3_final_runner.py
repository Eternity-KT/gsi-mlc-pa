import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_v6_3_3_final import label_records, make_model, metric_values, validate_predictions
from src.models.gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/v6_3_3_final.json").read_text(encoding="utf-8"))


def test_final_defaults_minimal_three_hyperparameters():
    obsolete = (
        "mf_alpha",
        "residual_corr_threshold",
        "separation_aware",
        "min_tau_1",
        "tau_balance",
        "blend_kappa",
        "tau_1_cap",
        "tau_0_floor",
        "max_peeling_depth",
        "decaying_threshold",
        "threshold_decay_step",
        "aug_normalization",
        "peeling_metric",
        "calibration_method",
        "calibrate_tail",
        "use_prior_adaptive",
        "precision_guard",
    )
    for key in obsolete:
        assert key not in CONFIG["gsi_v6_3_3"]
        assert key not in CONFIG["gsi_common"]
    model = make_model("GSI_v6_3_3", "logistic", 42, CONFIG)
    assert model.cost == 0.30
    assert model.anchor_threshold == 0.75
    assert model.gamma_min == 0.70
    assert model.cv_folds == 5


@pytest.mark.parametrize("learner", CONFIG["base_learners"])
def test_identical_binary_estimators_for_all_families(learner):
    seed = 45
    models = {name: make_model(name, learner, seed, CONFIG) for name in CONFIG["models"]}
    expected = models["BR"].base_estimator.get_params()
    estimators = [models["CC"].base_estimator, models["MLC_PA"].base_estimator.base_estimator]
    estimators += [models[name]._create_binary_model() for name in CONFIG["models"] if name.startswith("GSI")]
    for estimator in estimators:
        params = estimator.get_params()
        assert params.keys() == expected.keys()
        for key, value in expected.items():
            if hasattr(value, "get_params"):
                assert params[key].get_params() == value.get_params()
            else:
                assert params[key] == value


def test_metrics_distinguish_rejection_from_false_negative():
    truth = np.array([[1, 0], [1, 1]], dtype=np.int32)
    full = np.array([[0, 0], [1, 1]], dtype=np.int32)
    partial = np.array([[-1, 0], [1, 1]], dtype=np.int32)
    metrics = metric_values(truth, full, partial, 0.3)
    assert metrics["Selective_Macro_F1"] == 1.0
    assert metrics["Full_Macro_F1"] == pytest.approx(5 / 6)
    assert metrics["Coverage"] == 0.75
    assert metrics["Generalized_Loss"] == pytest.approx(0.075)


def test_invalid_prediction_artifact_is_rejected():
    y = np.zeros((2, 2), dtype=np.int32)
    with pytest.raises(ValueError, match="Probabilities"):
        validate_predictions(y, np.full((2, 2), np.nan), y, y)
    with pytest.raises(ValueError, match="shape"):
        validate_predictions(y, np.zeros((1, 2)), y, y)


@pytest.mark.parametrize("name", CONFIG["models"])
def test_all_final_models_fit_synthetic_fold_and_emit_al_cl_records(name):
    rng = np.random.RandomState(8)
    X = rng.normal(size=(80, 5)).astype(np.float32)
    Y = np.column_stack([X[:, 0] > 0, X[:, 1] > 0, X[:, 2] > 1]).astype(np.int32)
    model = make_model(name, "logistic", 42, CONFIG)
    model.fit(X[:60], Y[:60])
    probabilities = model.predict_proba(X[60:])
    full = model.predict_full(X[60:]) if hasattr(model, "predict_full") else model.predict(X[60:])
    partial = model.predict(X[60:])
    validate_predictions(Y[60:], probabilities, full, partial)
    assert 0 <= metric_values(Y[60:], full, partial, 0.3)["Coverage"] <= 1

    records, _, partitions = label_records(Y[60:], full, partial, Y[:60], ["L0", "L1", "L2"], model)
    if name == "GSI_v6_3_3":
        assert {r["Group"] for r in records}.issubset({"AL", "CL"})
        assert all("AL_Layer" in p and "Dependent_Parents" not in p for p in partitions)


def test_frozen_baselines_reuse_br_cc_mlc_pa_and_only_fit_gsi(tmp_path, monkeypatch):
    from scripts import run_v6_3_3_final as runner
    from scripts.verify_v6_3_3_results import verify

    if not runner.DEFAULT_FROZEN_BASELINES_DIR.is_dir():
        pytest.skip("Frozen reference run is not installed")

    fitted_models = []
    orig_make_model = runner.make_model

    def spy_make_model(name, learner, seed, cfg):
        fitted_models.append(name)
        return orig_make_model(name, learner, seed, cfg)

    monkeypatch.setattr(runner, "make_model", spy_make_model)
    run_dir = runner.execute(
        ROOT / "configs/v6_3_3_final.json",
        tmp_path,
        datasets=["viruspseaac"],
        learners=["logistic"],
        folds=[1],
        allow_dirty=True,
        frozen_baselines_dir=runner.DEFAULT_FROZEN_BASELINES_DIR,
    )

    # Only GSI_v6_3_3 is fitted; BR, CC, MLC_PA are copied byte-for-byte
    assert fitted_models == ["GSI_v6_3_3"]
    for baseline in ("BR", "CC", "MLC_PA"):
        stem = f"viruspseaac__logistic__{baseline}__fold1.npz"
        assert (run_dir / "predictions" / stem).read_bytes() == (
            runner.DEFAULT_FROZEN_BASELINES_DIR / "predictions" / stem
        ).read_bytes()
    verification = verify(run_dir, check_source=True)
    assert verification["status"] == "verified"
    assert verification["verified_cells"] == 4
