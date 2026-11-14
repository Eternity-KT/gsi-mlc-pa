import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_final import ROOT, command_env, locked_versions, prediction_state, run_step, source_consistency


def test_final_lock_covers_runtime_and_pytest():
    pins = locked_versions()
    assert pins["torch"] == "2.6.0+cu124"
    assert pins["joblib"] == "1.5.2"
    assert pins["pytest"] == "9.1.1"
    assert "pyarrow" not in pins


def test_child_processes_do_not_load_external_pytest_plugins(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "unrelated-global-packages")
    env = command_env()
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in env


def test_check_command_stops_on_failure():
    with pytest.raises(RuntimeError, match="exit code 7"):
        run_step([sys.executable, "-c", "raise SystemExit(7)"], 10, capture=True)


def test_check_command_has_a_real_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_step([sys.executable, "-c", "import time; time.sleep(5)"], 0.2, capture=True)


def test_final_imports_do_not_depend_on_future_versions():
    for relative in ["src/models/__init__.py", "src/models/gsi_v6_3_3.py", "scripts/run_v6_3_3_final.py"]:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "gsi_v6_3_4" not in (node.module or "")
                assert "gsi_v6_3_5" not in (node.module or "")


def test_completed_predictions_are_hash_and_timestamp_checked(tmp_path):
    (tmp_path / "predictions").mkdir()
    prediction = tmp_path / "predictions" / "fold.npz"
    prediction.write_bytes(b"first snapshot")
    before = prediction_state(tmp_path)
    prediction.write_bytes(b"different snapshot")
    assert prediction_state(tmp_path) != before


def test_p0_source_bytes_still_match_the_verified_benchmark():
    manifest_path = ROOT / "results/v6_3_3_final_four_models/run_6c5fb2bebb16a7dc/run_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("Published benchmark artifacts are not installed")
    result = source_consistency(manifest_path.parent)
    assert result["checked_source_files"] > 0


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "model.py"
    original = b"value = 1\r\n"
    source.write_bytes(original)
    artifact = tmp_path / "run"
    artifact.mkdir()
    (artifact / "run_manifest.json").write_text(json.dumps({
        "source_commit": "1" * 40,
        "identity": {"source_files": {"model.py": hashlib.sha256(original).hexdigest()}},
    }), encoding="utf-8")
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: b"value = 1\n")
    return artifact, source_root, source


def test_source_guard_accepts_exact_original_bytes(source_tree):
    artifact, root, _ = source_tree
    assert source_consistency(artifact, root)["eol_normalized_files"] == 0


def test_source_guard_accepts_git_confirmed_lf_crlf_changes(source_tree):
    artifact, root, source = source_tree
    source.write_bytes(b"value = 1\n")
    assert source_consistency(artifact, root)["eol_normalized_files"] == 1


def test_source_guard_rejects_actual_code_changes(source_tree):
    artifact, root, source = source_tree
    source.write_bytes(b"value = 2\n")
    with pytest.raises(ValueError, match="Source content changed"):
        source_consistency(artifact, root)


@pytest.fixture
def checkpoint_run(tmp_path, monkeypatch):
    """Exercise the real checkpoint writer with lightweight probability models."""
    import numpy as np
    from scripts import run_v6_3_3_final as runner

    if not (ROOT / "data/VirusPseAAC/VirusPseAAC207-train.txt").is_file():
        pytest.skip("VirusPseAAC is not installed")
    fitted = {"count": 0}

    class FixedModel:
        def fit(self, X, Y):
            fitted["count"] += 1
            self.n_labels = Y.shape[1]
            return self

        def predict_proba(self, X):
            return np.full((len(X), self.n_labels), 0.8)

        def predict_full(self, X):
            return (self.predict_proba(X) >= 0.5).astype(np.int32)

        predict = predict_full

    monkeypatch.setattr(runner, "make_model", lambda *args: FixedModel())
    config = json.loads((ROOT / "configs/v6_3_3_final.json").read_text(encoding="utf-8"))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def execute():
        return runner.execute(config_path, tmp_path / "runs", ["viruspseaac"], ["logistic"], [1], allow_dirty=True)

    return execute, config_path, fitted


def test_identical_run_resumes_without_refitting(checkpoint_run):
    execute, config_path, fitted = checkpoint_run
    model_count = len(json.loads(config_path.read_text(encoding="utf-8"))["models"])
    first = execute()
    before = prediction_state(first)
    assert fitted["count"] == model_count
    assert execute() == first
    assert fitted["count"] == model_count
    assert prediction_state(first) == before


def test_changed_config_creates_new_run_and_preserves_previous_predictions(checkpoint_run):
    execute, config_path, fitted = checkpoint_run
    model_count = len(json.loads(config_path.read_text(encoding="utf-8"))["models"])
    first = execute()
    before = prediction_state(first)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["gsi_v6_3_3"]["blend_kappa"] = 21.0
    config_path.write_text(json.dumps(config), encoding="utf-8")
    second = execute()
    assert first != second
    assert fitted["count"] == 2 * model_count
    assert prediction_state(first) == before


def test_corrupt_prediction_cannot_be_reused(checkpoint_run):
    execute, config_path, fitted = checkpoint_run
    model_count = len(json.loads(config_path.read_text(encoding="utf-8"))["models"])
    first = execute()
    path = next((first / "predictions").glob("*__BR__*.npz"))
    path.write_bytes(b"corrupted checkpoint")
    with pytest.raises(ValueError, match="Checkpoint prediction mismatch"):
        execute()
    assert fitted["count"] == model_count
