from scripts import run_experiments, verify_results
from scripts.run_v6_3_3_final import DEFAULT_FROZEN_BASELINES_DIR


def test_experiment_command_preserves_reference_outputs_and_forwards_selection(monkeypatch):
    calls = []
    monkeypatch.setattr(run_experiments, "execute", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert run_experiments.main(["--datasets", "emotions", "--base-learners", "logistic", "--folds", "1"]) == 0
    args, kwargs = calls[0]
    config, output_root, datasets, learners, folds, allow_dirty = args
    assert config == run_experiments.BENCHMARK_CONFIG
    assert output_root == run_experiments.ROOT / "results/reproductions"
    assert not verify_results.RELEASE.is_relative_to(output_root)
    assert (datasets, learners, folds, allow_dirty) == (["emotions"], ["logistic"], [1], False)
    assert kwargs["frozen_baselines_dir"] == DEFAULT_FROZEN_BASELINES_DIR
    assert kwargs["models"] is None


def test_experiment_command_supports_retrain_baselines_and_custom_paths(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run_experiments, "execute", lambda *args, **kwargs: calls.append((args, kwargs)))
    config = tmp_path / "protocol.json"
    output = tmp_path / "experiments"
    run_experiments.main(["--config", str(config), "--output-root", str(output), "--retrain-baselines"])
    args, kwargs = calls[0]
    assert args[:2] == (config, output)
    assert kwargs["frozen_baselines_dir"] is None


def test_verification_command_defaults_to_saved_reference(monkeypatch):
    calls = []
    monkeypatch.setattr(verify_results, "verify_artifacts", lambda *args: calls.append(args))
    assert verify_results.main([]) == 0
    assert calls == [(verify_results.RELEASE.resolve(), 600)]


def test_verification_command_accepts_a_reproduced_run(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(verify_results, "verify_artifacts", lambda *args: calls.append(args))
    verify_results.main([str(tmp_path / "run_example"), "--timeout", "120"])
    assert calls == [((tmp_path / "run_example").resolve(), 120)]
