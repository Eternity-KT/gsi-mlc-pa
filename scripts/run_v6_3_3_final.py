"""Frozen, resumable benchmark; writes numeric artifacts and provenance only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MaxAbsScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_v6_3_3_datasets import load_manifest, validate_dataset, validate_summary
from src.data.loader import canonicalize_dataset_name, load_dataset
from src.evaluation.metrics import (
    compute_all_metrics, compute_partial_abstention_metrics,
    compute_selective_macro_f1, compute_selective_micro_f1,
)
from src.models.base_learners import create_binary_estimator, load_experiment_config
from src.models.binary_relevance import BinaryRelevanceClassifier
from src.models.classifier_chain import ClassifierChainClassifier
from src.models.gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier
from src.models.mlc_pa import MLCPartialAbstentionClassifier


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def json_bytes(value):
    return json.dumps(clean(value), sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def source_fingerprints():
    files = list((ROOT / "src").rglob("*.py"))
    files += [ROOT / "scripts" / name for name in (
        "run_v6_3_3_final.py", "verify_v6_3_3_results.py", "validate_v6_3_3_datasets.py",
    )]
    files += list((ROOT / "configs").glob("*.json"))
    files += [ROOT / "requirements-final.lock.txt"]
    return {p.relative_to(ROOT).as_posix(): digest_file(p) for p in sorted(files) if p.is_file()}


def environment():
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "iterative-stratification", "liac-arff", "torch", "threadpoolctl"]
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def make_model(name, learner, seed, config):
    binary = create_binary_estimator(learner, random_state=seed)
    if name == "BR":
        return BinaryRelevanceClassifier(base_estimator=binary, random_state=seed)
    if name == "CC":
        return ClassifierChainClassifier(base_estimator=binary, order=None, random_state=seed)
    if name == "MLC_PA":
        marginal = BinaryRelevanceClassifier(base_estimator=binary, random_state=seed)
        return MLCPartialAbstentionClassifier(base_estimator=marginal, random_state=seed, **config["mlc_pa"])
    common = {**config["gsi_common"], "base_learner": learner, "random_state": seed}
    if name == "GSI_v6_3_3":
        return GSIMLCPAv6_3_3Classifier(**common, **config["gsi_v6_3_3"])
    raise ValueError(f"Unknown model: {name}")


def metric_values(truth, full, partial, cost):
    full_metrics = compute_all_metrics(truth, full)
    partial_metrics = compute_partial_abstention_metrics(truth, partial, cost=cost, penalty="linear")
    return {
        "Full_Macro_F1": full_metrics["Macro-F1"],
        "Full_Micro_F1": full_metrics["Micro-F1"],
        "Full_Example_F1": full_metrics["Example-F1"],
        "Hamming_Loss": full_metrics["Hamming Loss"],
        "Subset_Accuracy": full_metrics["Subset Accuracy"],
        "Selective_Macro_F1": compute_selective_macro_f1(truth, partial),
        "Selective_Micro_F1": compute_selective_micro_f1(truth, partial),
        "Selective_Hamming_Loss": float(np.mean(partial[partial != -1] != truth[partial != -1])) if np.any(partial != -1) else None,
        "Generalized_Loss": partial_metrics["Generalized Loss"],
        "Coverage": float(np.mean(partial != -1)),
        "ABS": partial_metrics["ABS"],
        "AABS": partial_metrics["AABS"],
    }


def _f1(tp, fp, fn):
    return float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0


def label_records(truth, full, partial, y_train, label_names, model):
    records, thresholds, partitions = [], [], []
    il = set(getattr(model, "independent_labels_", []))
    dl = set(getattr(model, "dependent_labels_", []))
    layers = {label: depth for depth, labels in enumerate(getattr(model, "independent_layers_", []), 1) for label in labels}
    adaptive = getattr(model, "adaptive_thresholds_", {})
    for label, label_name in enumerate(label_names):
        y, p = truth[:, label], partial[:, label]
        tp = int(np.sum((y == 1) & (p == 1)))
        fp = int(np.sum((y == 0) & (p == 1)))
        fn = int(np.sum((y == 1) & (p == 0)))
        tn = int(np.sum((y == 0) & (p == 0)))
        group = "IL" if label in il else ("DL" if label in dl else "NA")
        records.append({
            "Label": label, "Label_Name": label_name, "Group": group,
            "Train_Positives": int(y_train[:, label].sum()),
            "Train_Negatives": int(len(y_train) - y_train[:, label].sum()),
            "Test_Positives": int(y.sum()), "Test_Negatives": int(len(y) - y.sum()),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Abstained_Positive": int(np.sum((y == 1) & (p == -1))),
            "Abstained_Negative": int(np.sum((y == 0) & (p == -1))),
            "Coverage": float(np.mean(p != -1)),
            "Selective_F1_Positive": _f1(tp, fp, fn),
            "Selective_F1_Negative": _f1(tn, fn, fp),
            "Full_F1": _f1(int(np.sum((y == 1) & (full[:, label] == 1))), int(np.sum((y == 0) & (full[:, label] == 1))), int(np.sum((y == 1) & (full[:, label] == 0)))),
        })
        if label in adaptive:
            th = dict(adaptive[label])
            full_t = 0.5
            if isinstance(model, GSIMLCPAv6_3_3Classifier):
                if th["prior"] <= 0.2:
                    full_t = th["tau_1"]
                elif th["prior"] >= 0.8:
                    full_t = th["tau_0"]
        else:
            c = getattr(model, "cost", None)
            th = {"tau_0": c, "tau_1": 1.0 - c if c is not None else None, "prior": float(y_train[:, label].mean()), "regime": "symmetric_chow" if c is not None else "native_binary"}
            full_t = 0.5
        thresholds.append({"Label": label, "Full_Threshold": full_t, "SAMG": False, **th})
        partitions.append({"Label": label, "Label_Name": label_name, "Group": group, "IL_Layer": layers.get(label), "Dependent_Parents": json.dumps(getattr(model, "dependency_graph_", {}).get(label, []))})
    return records, thresholds, partitions


def validate_predictions(truth, probabilities, full, partial):
    if not (truth.shape == probabilities.shape == full.shape == partial.shape):
        raise ValueError("Prediction shape mismatch")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not np.isin(full, [0, 1]).all() or not np.isin(partial, [-1, 0, 1]).all():
        raise ValueError("Invalid predicted labels")


def aggregate(run_dir):
    cells = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run_dir / "cells").glob("*.json"))]
    rows, labels, thresholds, partitions = [], [], [], []
    for cell in cells:
        key = cell["key"]
        rows.append({**key, **cell["metrics"]})
        for target, field in ((labels, "per_label"), (thresholds, "thresholds"), (partitions, "partitions")):
            target.extend({**key, **row} for row in cell[field])
    detailed = pd.DataFrame(rows)
    if detailed.empty:
        return 0
    detailed.to_csv(run_dir / "detailed_folds.csv", index=False)
    pd.DataFrame(labels).to_csv(run_dir / "per_label_folds.csv", index=False)
    pd.DataFrame(thresholds).to_csv(run_dir / "thresholds_audit.csv", index=False)
    pd.DataFrame(partitions).to_csv(run_dir / "partitions_audit.csv", index=False)
    group = ["Dataset", "Base_Learner", "Model"]
    metrics = [c for c in detailed.columns if c not in {*group, "Fold", "Seed"}]
    summary = detailed.groupby(group, sort=True)[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(run_dir / "dataset_summary.csv", index=False)
    dataset_means = detailed.groupby(group, sort=True)[metrics].mean().reset_index()
    dataset_means.groupby(["Base_Learner", "Model"])[metrics].mean().reset_index().to_csv(run_dir / "overall_summary.csv", index=False)
    return len(cells)


def write_checksums(run_dir):
    paths = [p for p in run_dir.rglob("*") if p.is_file() and p.name not in {"checksums.sha256", "runner.log"} and not p.name.endswith(".tmp")]
    lines = [f"{digest_file(p)}  {p.relative_to(run_dir).as_posix()}" for p in sorted(paths)]
    (run_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(config_path, output_root, datasets=None, learners=None, folds=None, allow_dirty=False):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["gsi_v6_3_3"]["separation_aware"] or config["variant"] != "standard_no_samg":
        raise ValueError("Final benchmark requires standard v6.3.3 without SAMG")
    selected_ds = [canonicalize_dataset_name(d) for d in (datasets or config["datasets"])]
    selected_learners = list(learners or config["base_learners"])
    selected_folds = list(folds or range(1, config["cv"]["outer_splits"] + 1))
    for values, allowed in ((selected_ds, config["datasets"]), (selected_learners, config["base_learners"]), (selected_folds, list(range(1, 6)))):
        if len(set(values)) != len(values) or not set(values).issubset(allowed):
            raise ValueError(f"Invalid or duplicate selection: {values}")
    paths = ["src", "configs", "scripts/run_v6_3_3_final.py", "scripts/verify_v6_3_3_results.py", "scripts/validate_v6_3_3_datasets.py", "tests/test_v6_3_3_final_runner.py", "requirements-final.lock.txt"]
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *paths)
    if dirty and not allow_dirty:
        raise RuntimeError(f"Commit source/config changes before the final run:\n{dirty}")
    runtime = config["runtime"]
    if "mlp" in selected_learners and runtime["require_cuda_for_mlp"] and not torch.cuda.is_available():
        raise RuntimeError("Frozen MLP configuration requires CUDA")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = runtime["cublas_workspace_config"]
    torch.set_num_threads(runtime["torch_threads"])
    torch.use_deterministic_algorithms(runtime["deterministic_torch"])
    torch.backends.cuda.matmul.allow_tf32 = runtime["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = runtime["allow_tf32"]
    torch.backends.cudnn.benchmark = False

    data_manifest = load_manifest(ROOT / config["dataset_manifest"])
    errors = validate_summary(ROOT / "data/dataset_manifest.csv", data_manifest)
    dataset_cache = {}
    for dataset in selected_ds:
        result = validate_dataset(dataset, data_manifest["datasets"][dataset], ROOT)
        errors.extend(result["errors"])
        dataset_cache[dataset] = load_dataset(dataset, base_dir=str(ROOT))
    if errors:
        raise ValueError(f"Dataset preflight failed: {errors}")

    identity = {
        "config": config, "config_sha256": digest_file(config_path),
        "source_files": source_fingerprints(), "environment": environment(),
        "data_manifest_sha256": digest_file(ROOT / config["dataset_manifest"]),
        "base_learner_config": load_experiment_config(),
        "selection": {"datasets": selected_ds, "base_learners": selected_learners, "folds": selected_folds},
    }
    run_id = hashlib.sha256(json_bytes(identity)).hexdigest()[:16]
    run_dir = Path(output_root).resolve() / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cells").mkdir(exist_ok=True)
    (run_dir / "predictions").mkdir(exist_ok=True)
    expected = len(selected_ds) * len(selected_learners) * len(selected_folds) * len(config["models"])
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != clean(identity):
            raise ValueError("Existing run identity differs; refusing to overwrite")
    else:
        manifest = {"run_id": run_id, "status": "running", "created_utc": utc_now(), "source_commit": git("rev-parse", "HEAD"), "source_branch": git("branch", "--show-current"), "source_dirty": bool(dirty), "identity": identity, "expected_cells": expected}
    manifest["status"] = "running"
    write_json(manifest_path, manifest)
    write_json(run_dir / "config_snapshot.json", config)
    write_json(run_dir / "dataset_manifest_snapshot.json", data_manifest)

    split_records = {}
    for dataset, (X, Y, _, _) in dataset_cache.items():
        cv = MultilabelStratifiedKFold(n_splits=config["cv"]["outer_splits"], shuffle=config["cv"]["shuffle"], random_state=config["cv"]["random_state"])
        split_records[dataset] = [{"fold": fold, "train_indices": tr.tolist(), "test_indices": te.tolist()} for fold, (tr, te) in enumerate(cv.split(X, Y), 1)]
    fold_path = run_dir / "fold_indices.json"
    if fold_path.exists() and json.loads(fold_path.read_text(encoding="utf-8")) != split_records:
        raise ValueError("Saved fold indices differ from current splits")
    write_json(fold_path, split_records)
    print(f"RUN_DIR={run_dir}\nExpected fits: {expected}", flush=True)
    completed = 0
    with threadpool_limits(limits=runtime["blas_threads"]):
        for dataset in selected_ds:
            X, Y, _, names = dataset_cache[dataset]
            for fold_record in split_records[dataset]:
                fold = fold_record["fold"]
                if fold not in selected_folds:
                    continue
                train = np.asarray(fold_record["train_indices"])
                test = np.asarray(fold_record["test_indices"])
                scaler = MaxAbsScaler().fit(X[train])
                x_train, x_test = scaler.transform(X[train]), scaler.transform(X[test])
                y_train, y_test = Y[train], Y[test]
                seed = config["cv"]["random_state"] + fold - 1
                for learner in selected_learners:
                    for name in config["models"]:
                        stem = f"{dataset}__{learner}__{name}__fold{fold}"
                        cell_path = run_dir / "cells" / f"{stem}.json"
                        prediction_path = run_dir / "predictions" / f"{stem}.npz"
                        if cell_path.exists():
                            cell = json.loads(cell_path.read_text(encoding="utf-8"))
                            if not prediction_path.is_file() or digest_file(prediction_path) != cell["prediction_sha256"]:
                                raise ValueError(f"Checkpoint prediction mismatch: {stem}")
                            completed += 1
                            continue
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)
                        model = make_model(name, learner, seed, config)
                        with warnings.catch_warnings(record=True) as captured:
                            warnings.simplefilter("always")
                            started = time.perf_counter()
                            model.fit(x_train, y_train)
                            fit_seconds = time.perf_counter() - started
                            started = time.perf_counter()
                            probabilities = np.asarray(model.predict_proba(x_test))
                            full = np.asarray(model.predict_full(x_test) if hasattr(model, "predict_full") else model.predict(x_test), dtype=np.int32)
                            partial = np.asarray(model.predict(x_test), dtype=np.int32) if name not in ("BR", "CC") else full.copy()
                            predict_seconds = time.perf_counter() - started
                        validate_predictions(y_test, probabilities, full, partial)
                        key = {"Dataset": dataset, "Base_Learner": learner, "Model": name, "Fold": fold, "Seed": seed}
                        metrics = metric_values(y_test, full, partial, config["mlc_pa"]["cost"])
                        records, thresholds, partitions = label_records(y_test, full, partial, y_train, names, model)
                        for group, indices in (("IL", getattr(model, "independent_labels_", [])), ("DL", getattr(model, "dependent_labels_", []))):
                            metrics[f"{group}_Label_Count"] = len(indices)
                            metrics[f"{group}_Coverage"] = float(np.mean(partial[:, indices] != -1)) if indices else None
                            metrics[f"{group}_Selective_Macro_F1"] = compute_selective_macro_f1(y_test[:, indices], partial[:, indices]) if indices else None
                        metrics.update(Fit_Time_s=fit_seconds, Predict_Time_s=predict_seconds)
                        temporary = prediction_path.with_suffix(".tmp")
                        with temporary.open("wb") as stream:
                            np.savez_compressed(stream, test_indices=test, y_true=y_test, probabilities=probabilities, full=full, partial=partial, scaler_max_abs=scaler.max_abs_)
                        temporary.replace(prediction_path)
                        write_json(cell_path, {"key": key, "metrics": metrics, "per_label": records, "thresholds": thresholds, "partitions": partitions, "prediction_sha256": digest_file(prediction_path), "warnings": sorted(set(f"{w.category.__name__}: {w.message}" for w in captured))})
                        completed += 1
                        manifest.update(completed_cells=completed, updated_utc=utc_now(), last_completed=key)
                        write_json(manifest_path, manifest)
                        print(f"[{completed}/{expected}] {dataset} {learner} fold={fold} {name}: F1={metrics['Selective_Macro_F1']:.4f} cov={metrics['Coverage']:.4f} fit={fit_seconds:.2f}s", flush=True)
                        del model
                    aggregate(run_dir)
    count = aggregate(run_dir)
    if count != expected:
        raise ValueError(f"Incomplete run: {count}/{expected}")
    manifest.update(status="complete", completed_cells=count, completed_utc=utc_now())
    write_json(manifest_path, manifest)
    write_checksums(run_dir)
    print(f"COMPLETE: {count}/{expected} fits; {run_dir}", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v6_3_3_final.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/v6_3_3_final_four_models")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--base-learners", nargs="+")
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--allow-dirty", action="store_true", help="Smoke tests only; records dirty source in provenance.")
    args = parser.parse_args()
    execute(args.config, args.output_root, args.datasets, args.base_learners, args.folds, args.allow_dirty)


if __name__ == "__main__":
    main()
