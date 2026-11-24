"""Verify numeric benchmark artifacts against saved out-of-sample predictions."""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_v6_3_3_final import digest_file, metric_values, validate_predictions


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(run_dir, check_source=True):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    require(manifest["status"] == "complete", "Run is incomplete")
    expected = manifest["expected_cells"]
    require(manifest["completed_cells"] == expected, "Completion count differs")
    identity = manifest["identity"]
    if check_source:
        for relative, expected_digest in identity["source_files"].items():
            require((ROOT / relative).is_file(), f"Missing source: {relative}")
            require(digest_file(ROOT / relative) == expected_digest, f"Changed source: {relative}")

    paths = set()
    for line in (run_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected_digest, relative = line.split("  ", 1)
        path = run_dir / relative
        require(path.is_file(), f"Missing artifact: {relative}")
        require(digest_file(path) == expected_digest, f"Changed artifact: {relative}")
        paths.add(relative)
    actual_paths = {p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file() and p.name not in {"checksums.sha256", "runner.log"} and not p.name.endswith(".tmp")}
    require(paths == actual_paths, "Checksum inventory has missing or unexpected artifacts")

    splits = json.loads((run_dir / "fold_indices.json").read_text(encoding="utf-8"))
    dataset_manifest = json.loads((run_dir / "dataset_manifest_snapshot.json").read_text(encoding="utf-8"))
    for dataset, records in splits.items():
        n_samples = dataset_manifest["datasets"][dataset]["expected"]["n_samples"]
        all_test = []
        for record in records:
            train, test = record["train_indices"], record["test_indices"]
            require(len(train) == len(set(train)) and len(test) == len(set(test)), f"Duplicate fold indices: {dataset}")
            require(not set(train).intersection(test), f"Train/test overlap: {dataset}")
            require(set(train).union(test) == set(range(n_samples)), f"Incomplete fold: {dataset}")
            all_test.extend(test)
        require(sorted(all_test) == list(range(n_samples)), f"Test folds do not cover dataset once: {dataset}")

    detailed = pd.read_csv(run_dir / "detailed_folds.csv")
    summary = pd.read_csv(run_dir / "dataset_summary.csv")
    per_label = pd.read_csv(run_dir / "per_label_folds.csv")
    thresholds = pd.read_csv(run_dir / "thresholds_audit.csv")
    partitions = pd.read_csv(run_dir / "partitions_audit.csv")
    selection = identity["selection"]
    config = identity["config"]
    columns = ["Dataset", "Base_Learner", "Model", "Fold"]
    selected_models = selection.get("models", config["models"])
    expected_keys = set(itertools.product(selection["datasets"], selection["base_learners"], selected_models, selection["folds"]))
    require(len(detailed) == expected == len(expected_keys), "Incorrect number of detailed rows")
    require(set(detailed[columns].itertuples(index=False, name=None)) == expected_keys, "Model/dataset/fold coverage differs")
    cells = sorted((run_dir / "cells").glob("*.json"))
    require(len(cells) == expected, "Incorrect checkpoint count")
    by_key = detailed.set_index(columns)
    warnings_count = 0
    for path in cells:
        cell = json.loads(path.read_text(encoding="utf-8"))
        key = cell["key"]
        key_tuple = tuple(key[col] for col in columns)
        prediction_path = run_dir / "predictions" / (path.stem + ".npz")
        require(digest_file(prediction_path) == cell["prediction_sha256"], f"Checkpoint hash differs: {path.name}")
        with np.load(prediction_path, allow_pickle=False) as arrays:
            truth, probabilities = arrays["y_true"], arrays["probabilities"]
            full, partial = arrays["full"], arrays["partial"]
            validate_predictions(truth, probabilities, full, partial)
            record = splits[key["Dataset"]][key["Fold"] - 1]
            require(np.array_equal(arrays["test_indices"], record["test_indices"]), f"Wrong test rows: {path.name}")
            actual = metric_values(truth, full, partial, config["mlc_pa"]["cost"])
            for metric, value in actual.items():
                saved = by_key.loc[key_tuple, metric]
                require((value is None and pd.isna(saved)) or (value is not None and np.isclose(value, saved, rtol=0, atol=1e-12)), f"Metric mismatch: {path.name} {metric}")
            require(key["Seed"] == config["cv"]["random_state"] + key["Fold"] - 1, f"Seed mismatch: {path.name}")
            require(len(cell["per_label"]) == truth.shape[1], f"Missing labels: {path.name}")
            for label in cell["per_label"]:
                decided_count = label["TP"] + label["TN"] + label["FP"] + label["FN"]
                require(decided_count + label["Abstained_Positive"] + label["Abstained_Negative"] == len(truth), f"Confusion count mismatch: {path.name}")
        warnings_count += len(cell["warnings"])

    expected_label_rows = sum(dataset_manifest["datasets"][d]["expected"]["n_labels"] for d in selection["datasets"]) * len(selection["base_learners"]) * len(selected_models) * len(selection["folds"])
    require(len(per_label) == len(thresholds) == len(partitions) == expected_label_rows, "Incomplete per-label audits")
    require(not thresholds["SAMG"].any(), "SAMG unexpectedly enabled")
    require(not detailed.duplicated(columns).any(), "Duplicate benchmark cells")
    metric_names = [col for col in detailed if col not in {*columns, "Seed"}]
    group = ["Dataset", "Base_Learner", "Model"]
    recomputed = detailed.groupby(group)[metric_names].agg(["mean", "std"])
    recomputed.columns = [f"{metric}_{stat}" for metric, stat in recomputed.columns]
    observed = summary.set_index(group).sort_index()
    pd.testing.assert_frame_equal(recomputed.sort_index(), observed[recomputed.columns], check_dtype=False, rtol=1e-10, atol=1e-12)
    return {
        "run_id": manifest["run_id"], "verified_cells": expected,
        "verified_prediction_files": len(cells), "verified_label_rows": expected_label_rows,
        "recorded_warning_messages": warnings_count,
        "v6_3_3_selective_macro_f1": detailed[detailed.Model == "GSI_v6_3_3"].Selective_Macro_F1.mean(),
        "v6_3_3_coverage": detailed[detailed.Model == "GSI_v6_3_3"].Coverage.mean(),
        "status": "verified",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--skip-source", action="store_true", help="Verify archived numeric artifacts without requiring this checkout to match.")
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir, check_source=not args.skip_source), indent=2))
