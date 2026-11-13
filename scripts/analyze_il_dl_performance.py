"""
Script: Analyze and Evaluate IL vs DL Performance in GSI-MLC-PA v6.3.3.

This script runs 5-Fold CV across benchmark datasets using the current mechanism (v6.3.3)
and breaks down performance separately for:
1. IL (Independent Labels set)
2. DL (Dependent Labels set)
3. Full System (All K labels)

It calculates:
- Label distribution: |IL|, |DL|, %IL, %DL
- Performance metrics: Selective Macro-F1, Full Macro-F1, Coverage,
  Selective Micro-F1, Selective Hamming Loss, Full Hamming Loss
- Performance contribution to the final Selective Macro-F1:
  Contrib_IL = (|IL| / K) * Sel_F1_IL
  Contrib_DL = (|DL| / K) * Sel_F1_DL
  Share_IL (%) = Contrib_IL / Sel_F1_All * 100
  Share_DL (%) = Contrib_DL / Sel_F1_All * 100
- Performance gap between DL and IL: Delta_F1 = Sel_F1_DL - Sel_F1_IL
- Algorithmic gain of OS-NMF on DL (DL before OS-NMF vs DL after OS-NMF)
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, precision_score
from sklearn.preprocessing import MaxAbsScaler

# Ensure workspace root in path
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.loader import load_dataset
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_selective_macro_f1,
    compute_selective_micro_f1,
)
from src.models.gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier
from src.selection.tri_regime_decision import apply_tri_regime_abstention

DEFAULT_DATASETS = [
    "emotions",
    "scene",
    "yeast",
    "plantpseaac",
    "humanpseaac",
    "chd49",
    "music",
    "gpositivepseaac",
    "genbase",
    "viruspseaac",
]


def evaluate_subset_metrics(
    y_true_sub: np.ndarray,
    sel_preds_sub: np.ndarray,
    full_preds_sub: np.ndarray,
    abstain_value: int = -1,
) -> Dict[str, float]:
    """Compute selective and full metrics for a subset of labels."""
    n_samples, n_labels = y_true_sub.shape
    if n_labels == 0:
        return {
            "num_labels": 0,
            "coverage": np.nan,
            "selective_macro_f1": np.nan,
            "full_macro_f1": np.nan,
            "selective_micro_f1": np.nan,
            "full_micro_f1": np.nan,
            "selective_hamming_loss": np.nan,
            "full_hamming_loss": np.nan,
        }

    # Coverage
    decided = sel_preds_sub != abstain_value
    coverage = float(np.mean(decided))

    # Selective Hamming Loss
    if np.any(decided):
        sel_hamming = float(np.mean(y_true_sub[decided] != sel_preds_sub[decided]))
    else:
        sel_hamming = 0.0

    # Full Hamming Loss
    full_hamming = float(hamming_loss(y_true_sub, full_preds_sub))

    # Selective Macro-F1
    sel_macro_f1 = float(compute_selective_macro_f1(y_true_sub, sel_preds_sub, abstain_value=abstain_value))

    # Full Macro-F1
    full_macro_f1 = float(f1_score(y_true_sub, full_preds_sub, average="macro", zero_division=0))

    # Selective Micro-F1
    sel_micro_f1 = float(compute_selective_micro_f1(y_true_sub, sel_preds_sub, abstain_value=abstain_value))

    # Full Micro-F1
    full_micro_f1 = float(f1_score(y_true_sub, full_preds_sub, average="micro", zero_division=0))

    return {
        "num_labels": int(n_labels),
        "coverage": coverage,
        "selective_macro_f1": sel_macro_f1,
        "full_macro_f1": full_macro_f1,
        "selective_micro_f1": sel_micro_f1,
        "full_micro_f1": full_micro_f1,
        "selective_hamming_loss": sel_hamming,
        "full_hamming_loss": full_hamming,
    }


def evaluate_dl_raw_br(
    model: GSIMLCPAv6_3_3Classifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dl_labels: List[int],
) -> Dict[str, float]:
    """
    Evaluate DL labels before OS-NMF refinement (raw BR on FS_context).
    This measures the baseline performance of DL without relational coupling.
    """
    if len(dl_labels) == 0:
        return {"dl_raw_sel_f1": np.nan, "dl_raw_coverage": np.nan}

    # Reconstruct FS_context on test
    X = np.asarray(X_test, dtype=np.float32)
    current_test_X = X
    accumulated_il_test_probs: List[np.ndarray] = []

    for layer_idx, layer in enumerate(model.independent_layers_):
        layer_models = model.br_il_layer_models_[layer_idx]
        for lbl in layer:
            m = layer_models[lbl]
            p_l = model._predict_prob_single(m, current_test_X)
            accumulated_il_test_probs.append(p_l)

        if accumulated_il_test_probs:
            p_raw = np.column_stack(accumulated_il_test_probs)
            from src.selection.stratified_peeling import normalize_augmented_probabilities
            p_norm = normalize_augmented_probabilities(
                p_raw, reference_X=X, strategy=model.aug_normalization
            )
            current_test_X = np.hstack([X, p_norm])

    FS_context_test = current_test_X

    # Raw base DL probabilities (without OS-NMF)
    P_base_DL = np.column_stack([
        model._predict_prob_single(model.br_base_dl_models_[lbl], FS_context_test)
        for lbl in dl_labels
    ])

    # Calibrate if model calibrator is present
    if model.calibrate_tail and model.calibrator_ is not None:
        # Construct full dummy matrix to pass through calibrator transform
        P_full_dummy = np.zeros((X.shape[0], model.n_labels_), dtype=np.float32)
        P_full_dummy[:, dl_labels] = P_base_DL
        P_full_transformed = model.calibrator_.transform(P_full_dummy)
        P_dl_calibrated = P_full_transformed[:, dl_labels]
    else:
        P_dl_calibrated = P_base_DL

    # Apply decision thresholds for DL labels
    dl_thresholds = {j: model.adaptive_thresholds_[dl_labels[j]] for j in range(len(dl_labels))}
    dl_sel_preds = apply_tri_regime_abstention(
        np.clip(P_dl_calibrated, 0.0, 1.0).astype(np.float32),
        dl_thresholds,
        abstain_value=model.abstain_value,
    )
    y_test_dl = y_test[:, dl_labels]

    sel_f1 = float(compute_selective_macro_f1(y_test_dl, dl_sel_preds, abstain_value=model.abstain_value))
    decided = dl_sel_preds != model.abstain_value
    cov = float(np.mean(decided))

    return {
        "dl_raw_sel_f1": sel_f1,
        "dl_raw_coverage": cov,
    }


def run_il_dl_analysis(
    datasets: List[str] = DEFAULT_DATASETS,
    base_learner: str = "logistic",
    n_splits: int = 5,
    random_state: int = 42,
    cost: float = 0.30,
    output_dir: Path = WORKSPACE_ROOT / "results_v6_3" / "il_dl_analysis",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run comprehensive 5-fold evaluation breaking down IL and DL."""
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 90)
    print(f"IL vs DL EVALUATION & DECOMPOSITION ANALYSIS (GSI-MLC-PA v6.3.3)")
    print(f"Base Learner: {base_learner} | Folds: {n_splits} | Cost c = {cost:.2f}")
    print(f"Datasets: {datasets}")
    print("=" * 90)

    for dataset_name in datasets:
        print(f"\n---> Running Dataset: {dataset_name} ...", flush=True)
        t_start = time.time()

        X, Y, _, _ = load_dataset(dataset_name, base_dir=str(WORKSPACE_ROOT))
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)
        n_samples, K = Y.shape

        cv = get_multilabel_cv(n_splits=n_splits, random_state=random_state)
        fold_indices = list(cv.split(X, Y))

        for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
            X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
            Y_tr, Y_te = Y[train_idx], Y[test_idx]

            scaler = MaxAbsScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_te = scaler.transform(X_te_raw)

            seed = random_state + fold_idx

            model = GSIMLCPAv6_3_3Classifier(
                base_learner=base_learner,
                mf_alpha=0.25,
                mf_z_max=0.5,
                tau_balance=0.85,
                blend_kappa=20.0,
                cost=cost,
                gamma_min=0.70,
                tau_1_cap=0.55,
                tau_0_floor=0.45,
                random_state=seed,
            )
            model.fit(X_tr, Y_tr)

            # Predictions on full test fold
            full_preds = model.predict_full(X_te)
            sel_preds = model.predict(X_te, cost=cost)

            il_labels = sorted(list(model.independent_labels_))
            dl_labels = sorted(list(model.dependent_labels_))
            k_il = len(il_labels)
            k_dl = len(dl_labels)

            # 1. Overall System Metrics
            all_metrics = evaluate_subset_metrics(Y_te, sel_preds, full_preds, model.abstain_value)
            all_metrics["subset_accuracy"] = float(np.mean(np.all(Y_te == full_preds, axis=1)))

            # 2. IL Subset Metrics
            if k_il > 0:
                il_metrics = evaluate_subset_metrics(
                    Y_te[:, il_labels],
                    sel_preds[:, il_labels],
                    full_preds[:, il_labels],
                    model.abstain_value,
                )
            else:
                il_metrics = {k: np.nan for k in all_metrics}
                il_metrics["num_labels"] = 0

            # 3. DL Subset Metrics
            if k_dl > 0:
                dl_metrics = evaluate_subset_metrics(
                    Y_te[:, dl_labels],
                    sel_preds[:, dl_labels],
                    full_preds[:, dl_labels],
                    model.abstain_value,
                )
                # Raw DL without OS-NMF
                raw_dl = evaluate_dl_raw_br(model, X_te, Y_te, dl_labels)
            else:
                dl_metrics = {k: np.nan for k in all_metrics}
                dl_metrics["num_labels"] = 0
                raw_dl = {"dl_raw_sel_f1": np.nan, "dl_raw_coverage": np.nan}

            # 4. Mathematical Contributions to Final Selective Macro-F1
            w_il = k_il / K
            w_dl = k_dl / K
            f1_all = all_metrics["selective_macro_f1"]
            f1_il = il_metrics["selective_macro_f1"] if k_il > 0 else 0.0
            f1_dl = dl_metrics["selective_macro_f1"] if k_dl > 0 else 0.0

            contrib_il = w_il * f1_il
            contrib_dl = w_dl * f1_dl
            share_il = (contrib_il / f1_all * 100.0) if f1_all > 1e-6 else 0.0
            share_dl = (contrib_dl / f1_all * 100.0) if f1_all > 1e-6 else 0.0

            f1_gap = (f1_dl - f1_il) if (k_il > 0 and k_dl > 0) else np.nan
            cov_gap = (dl_metrics["coverage"] - il_metrics["coverage"]) if (k_il > 0 and k_dl > 0) else np.nan
            os_nmf_gain = (dl_metrics["selective_macro_f1"] - raw_dl["dl_raw_sel_f1"]) if k_dl > 0 else np.nan

            detailed_rows.append({
                "Dataset": dataset_name,
                "Fold": fold_idx + 1,
                "Total_Labels_K": K,
                "IL_Count": k_il,
                "DL_Count": k_dl,
                "IL_Ratio": w_il,
                "DL_Ratio": w_dl,
                # All Metrics
                "All_Sel_Macro_F1": all_metrics["selective_macro_f1"],
                "All_Full_Macro_F1": all_metrics["full_macro_f1"],
                "All_Coverage": all_metrics["coverage"],
                "All_Sel_Micro_F1": all_metrics["selective_micro_f1"],
                "All_Full_Micro_F1": all_metrics["full_micro_f1"],
                "All_Sel_Hamming": all_metrics["selective_hamming_loss"],
                "All_Full_Hamming": all_metrics["full_hamming_loss"],
                "All_Subset_Acc": all_metrics["subset_accuracy"],
                # IL Metrics
                "IL_Sel_Macro_F1": il_metrics["selective_macro_f1"],
                "IL_Full_Macro_F1": il_metrics["full_macro_f1"],
                "IL_Coverage": il_metrics["coverage"],
                "IL_Sel_Micro_F1": il_metrics["selective_micro_f1"],
                "IL_Sel_Hamming": il_metrics["selective_hamming_loss"],
                # DL Metrics
                "DL_Sel_Macro_F1": dl_metrics["selective_macro_f1"],
                "DL_Full_Macro_F1": dl_metrics["full_macro_f1"],
                "DL_Coverage": dl_metrics["coverage"],
                "DL_Sel_Micro_F1": dl_metrics["selective_micro_f1"],
                "DL_Sel_Hamming": dl_metrics["selective_hamming_loss"],
                "DL_Raw_Sel_Macro_F1": raw_dl["dl_raw_sel_f1"],
                "DL_OS_NMF_Gain": os_nmf_gain,
                # Contributions & Decompositions
                "Contrib_IL": contrib_il,
                "Contrib_DL": contrib_dl,
                "Share_IL_pct": share_il,
                "Share_DL_pct": share_dl,
                "Delta_F1_DL_minus_IL": f1_gap,
                "Delta_Cov_DL_minus_IL": cov_gap,
            })

            print(
                f"  Fold {fold_idx + 1}/{n_splits} | "
                f"K={K} (IL={k_il}, DL={k_dl}) | "
                f"IL Sel-F1: {il_metrics['selective_macro_f1']:.4f} | "
                f"DL Sel-F1: {dl_metrics['selective_macro_f1']:.4f} | "
                f"All Sel-F1: {all_metrics['selective_macro_f1']:.4f} | "
                f"Contrib: IL={share_il:.1f}%, DL={share_dl:.1f}%",
                flush=True,
            )

        print(f"  Finished {dataset_name} in {time.time() - t_start:.2f}s")

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_csv = output_dir / "il_dl_detailed_folds.csv"
    detailed_df.to_csv(detailed_csv, index=False)
    print(f"\n[Saved] Detailed fold results: {detailed_csv}")

    # Dataset Summary Aggregation
    summary_cols = [
        "Total_Labels_K",
        "IL_Count",
        "DL_Count",
        "IL_Ratio",
        "DL_Ratio",
        "All_Sel_Macro_F1",
        "All_Coverage",
        "All_Full_Macro_F1",
        "IL_Sel_Macro_F1",
        "IL_Coverage",
        "IL_Full_Macro_F1",
        "DL_Sel_Macro_F1",
        "DL_Coverage",
        "DL_Full_Macro_F1",
        "DL_Raw_Sel_Macro_F1",
        "DL_OS_NMF_Gain",
        "Contrib_IL",
        "Contrib_DL",
        "Share_IL_pct",
        "Share_DL_pct",
        "Delta_F1_DL_minus_IL",
        "Delta_Cov_DL_minus_IL",
    ]

    agg_dict = {}
    for col in summary_cols:
        agg_dict[f"{col}_Mean"] = (col, "mean")
        agg_dict[f"{col}_Std"] = (col, "std")

    summary_df = detailed_df.groupby("Dataset").agg(**agg_dict).reset_index()
    summary_csv = output_dir / "il_dl_dataset_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[Saved] Summary results: {summary_csv}")

    # Concise Presentation Table
    presentation_df = pd.DataFrame({
        "Dataset": summary_df["Dataset"],
        "K": summary_df["Total_Labels_K_Mean"].astype(int),
        "K_IL": summary_df["IL_Count_Mean"].round(1),
        "K_DL": summary_df["DL_Count_Mean"].round(1),
        "IL_Ratio (%)": (summary_df["IL_Ratio_Mean"] * 100).round(1),
        "DL_Ratio (%)": (summary_df["DL_Ratio_Mean"] * 100).round(1),
        "IL Sel-F1": summary_df["IL_Sel_Macro_F1_Mean"].round(4),
        "DL Sel-F1": summary_df["DL_Sel_Macro_F1_Mean"].round(4),
        "All Sel-F1": summary_df["All_Sel_Macro_F1_Mean"].round(4),
        "Contrib IL": summary_df["Contrib_IL_Mean"].round(4),
        "Contrib DL": summary_df["Contrib_DL_Mean"].round(4),
        "Share IL (%)": summary_df["Share_IL_pct_Mean"].round(1),
        "Share DL (%)": summary_df["Share_DL_pct_Mean"].round(1),
        "Gap (DL - IL)": summary_df["Delta_F1_DL_minus_IL_Mean"].round(4),
        "IL Coverage": (summary_df["IL_Coverage_Mean"] * 100).round(1),
        "DL Coverage": (summary_df["DL_Coverage_Mean"] * 100).round(1),
        "All Coverage": (summary_df["All_Coverage_Mean"] * 100).round(1),
        "OS-NMF Gain": summary_df["DL_OS_NMF_Gain_Mean"].round(4),
    })

    # Sort consistent with benchmark standard
    presentation_df["Dataset"] = pd.Categorical(
        presentation_df["Dataset"], categories=DEFAULT_DATASETS, ordered=True
    )
    presentation_df = presentation_df.sort_values("Dataset").reset_index(drop=True)

    concise_csv = output_dir / "il_dl_presentation_summary.csv"
    presentation_df.to_csv(concise_csv, index=False)
    print(f"[Saved] Presentation summary: {concise_csv}")

    print("\n" + "=" * 90)
    print("CONCISE SUMMARY TABLE: IL vs DL PERFORMANCE & CONTRIBUTION")
    print("=" * 90)
    print(presentation_df.to_string(index=False))

    return detailed_df, presentation_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate IL vs DL performance in GSI-MLC-PA v6.3.3")
    parser.add_argument("--base_learner", type=str, default="logistic", choices=["logistic", "svm_calibrated", "mlp"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cost", type=float, default=0.30)
    args = parser.parse_args()

    run_il_dl_analysis(
        base_learner=args.base_learner,
        n_splits=args.folds,
        cost=args.cost,
    )
