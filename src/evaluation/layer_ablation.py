"""
Layer-Wise Ablation Framework for GSI-MLC-PA v5.1.1.

Implements the six historical v5.1.1 layer ablation regimes:
1. ABL_1_ONLY_IL1: Evaluate performance on IL_1 only.
2. ABL_2_ONLY_IL2: Evaluate performance on IL_2 only.
3. ABL_2_ACCUM_IL12: Evaluate performance on IL_1 + IL_2.
4. ABL_3_ONLY_IL3: Evaluate performance on IL_3 only.
5. ABL_3_ALL_IL: Evaluate performance on all IL = IL_1 + IL_2 + IL_3.
6. ABL_3_ONLY_DL: Evaluate performance on residual DL.
7. FULL_SYSTEM: Baseline reference on all K labels.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score

from .metrics import compute_selective_macro_f1, compute_selective_micro_f1


def _compute_selective_macro_precision(
    y_true: np.ndarray,
    y_partial: np.ndarray,
    abstain_value: int = -1,
) -> float:
    """Compute Macro-Precision restricted to decided positions per label."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_partial = np.asarray(y_partial, dtype=np.int32)
    n_labels = y_true.shape[1]

    precisions = []
    for j in range(n_labels):
        decided = y_partial[:, j] != abstain_value
        if not np.any(decided):
            precisions.append(0.0)
            continue
        prec = precision_score(y_true[decided, j], y_partial[decided, j], zero_division=0)
        precisions.append(float(prec))

    return float(np.mean(precisions)) if precisions else 0.0


def evaluate_label_subset(
    y_true: np.ndarray,
    y_full: np.ndarray,
    y_partial: np.ndarray,
    label_indices: Sequence[int],
    abstain_value: int = -1,
) -> Dict[str, Any]:
    """Evaluate full and selective metrics for a specified subset of labels."""
    label_indices = [int(l) for l in label_indices]
    if len(label_indices) == 0:
        return {
            "num_labels": 0,
            "labels": [],
            "coverage": np.nan,
            "selective_macro_f1": np.nan,
            "full_macro_f1": np.nan,
            "selective_macro_precision": np.nan,
            "full_macro_precision": np.nan,
            "selective_micro_f1": np.nan,
            "full_micro_f1": np.nan,
            "subset_01_accuracy": np.nan,
            "hamming_loss_full": np.nan,
            "hamming_loss_sel": np.nan,
            "hamming_accuracy_sel": np.nan,
        }

    sub_true = y_true[:, label_indices]
    sub_full = y_full[:, label_indices]
    sub_partial = y_partial[:, label_indices]

    decided = sub_partial != abstain_value
    coverage = float(np.mean(decided))

    hl_full = float(hamming_loss(sub_true, sub_full))
    if np.any(decided):
        hl_sel = float(np.mean(sub_true[decided] != sub_partial[decided]))
    else:
        hl_sel = 0.0
    ha_sel = float(1.0 - hl_sel)

    subset_acc = float(accuracy_score(sub_true, sub_full))
    full_macro_prec = float(precision_score(sub_true, sub_full, average="macro", zero_division=0))
    sel_macro_prec = _compute_selective_macro_precision(sub_true, sub_partial, abstain_value=abstain_value)

    full_macro_f1 = float(f1_score(sub_true, sub_full, average="macro", zero_division=0))
    sel_macro_f1 = float(compute_selective_macro_f1(sub_true, sub_partial, abstain_value=abstain_value))

    full_micro_f1 = float(f1_score(sub_true, sub_full, average="micro", zero_division=0))
    sel_micro_f1 = float(compute_selective_micro_f1(sub_true, sub_partial, abstain_value=abstain_value))

    return {
        "num_labels": int(len(label_indices)),
        "labels": label_indices,
        "coverage": coverage,
        "selective_macro_f1": sel_macro_f1,
        "full_macro_f1": full_macro_f1,
        "selective_macro_precision": sel_macro_prec,
        "full_macro_precision": full_macro_prec,
        "selective_micro_f1": sel_micro_f1,
        "full_micro_f1": full_micro_f1,
        "subset_01_accuracy": subset_acc,
        "hamming_loss_full": hl_full,
        "hamming_loss_sel": hl_sel,
        "hamming_accuracy_sel": ha_sel,
    }


class LayerAblationEvaluator:
    """Evaluates the 6 layer ablation regimes on a multi-label dataset."""

    def __init__(
        self,
        independent_layers: Sequence[Sequence[int]],
        dependent_residual_labels: Sequence[int],
        total_labels: int,
        abstain_value: int = -1,
    ):
        self.independent_layers = [list(layer) for layer in independent_layers]
        self.dependent_residual_labels = list(dependent_residual_labels)
        self.total_labels = int(total_labels)
        self.abstain_value = int(abstain_value)

        # Pre-extract layers
        self.il_1 = self.independent_layers[0] if len(self.independent_layers) > 0 else []
        self.il_2 = self.independent_layers[1] if len(self.independent_layers) > 1 else []
        self.il_3 = self.independent_layers[2] if len(self.independent_layers) > 2 else []
        self.all_il = [l for layer in self.independent_layers for l in layer]
        self.dl = self.dependent_residual_labels

    def get_regime_subsets(self) -> Dict[str, List[int]]:
        """Return the dictionary of label index subsets for each ablation regime."""
        accum_12 = sorted(list(set(self.il_1 + self.il_2)))
        return {
            "ABL_1_ONLY_IL1": sorted(list(set(self.il_1))),
            "ABL_2_ONLY_IL2": sorted(list(set(self.il_2))),
            "ABL_2_ACCUM_IL12": accum_12,
            "ABL_3_ONLY_IL3": sorted(list(set(self.il_3))),
            "ABL_3_ALL_IL": sorted(list(set(self.all_il))),
            "ABL_3_ONLY_DL": sorted(list(set(self.dl))),
            "FULL_SYSTEM": list(range(self.total_labels)),
        }

    def evaluate_all_regimes(
        self,
        y_true: np.ndarray,
        y_full: np.ndarray,
        y_partial: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """Run evaluation across all 6 ablation regimes and the full system."""
        regimes = self.get_regime_subsets()
        results = {}
        for regime_name, subset in regimes.items():
            results[regime_name] = evaluate_label_subset(
                y_true=y_true,
                y_full=y_full,
                y_partial=y_partial,
                label_indices=subset,
                abstain_value=self.abstain_value,
            )
        return results
