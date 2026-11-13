"""BR Residual Error Correlation and Conditional Dependency Graph Module.

Reference:
    - meeting_summary.md (Core Section, v6.2)
    - spec/spec_v6_3_3.md

This module computes the prediction error residuals of base Binary Relevance (BR)
classifiers on candidate dependent labels (DL), calculates their Pearson Correlation
Coefficients (PCC), and extracts the local conditional dependency graph DL_temp[l]
for each label l in DL.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd


def compute_br_residual_matrix(
    Y_true: np.ndarray,
    P_prob: np.ndarray,
    error_format: str = "residual",
) -> np.ndarray:
    """Compute prediction error residual matrix for candidate labels.

    Parameters:
        Y_true: Ground-truth binary label matrix of shape (n_samples, n_labels),
                values in {0, 1}.
        P_prob: Out-of-fold predicted probability matrix of shape (n_samples, n_labels),
                values in [0.0, 1.0].
        error_format: Format of residual:
            - 'residual' (default): Signed difference (Y - P) in [-1.0, 1.0].
            - 'abs_residual': Absolute continuous error |Y - P| in [0.0, 1.0].
            - 'binary': Discrete 0/1 error |Y - (P >= 0.5)|.

    Returns:
        Residual matrix of shape (n_samples, n_labels) with dtype float64.
    """
    Y_true = np.asarray(Y_true, dtype=np.float64)
    P_prob = np.asarray(P_prob, dtype=np.float64)

    if Y_true.shape != P_prob.shape:
        raise ValueError(
            f"Shape mismatch: Y_true shape {Y_true.shape} vs P_prob shape {P_prob.shape}"
        )

    if error_format == "residual":
        return Y_true - P_prob
    elif error_format == "abs_residual":
        return np.abs(Y_true - P_prob)
    elif error_format == "binary":
        y_hat = (P_prob >= 0.5).astype(np.float64)
        return np.abs(Y_true - y_hat)
    else:
        raise ValueError(
            f"Unknown error_format '{error_format}'. Must be 'residual', 'abs_residual', or 'binary'."
        )


def compute_residual_pcc_matrix(
    residual_matrix: np.ndarray,
    use_absolute: bool = True,
) -> np.ndarray:
    """Compute pairwise Pearson Correlation Coefficient (PCC) matrix across error residuals.

    Corr(l, p) = PCC((l - f(l)), (p - f(p)))

    Parameters:
        residual_matrix: Matrix of shape (n_samples, n_labels) containing prediction errors.
        use_absolute: If True (default), returns absolute PCC in [0.0, 1.0], since strong negative
                      correlation also represents a critical conditional dependency.

    Returns:
        Symmetric matrix of shape (n_labels, n_labels) with diagonal entries set to 1.0.
    """
    residuals = np.asarray(residual_matrix, dtype=np.float64)
    if residuals.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {residuals.shape}")

    n_samples, n_labels = residuals.shape
    if n_labels == 0:
        return np.empty((0, 0), dtype=np.float64)
    if n_labels == 1:
        return np.ones((1, 1), dtype=np.float64)

    stds = np.std(residuals, axis=0)
    valid_mask = stds > 1e-12

    corr_matrix = np.zeros((n_labels, n_labels), dtype=np.float64)
    np.fill_diagonal(corr_matrix, 1.0)

    # If fewer than 2 columns have non-zero variance, return identity
    if np.sum(valid_mask) < 2:
        return corr_matrix

    with np.errstate(all="ignore"):
        raw_corr = np.corrcoef(residuals, rowvar=False)

    raw_corr = np.nan_to_num(raw_corr, nan=0.0, posinf=0.0, neginf=0.0)
    raw_corr = np.clip(raw_corr, -1.0, 1.0)

    if use_absolute:
        result = np.abs(raw_corr)
    else:
        result = raw_corr

    np.fill_diagonal(result, 1.0)
    return result


def build_residual_dependency_graph(
    corr_matrix: np.ndarray,
    dl_labels: Sequence[int],
    threshold: float = 0.25,
) -> Dict[int, List[int]]:
    """Build conditional dependency mapping DL_temp[l] based on PCC threshold.

    For each label l in DL:
        DL_temp[l] = { p in DL \\ {l} | Corr(l, p) >= threshold }
    ordered by descending correlation strength.

    Parameters:
        corr_matrix: Symmetric correlation matrix of shape (len(dl_labels), len(dl_labels))
                     or full (K, K) indexed by dl_labels.
        dl_labels: List of label indices in DL.
        threshold: Correlation threshold tau_corr (default 0.25).

    Returns:
        Dictionary mapping label index l -> sorted list of coupled labels DL_temp[l].
    """
    dl_list = [int(lbl) for lbl in dl_labels]
    n_dl = len(dl_list)
    graph: Dict[int, List[int]] = {lbl: [] for lbl in dl_list}

    if n_dl <= 1:
        return graph

    is_submatrix = (corr_matrix.shape == (n_dl, n_dl))

    for idx_l, l in enumerate(dl_list):
        coupled: List[Tuple[int, float]] = []
        for idx_p, p in enumerate(dl_list):
            if p == l:
                continue
            corr_val = (
                float(corr_matrix[idx_l, idx_p])
                if is_submatrix
                else float(corr_matrix[l, p])
            )
            if corr_val >= float(threshold):
                coupled.append((p, corr_val))

        # Sort descending by correlation strength, break ties by label index
        coupled.sort(key=lambda item: (-item[1], item[0]))
        graph[l] = [item[0] for item in coupled]

    return graph


def export_residual_correlation_audit_table(
    corr_matrix: np.ndarray,
    dl_labels: Sequence[int],
    dependency_graph: Dict[int, List[int]],
    error_rates: Optional[Dict[int, float]] = None,
    label_names: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """Create an audit report DataFrame detailing residual error correlations.

    Parameters:
        corr_matrix: Submatrix of shape (len(dl_labels), len(dl_labels)) or full (K, K).
        dl_labels: List of label indices in DL.
        dependency_graph: Mapping l -> DL_temp[l].
        error_rates: Optional mapping l -> mean absolute error rate of base BR model.
        label_names: Optional mapping l -> string name of label.

    Returns:
        pd.DataFrame containing audit rows for each label in DL.
    """
    dl_list = [int(lbl) for lbl in dl_labels]
    n_dl = len(dl_list)
    is_submatrix = (corr_matrix.shape == (n_dl, n_dl))

    rows = []
    for idx_l, l in enumerate(dl_list):
        lname = label_names.get(l, f"Label_{l}") if label_names else f"Label_{l}"
        err = error_rates.get(l, 0.0) if error_rates else 0.0

        # Find max correlated partner
        max_partner = None
        max_corr = 0.0
        for idx_p, p in enumerate(dl_list):
            if p == l:
                continue
            c = (
                float(corr_matrix[idx_l, idx_p])
                if is_submatrix
                else float(corr_matrix[l, p])
            )
            if c > max_corr:
                max_corr = c
                max_partner = p

        partner_name = (
            label_names.get(max_partner, f"Label_{max_partner}")
            if (max_partner is not None and label_names)
            else (f"Label_{max_partner}" if max_partner is not None else "None")
        )

        coupled_labels = dependency_graph.get(l, [])
        coupled_str = (
            ", ".join(str(p) for p in coupled_labels)
            if coupled_labels
            else "empty"
        )

        rows.append({
            "label_idx": l,
            "label_name": lname,
            "error_rate": float(err),
            "top_correlated_partner": partner_name,
            "max_correlation": float(max_corr),
            "coupled_dl_temp": coupled_str,
            "num_coupled": len(coupled_labels),
        })

    return pd.DataFrame(rows)


def one_step_normalized_mean_field(
    P_base: np.ndarray,
    corr_matrix: np.ndarray,
    alpha: float = 0.25,
    threshold: float = 0.25,
    z_max: float = 0.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """Refine base predicted probabilities via one-step normalized mean-field belief propagation.

    Parameters:
        P_base: Base predicted probabilities of shape (n_samples, n_labels), values in [0.0, 1.0].
        corr_matrix: Pairwise Pearson correlation coefficient matrix of shape (n_labels, n_labels).
        alpha: Maximum coupling scaling factor (default: 0.25).
        threshold: Correlation magnitude threshold for sparse coupling (default: 0.25).
        z_max: Maximum logit adjustment shift (default: 0.5).
        eps: Small numerical epsilon (default: 1e-6).

    Returns:
        Refined probability matrix of shape (n_samples, n_labels).
    """
    P_base = np.asarray(P_base, dtype=np.float64)
    if P_base.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {P_base.shape}")
    n_samples, n_labels = P_base.shape
    if n_labels <= 1 or corr_matrix is None or corr_matrix.shape != (n_labels, n_labels):
        return P_base

    W = np.copy(corr_matrix)
    W[np.abs(W) < threshold] = 0.0
    np.fill_diagonal(W, 0.0)

    # Degree normalization with maximum denominator guard
    row_sums = np.sum(np.abs(W), axis=1, keepdims=True)
    norm_factor = np.maximum(1.0, row_sums)
    W_norm = alpha * (W / norm_factor)

    # Symmetric spin-centering in [-1.0, +1.0]
    delta = 2.0 * P_base - 1.0

    # 1-step interaction calculation
    interaction = np.dot(delta, W_norm.T)
    interaction = np.clip(interaction, -z_max, z_max)

    # Apply shift to base logits
    P_clipped = np.clip(P_base, eps, 1.0 - eps)
    theta = np.log(P_clipped / (1.0 - P_clipped))

    return 1.0 / (1.0 + np.exp(-(theta + interaction)))

