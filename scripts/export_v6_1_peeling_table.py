"""
Script trích xuất và hiển thị Bảng Chi Tiết Bóc Tách Nhãn IL Qua Từng Tầng (v6.1 Peeling Breakdown).

Tính năng:
1. Đánh giá 5-Fold CV Peeling tuần tự cho 10 tập benchmark chuẩn.
2. Hiển thị chi tiết:
   - Tầng 1: Số nhãn IL_1 và danh sách chỉ số nhãn
   - Tầng 2: Số nhãn IL_2 và danh sách chỉ số nhãn (dựa trên đặc trưng tăng cường OOF)
   - Tầng 3: Số nhãn IL_3 (nếu có)
   - Nhãn Singleton DL được thăng hạng vào IL (quy tắc: len(DL) == 1 => IL)
   - Tổng số nhãn IL (K_IL) và Tỷ lệ % IL
   - Số nhãn DL còn lại (K_DL) và danh sách thứ tự chuỗi CC
   - Hiệu năng của Binary Relevance (BR) trên tập IL (Selective-F1, Coverage, Precision)
   - Điểm số Selective-F1 OOF chi tiết của từng nhãn qua từng tầng
3. Xuất ra file Markdown và CSV tại `results_v6_1/`.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.loader import load_dataset
from src.models.base_learners import create_binary_estimator
from src.selection.cv_peeling import CVPeelingConfig, CVStratifiedPeelingSelector

ALL_10_DATASETS = [
    "emotions",
    "scene",
    "chd49",
    "music",
    "gpositivepseaac",
    "genbase",
    "humanpseaac",
    "plantpseaac",
    "viruspseaac",
    "yeast",
]

DEFAULT_LEARNERS = ["logistic", "svm_calibrated", "mlp"]


def extract_peeling_breakdown(
    datasets: List[str] = ALL_10_DATASETS,
    base_learners: List[str] = DEFAULT_LEARNERS,
    threshold: float = 0.75,
    promote_singleton: bool = True,
    n_folds: int = 5,
    cost: float = 0.30,
    output_dir: Path = WORKSPACE_ROOT / "results_v6_1",
) -> pd.DataFrame:
    """Run CV peeling and generate detailed multi-layer breakdown tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 90)
    print("XUẤT BẢNG CHI TIẾT BÓC TÁCH NHÃN IL QUA TỪNG TẦNG (GSI-MLC-PA v6.1)")
    print(f"Ngưỡng tau: {threshold} | Chi phí c: {cost} | Số Folds CV: {n_folds} | Singleton Rule: {promote_singleton}")
    print("=" * 90)

    for learner in base_learners:
        display_learner = {
            "logistic": "Logistic",
            "svm_calibrated": "SVM",
            "svm": "SVM",
            "mlp": "MLP",
        }.get(learner, learner.upper())

        print(f"\n--- Đang xử lý bộ học cơ sở: {display_learner} ({learner}) ---")

        for ds_name in datasets:
            X, Y, _, _ = load_dataset(ds_name, base_dir=str(WORKSPACE_ROOT))
            if hasattr(X, "toarray"):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float32)
            Y = np.asarray(Y, dtype=np.int32)
            n_samples, n_labels = Y.shape

            scaler = MaxAbsScaler()
            X = scaler.fit_transform(X)

            t0 = time.perf_counter()
            config = CVPeelingConfig(
                threshold=threshold,
                n_folds=n_folds,
                max_depth=3,
                cost=cost,
                metric="selective_f1",
                promote_singleton_dl=promote_singleton,
                random_state=42,
            )
            selector = CVStratifiedPeelingSelector(
                config=config,
                base_estimator_factory=lambda l=learner: create_binary_estimator(l, random_state=42),
            )
            res = selector.fit_partition(X, Y)
            elapsed = time.perf_counter() - t0

            il_1 = list(res.labels_il_stage_1)
            il_2 = list(res.labels_il_stage_2)
            il_3 = list(res.labels_il_stage_3)
            all_il = list(res.all_independent_labels)
            dl_res = list(res.dependent_residual_labels)
            br_metrics = res.br_independent_metrics

            row = {
                "Tập dữ liệu": ds_name,
                "Bộ học": display_learner,
                "N": n_samples,
                "K": n_labels,
                "K_IL_1": len(il_1),
                "Nhãn IL_1": str(il_1),
                "K_IL_2": len(il_2),
                "Nhãn IL_2": str(il_2),
                "K_IL_3": len(il_3),
                "Nhãn IL_3": str(il_3),
                "Tổng K_IL": len(all_il),
                "Tỷ lệ IL (%)": f"{res.pct_total_il:.1f}%",
                "K_DL": len(dl_res),
                "Nhãn Residual DL": str(dl_res),
                "BR Selective-F1": f"{br_metrics.get('mean_selective_f1', 0.0):.4f}" if all_il else "—",
                "BR Coverage": f"{br_metrics.get('mean_coverage', 0.0)*100:.1f}%" if all_il else "—",
                "Thời gian (s)": f"{elapsed:.2f}s",
                "Lý do dừng": res.stopping_reason,
            }
            summary_rows.append(row)
            print(f"  [{ds_name:15s}] K={n_labels:2d} -> IL_1={len(il_1)} {il_1}, IL_2={len(il_2)} {il_2}, DL={len(dl_res)} {dl_res} | Thời gian: {elapsed:.2f}s")

    df = pd.DataFrame(summary_rows)

    # Export CSV
    csv_path = output_dir / "table_il_peeling_breakdown.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Export Markdown
    md_path = output_dir / "table_il_peeling_breakdown.md"
    md_lines = [
        "# BẢNG CHI TIẾT PHÂN TÁCH NHÃN ĐỘC LẬP (IL) VÀ PHỤ THUỘC (DL) QUA TỪNG TẦNG (GSI-MLC-PA v6.1)\n",
        f"**Thời gian trích xuất:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        "**Tài liệu tham chiếu:** thiết kế lịch sử v6.1  ",
        "**Cấu hình:** 5-Fold Cross-Validation Out-Of-Fold, Ngưỡng thăng hạng $\\tau = 0.75$, Chi phí từ chối $c = 0.30$\n",
        "---\n",
        "## 1. Bảng Phân Tách Chi Tiết $IL_1, IL_2, IL_3$ và $DL_{\\text{residual}}$ Trên 10 Tập Dữ Liệu\n",
        df.to_markdown(index=False),
        "\n---\n",
        "## 2. Ghi Chú Kỹ Thuật:",
        "- **Tầng 1 ($IL_1$):** Các nhãn đạt $\\text{Selective-F1} \\ge 0.75$ khi chỉ dùng không gian thuộc tính gốc $X$.",
        "- **Tầng 2 ($IL_2$):** Các nhãn ban đầu không đạt, nhưng sau khi được bổ sung thuộc tính xác suất Out-Of-Fold của $IL_1$ ($[X, \\hat{P}^{\\text{OOF}}_{IL_1}]$) đã vượt qua ngưỡng $0.75$.",
        "- **Quy tắc Singleton DL:** Nếu sau các tầng bóc tách, tập phụ thuộc chỉ còn 1 nhãn (`len(DL) == 1`), nhãn này được tự động chuyển vào $IL$ để mô hình BR dự đoán độc lập, tránh sự thoái hóa của chuỗi CC độ dài 1.",
        "- **Tập $DL_{\\text{residual}}$:** Các nhãn còn lại có độ phụ thuộc phức tạp, sẽ được đưa vào mô hình Ensemble Classifier Chains (ECC) ở v6.1.",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[Thành công] Đã lưu bảng phân tách tại:")
    print(f"  - CSV: {csv_path}")
    print(f"  - Markdown: {md_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Export IL Peeling Breakdown Table.")
    parser.add_argument("--datasets", nargs="+", default=ALL_10_DATASETS)
    parser.add_argument("--base-learners", nargs="+", default=["logistic"])
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--cost", type=float, default=0.30)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--no-singleton-rule", action="store_true")
    args = parser.parse_args()

    extract_peeling_breakdown(
        datasets=args.datasets,
        base_learners=args.base_learners,
        threshold=args.threshold,
        promote_singleton=not args.no_singleton_rule,
        n_folds=args.cv_folds,
        cost=args.cost,
    )


if __name__ == "__main__":
    main()
