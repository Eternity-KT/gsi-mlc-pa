import sys
from pathlib import Path
import pandas as pd
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DECAY_DIR = WORKSPACE_ROOT / "results_v6_2" / "decay_study"

def generate_report():
    df_summary = pd.read_csv(DECAY_DIR / "dl_imbalance_summary_v6_2_1.csv")
    df_labels = pd.read_csv(DECAY_DIR / "dl_per_label_imbalance_v6_2_1.csv")
    df_br_f1 = pd.read_csv(DECAY_DIR / "br_f1_vs_imbalance_v6_2_1.csv")

    # Filter for Logistic learner to make clear baseline table
    df_log = df_summary[df_summary["learner"] == "Logistic"].copy()

    # Markdown content
    lines = []
    lines.append("# BÁO CÁO KHẢO SÁT: ĐỘ MẤT CÂN BẰNG CỦA CÁC NHÃN TRONG TẬP DL (DEPENDENT LABELS) Ở PHIÊN BẢN v6.2.1\n")
    lines.append("**Tài liệu tham chiếu:** thiết kế lịch sử v6.2\n")
    lines.append("**Phiên bản khảo sát:** GSI-MLC-PA v6.2.1 (`GSI_v6_2_Decay` - Hạ ngưỡng phân tầng IL: Tầng 1 $\\tau=0.75$, Tầng 2 $\\tau=0.70$, Tầng 3 $\\tau=0.65$)\n")
    lines.append("**Bộ phân loại cơ sở:** Logistic Regression, Calibrated SVM, Multi-Layer Perceptron (MLP)\n")
    lines.append("\n---\n")

    lines.append("## 1. Tổng Quan Về Tập DL & Tỷ Lệ Nhãn Mất Cân Bằng (Toàn Bộ 10 Tập Dữ Liệu)\n")
    lines.append("Bảng dưới đây thống kê số lượng nhãn bị kẹt lại trong tập phụ thuộc $DL$ so với tập độc lập $IL$, cùng các chỉ số mất cân bằng (Imbalance Ratio - IR, tần suất nhãn hiếm < 5%):\n\n")

    lines.append("| Tập Dữ Liệu | Tổng Nhãn $K$ | Nhãn $IL$ | Nhãn $DL$ | Tỷ lệ $DL$ (%) | Mean IR ($DL$) | Median IR ($DL$) | Tần suất TB ($DL$) | Số nhãn hiếm (<5%) trong $DL$ | Mean IR ($IL$) |\n")
    lines.append("|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")

    for _, r in df_log.iterrows():
        ds = r["dataset"]
        K = int(r["K"])
        n_il = int(r["n_il"])
        n_dl = int(r["n_dl"])
        pct_dl = f"{r['pct_dl']:.1f}%"
        dl_mean_ir = f"{r['dl_mean_ir']:.2f}" if not pd.isna(r["dl_mean_ir"]) else "N/A"
        dl_med_ir = f"{r['dl_median_ir']:.2f}" if not pd.isna(r["dl_median_ir"]) else "N/A"
        dl_freq = f"{r['dl_mean_freq_pct']:.2f}%" if not pd.isna(r["dl_mean_freq_pct"]) else "N/A"
        dl_rare = int(r["dl_rare_lt_5pct"])
        il_mean_ir = f"{r['il_mean_ir']:.2f}" if not pd.isna(r["il_mean_ir"]) else "0 nhãn IL"

        lines.append(f"| **{ds}** | {K} | {n_il} | {n_dl} | {pct_dl} | {dl_mean_ir} | {dl_med_ir} | {dl_freq} | {dl_rare}/{n_dl} | {il_mean_ir} |\n")

    lines.append("\n---\n")

    lines.append("## 2. Phân Tích Chi Tiết 3 Mô Hình Cơ Sở (Logistic, SVM, MLP)\n")
    lines.append("Sự khác biệt về phân tách $IL$ và $DL$ giữa các bộ phân loại cơ sở:\n\n")

    lines.append("| Base Learner | Dataset | $n_{IL}$ | $n_{DL}$ | % DL | Mean IR ($DL$) | Mean IR ($IL$) | Số nhãn < 5% trong DL |\n")
    lines.append("|:---|:---|---:|---:|---:|---:|---:|---:|\n")

    for _, r in df_summary.iterrows():
        bl = r["learner"]
        ds = r["dataset"]
        n_il = int(r["n_il"])
        n_dl = int(r["n_dl"])
        pct_dl = f"{r['pct_dl']:.1f}%"
        dl_mean_ir = f"{r['dl_mean_ir']:.2f}" if not pd.isna(r["dl_mean_ir"]) else "N/A"
        il_mean_ir = f"{r['il_mean_ir']:.2f}" if not pd.isna(r["il_mean_ir"]) else "N/A"
        rare = int(r["dl_rare_lt_5pct"])
        lines.append(f"| {bl} | {ds} | {n_il} | {n_dl} | {pct_dl} | {dl_mean_ir} | {il_mean_ir} | {rare} |\n")

    lines.append("\n---\n")

    lines.append("## 3. Khảo Sát Chi Tiết Từng Nhãn Trong Tập $DL$ và Hiệu Năng BR F1\n")
    lines.append("### 3.1. Nhóm Bị Kẹt 100% Trong DL: `humanpseaac` & `plantpseaac`\n")
    lines.append("Hai tập dữ liệu này có đặc điểm chung: **Không có bất kỳ nhãn nào lọt được vào $IL$ ($n_{IL} = 0, n_{DL} = K$)** trên cả 3 mô hình cơ sở.\n\n")

    for target_ds in ["humanpseaac", "plantpseaac"]:
        sub_df = df_br_f1[df_br_f1["dataset"] == target_ds].sort_values(by="pos_count", ascending=False)
        lines.append(f"#### Chi tiết tập nhãn `{target_ds}` ($N = {len(sub_df)}$ nhãn):\n")
        lines.append("| Index | Tên Nhãn | Trạng thái v6.2.1 | Số mẫu Positive | Tần suất (%) | Imbalance Ratio (IR) | BR F1 Score |\n")
        lines.append("|---:|:---|:---:|---:|---:|---:|---:|\n")
        for _, r in sub_df.iterrows():
            idx = int(r["label_idx"])
            name = r["label_name"]
            st = r["status"]
            pos = int(r["pos_count"])
            pct = f"{r['pos_freq_pct']:.2f}%"
            ir = f"{r['ir']:.2f}"
            f1 = f"{r['br_f1']:.4f}"
            lines.append(f"| {idx} | `{name}` | **{st}** | {pos} | {pct} | {ir} | {f1} |\n")
        lines.append("\n")

    lines.append("### 3.2. Nhóm Mất Cân Bằng Cực Đoan: `genbase`\n")
    lines.append("`genbase` có 27 nhãn, trong đó các nhãn phân bố thành 2 cực rõ rệt:\n\n")
    sub_gb = df_br_f1[df_br_f1["dataset"] == "genbase"].sort_values(by="pos_count", ascending=False)
    lines.append("| Index | Tên Nhãn | Trạng thái v6.2.1 | Số mẫu Positive | Tần suất (%) | Imbalance Ratio (IR) | BR F1 Score |\n")
    lines.append("|---:|:---|:---:|---:|---:|---:|---:|\n")
    for _, r in sub_gb.iterrows():
        idx = int(r["label_idx"])
        name = r["label_name"]
        st = r["status"]
        pos = int(r["pos_count"])
        pct = f"{r['pos_freq_pct']:.2f}%"
        ir = f"{r['ir']:.2f}"
        f1 = f"{r['br_f1']:.4f}"
        lines.append(f"| {idx} | `{name}` | **{st}** | {pos} | {pct} | {ir} | {f1} |\n")
    lines.append("\n")

    lines.append("### 3.3. Các Tập Dữ Liệu Còn Lại: `yeast`, `chd49`, `gpositivepseaac`, `viruspseaac`, `scene`, `emotions`, `music`\n\n")
    for target_ds in ["yeast", "chd49", "gpositivepseaac", "viruspseaac", "scene", "emotions", "music"]:
        sub_df = df_br_f1[df_br_f1["dataset"] == target_ds].sort_values(by="pos_count", ascending=False)
        lines.append(f"#### Tập `{target_ds}`:\n")
        lines.append("| Index | Tên Nhãn | Trạng thái v6.2.1 | Số mẫu Positive | Tần suất (%) | Imbalance Ratio (IR) | BR F1 Score |\n")
        lines.append("|---:|:---|:---:|---:|---:|---:|---:|\n")
        for _, r in sub_df.iterrows():
            idx = int(r["label_idx"])
            name = r["label_name"]
            st = r["status"]
            pos = int(r["pos_count"])
            pct = f"{r['pos_freq_pct']:.2f}%"
            ir = f"{r['ir']:.2f}"
            f1 = f"{r['br_f1']:.4f}"
            lines.append(f"| {idx} | `{name}` | **{st}** | {pos} | {pct} | {ir} | {f1} |\n")
        lines.append("\n")

    report_text = "".join(lines)
    out_file = DECAY_DIR / "khao_sat_mat_can_bang_nhan_dl_v6_2_1.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Report written to {out_file}")

if __name__ == "__main__":
    generate_report()
