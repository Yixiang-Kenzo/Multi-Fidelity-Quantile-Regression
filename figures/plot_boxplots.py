"""Section 5.4 boxplots: coverage (left) + CQR interval width (right) per dataset.

Consolidates three previous boxplot scripts (plot_boxplots_individual.py,
plot_extra_boxplots.py, plot_burgers_ncal300.py) into one config-driven script.

Reads canonical jsonl results from --results-dir (organised into rf_results/
and gp_results/ subdirectories) and writes one PNG per dataset to --output-dir.

The plot style matches the paper: vertical boxes, 5 methods on x-axis,
coverage on left (50%-100% range, with red dashed line at 90%), width on right
with mean values labelled in coloured boxes.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Method names, display labels, and box colours all consistent with the paper.
METHODS = [
    "HF-Only", "HF-Only (offset)", "HF-Only (augment)",
    "Transfer (offset)", "Transfer (offset) + 1Step",
]
SHORT = ["HF-Only", "Tr-Mean", "Tr-Augment", "MFQR", "MFQR + OS"]
BOX_COLORS = ["#e0a0b0", "#d4c48c", "#d4c48c", "#9cb0e0", "#9cb0e0"]

# Configuration table: (display name, jsonl path relative to --results-dir,
# output filename).  rf_results/ and gp_results/ are the two backend subdirs.
PLOTS = [
    # ---- §5.4 main-paper RF boxplots ----
    ("Acrolein",         "rf_results/QeMFi_acrolein_rf_results.jsonl",                  "boxplot_Acrolein.png"),
    ("Thymine",          "rf_results/QeMFi_thymine_rf_results.jsonl",                   "boxplot_Thymine.png"),
    ("o-HBDI RF",        "rf_results/QeMFi_o-hbdi_clean_rf_results.jsonl",              "boxplot_o-HBDI_RF.png"),
    ("Burgers",          "rf_results/burgers_mf_N5000_rf_results.jsonl",                "boxplot_Burgers.png"),
    ("Formation Energy", "rf_results/materials_project_formation_energy_rf_results.jsonl", "boxplot_Formation_Energy.png"),
    # ---- Appendix GP boxplots ----
    ("Acrolein GP",         "gp_results/QeMFi_acrolein_gp_results.jsonl",                  "boxplot_Acrolein_GP.png"),
    ("Thymine GP",          "gp_results/QeMFi_thymine_gp_results.jsonl",                   "boxplot_Thymine_GP.png"),
    ("o-HBDI GP",           "gp_results/QeMFi_o-hbdi_clean_gp_results.jsonl",              "boxplot_o-HBDI_GP.png"),
    ("Burgers GP",          "gp_results/burgers_mf_N5000_gp_results.jsonl",                "boxplot_Burgers_GP.png"),
    ("Formation Energy GP", "gp_results/materials_project_formation_energy_gp_results.jsonl", "boxplot_Formation_Energy_GP.png"),
]


def load_dedupe(path):
    """Load a jsonl, de-duplicating by (seed, method) — the last entry wins."""
    by_key = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_key[(r["seed"], r["method"])] = r
    return list(by_key.values())


def plot_one(name, jsonl_path, out_path):
    rows = load_dedupe(jsonl_path)

    fig, (ax_c, ax_w) = plt.subplots(1, 2, figsize=(9, 4.2))
    box_kw = dict(
        vert=True, patch_artist=True, widths=0.55,
        boxprops=dict(linewidth=0.8),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        medianprops=dict(color="#333333", linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="grey",
                        markeredgecolor="grey", alpha=0.6),
    )

    wid_data, cov_data, means_w = [], [], []
    for m in METHODS:
        w = [r["cqr_wid"] for r in rows if r["method"] == m]
        c = [r["cqr_cov"] for r in rows if r["method"] == m]
        wid_data.append(w)
        cov_data.append(c)
        means_w.append(np.mean(w) if w else 0)

    # Width panel (right)
    bp_w = ax_w.boxplot(wid_data, **box_kw)
    for patch, col in zip(bp_w["boxes"], BOX_COLORS):
        patch.set_facecolor(col); patch.set_alpha(0.8); patch.set_edgecolor("#888888")
    ax_w.set_xticks(range(1, len(SHORT) + 1))
    ax_w.set_xticklabels(SHORT, fontweight="bold", fontsize=9, rotation=20, ha="right")
    ax_w.set_title("Avg. Width", fontsize=10, fontweight="bold")
    auto_bot, auto_top = ax_w.get_ylim()
    data_range = auto_top - auto_bot
    ax_w.set_ylim(auto_bot, auto_top + 0.12 * data_range)
    for i, avg in enumerate(means_w):
        ax_w.text(i + 1, auto_top + 0.05 * data_range, f"{avg:.2f}",
                  ha="center", va="center", fontweight="bold", fontsize=8,
                  bbox=dict(boxstyle="square,pad=0.2", facecolor=BOX_COLORS[i],
                            edgecolor="#aaaaaa", alpha=0.9), zorder=5)

    # Coverage panel (left)
    bp_c = ax_c.boxplot(cov_data, **box_kw)
    for patch, col in zip(bp_c["boxes"], BOX_COLORS):
        patch.set_facecolor(col); patch.set_alpha(0.8); patch.set_edgecolor("#888888")
    ax_c.set_xticks(range(1, len(SHORT) + 1))
    ax_c.set_xticklabels(SHORT, fontweight="bold", fontsize=9, rotation=20, ha="right")
    ax_c.axhline(0.90, color="red", ls="--", lw=1, alpha=0.5)
    ax_c.set_ylim(0.50, 1.00)
    ax_c.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax_c.set_title("Avg. Coverage", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}  ({name}, {len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing rf_results/ and gp_results/ subfolders.")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for boxplot PNGs.  Default: alongside script.")
    parser.add_argument("--only", default=None,
                        help="Comma-separated PLOTS names to render (default: all 10).")
    args = parser.parse_args()
    outdir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(outdir, exist_ok=True)
    only = set(s.strip() for s in args.only.split(",")) if args.only else None

    for name, rel_jsonl, out_name in PLOTS:
        if only and name not in only:
            continue
        jsonl_path = os.path.join(args.results_dir, rel_jsonl)
        if not os.path.exists(jsonl_path):
            print(f"  SKIP   {name}: {jsonl_path} not found")
            continue
        out_path = os.path.join(outdir, out_name)
        plot_one(name, jsonl_path, out_path)


if __name__ == "__main__":
    main()
