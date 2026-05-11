"""Generate the LaTeX numbers for tab:results (and tab:gp_results in appendix).

Reads jsonls from --results-dir/{rf_results,gp_results}/ and prints the table body.
"""
import argparse
import json
import os

import numpy as np

DATASETS_RF = [
    ("Acrolein",         "rf_results/QeMFi_acrolein_rf_results.jsonl"),
    ("Thymine",          "rf_results/QeMFi_thymine_rf_results.jsonl"),
    ("o-HBDI",           "rf_results/QeMFi_o-hbdi_clean_rf_results.jsonl"),
    ("Burgers",          "rf_results/burgers_mf_N5000_rf_results.jsonl"),
    ("Formation Energy", "rf_results/materials_project_formation_energy_rf_results.jsonl"),
]
DATASETS_GP = [
    ("Acrolein",         "gp_results/QeMFi_acrolein_gp_results.jsonl"),
    ("Thymine",          "gp_results/QeMFi_thymine_gp_results.jsonl"),
    ("o-HBDI",           "gp_results/QeMFi_o-hbdi_clean_gp_results.jsonl"),
    ("Burgers",          "gp_results/burgers_mf_N5000_gp_results.jsonl"),
    ("Formation Energy", "gp_results/materials_project_formation_energy_gp_results.jsonl"),
]
METHODS = [
    "HF-Only", "HF-Only (offset)", "HF-Only (augment)",
    "Transfer (offset)", "Transfer (offset) + 1Step",
]
SHORT = ["HF-Only", "Tr-Mean", "Tr-Augment", "MFQR", "MFQR + OS"]


def load_dedupe(path):
    by_key = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_key[(r["seed"], r["method"])] = r
    return list(by_key.values())


def summarise_dataset(jsonl_path):
    rows = load_dedupe(jsonl_path)
    table = {}
    for m in METHODS:
        w = np.array([r["cqr_wid"] for r in rows if r["method"] == m])
        c = np.array([r["cqr_cov"] for r in rows if r["method"] == m])
        table[m] = (float(w.mean()), float(c.mean()))
    return table


def render_table(results_dir, datasets, label, caption):
    """Build a LaTeX wide table: Method | (Width Cov.) per dataset."""
    cols = "l" + "cc" * len(datasets)
    out = []
    out.append("\\begin{table}[H]")
    out.append("\\centering")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\small")
    out.append("\\resizebox{\\textwidth}{!}{%")
    out.append("\\begin{tabular}{" + cols + "}")
    out.append("\\toprule")
    # Multicolumn header row
    parts = [" "] + [f"\\multicolumn{{2}}{{c}}{{{name}}}" for name, _ in datasets]
    out.append(" & ".join(parts) + " \\\\")
    # cmidrules
    rule = "  ".join([f"\\cmidrule(lr){{{2*i + 2}-{2*i + 3}}}" for i in range(len(datasets))])
    out.append(rule)
    # Subheader: Width | Cov.
    sub = ["Method"] + ["Width & Cov."] * len(datasets)
    out.append(" & ".join(sub) + " \\\\")
    out.append("\\midrule")

    # Compute all summaries; bold the per-dataset narrowest width
    summaries = {ds_name: summarise_dataset(os.path.join(results_dir, rel)) for ds_name, rel in datasets}
    best_per_ds = {}
    for ds_name, _ in datasets:
        ws = {m: summaries[ds_name][m][0] for m in METHODS}
        best_per_ds[ds_name] = min(ws, key=ws.get)

    for m, s in zip(METHODS, SHORT):
        cells = [s]
        for ds_name, _ in datasets:
            w, c = summaries[ds_name][m]
            w_str = f"{w:.2f}"
            if best_per_ds[ds_name] == m:
                w_str = f"\\textbf{{{w_str}}}"
            cells.append(f"{w_str} & {c * 100:.1f}")
        out.append(" & ".join(cells) + " \\\\")

    out.append("\\bottomrule")
    out.append("\\end{tabular}%")
    out.append("}")
    out.append("\\end{table}")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing rf_results/ and gp_results/ subfolders.")
    parser.add_argument("--backend", choices=["rf", "gp", "both"], default="both",
                        help="Which table to print (rf, gp, or both).")
    args = parser.parse_args()

    if args.backend in ("rf", "both"):
        print(render_table(
            args.results_dir, DATASETS_RF, "tab:results",
            "Random-forest backend: mean CQR interval width and empirical coverage (\\%) over 20 seeds. "
            "Bold = narrowest width per dataset.",
        ))
        print()
    if args.backend in ("gp", "both"):
        print(render_table(
            args.results_dir, DATASETS_GP, "tab:gp_results",
            "Gaussian-process backend: mean CQR interval width and empirical coverage (\\%) over 20 seeds.",
        ))


if __name__ == "__main__":
    main()
