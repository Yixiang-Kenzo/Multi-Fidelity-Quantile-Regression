"""
Run synthetic multi-fidelity experiments for the paper.

Two experiments:
  1. P1 (distributional mismatch): parabolic mean, HF=Normal, LF=t(10), no affine link
  2. Break2 (heteroscedastic mismatch): sinusoidal mean, HF=Normal(const), LF=t(10)(hetero)

Usage:
  python run_synthetic.py --experiment P1
  python run_synthetic.py --experiment Break2
  python run_synthetic.py --experiment all
  python run_synthetic.py --experiment all --plot-only   # skip fitting, just plot from saved results
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Allow running this script directly from final_scripts/runners/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mfqr import build_methods, standardise_separately, conformal_calibrate
from mfqr import ConditionalCDF_QRF

# ── Constants ─────────────────────────────────────────────────────
ALPHA = 0.10
DF = 10
S_T = np.sqrt((DF - 2) / DF)  # variance correction for t(df)

METHODS_5 = [
    "HF-Only", "HF-Only (augment)", "HF-Only (offset)",
    "Transfer (offset)", "Transfer (offset) + 1Step",
]
METHODS_6 = [
    "HF-Only", "HF-Only (augment)", "HF-Only (offset)",
    "Transfer", "Transfer (offset)", "Transfer (offset) + 1Step",
]
METHODS_4 = [
    "HF-Only", "Transfer",
    "Transfer (offset)", "Transfer (offset) + 1Step",
]
SHORT = {
    "HF-Only": "HF-Only", "HF-Only (augment)": "Transfer-Augment",
    "HF-Only (offset)": "Transfer-Mean", "Transfer": "Transfer",
    "Transfer (offset)": "MFQR", "Transfer (offset) + 1Step": "MFQR + OS",
}
COLORS = {
    "HF-Only": "#d62728", "HF-Only (augment)": "#e377c2",
    "HF-Only (offset)": "#9467bd", "Transfer": "#ff7f0e",
    "Transfer (offset)": "#2ca02c", "Transfer (offset) + 1Step": "#1f77b4",
}
SMOOTH_SIZE = 100


# ── Experiment definitions ────────────────────────────────────────
EXPERIMENTS = {
    "P1": {
        "tag": "Informative",
        "seed": 300,
        "xl": 1.5, "xr": 4.5,
        "N": 5000, "n_lf": 1250, "n_hf": 125, "n_cal": 250,
        "rf_mean_min_leaf": 30, "rf_crossfit_min_leaf": 30,
        "methods": METHODS_5,
        "f_fn": lambda X: 0.5 * X**2 - 2 * X + 1,
        "hf_gen": lambda rng, X, f: f + (0.1 + 0.35 * np.sin(3 * np.pi * X)**2) * rng.normal(0, 1, len(X)),
        "lf_gen": lambda rng, X, f: f + 1.7 * (0.1 + 0.35 * np.sin(3 * np.pi * X)**2) * rng.standard_t(DF, len(X)) * S_T,
    },
    # Paper mapping: this experiment corresponds to Section 5.2.2
    # ("Non-informative LF regime") of the published paper.  The DGP --
    # shared sin-based mean, HF homoscedastic noise, LF heteroscedastic
    # noise with parabolic variance -- is unchanged.  Output filenames use
    # the prefix "Noninformative_" to match the published paper.
    "Break2": {
        "tag": "Noninformative",
        "seed": 300,
        "xl": 0, "xr": 0.5,
        "N": 2500, "n_lf": 1250, "n_hf": 125, "n_cal": 250,
        "rf_mean_min_leaf": 30, "rf_crossfit_min_leaf": 30,
        "methods": METHODS_5,
        "f_fn": lambda X: 0.6 * (2 * np.sin(2 * np.pi * X) + X),
        "hf_gen": lambda rng, X, f: f + 0.1 * rng.normal(0, 1, len(X)),
        "lf_gen": lambda rng, X, f: f + (0.1 + 6.4 * (X - 0.25)**2) * rng.standard_t(DF, len(X)) * S_T,
    },
}


# ── Data generation (deterministic given seed) ────────────────────
def generate_data(cfg):
    seed = cfg["seed"]
    rng_base = np.random.RandomState(seed)
    X = rng_base.uniform(cfg["xl"], cfg["xr"], cfg["N"])
    f_mean = cfg["f_fn"](X)
    Y_hf = cfg["hf_gen"](rng_base, X, f_mean)
    rng_lf = np.random.RandomState(seed + 99)
    Y_lf = cfg["lf_gen"](rng_lf, X, f_mean)

    rng_split = np.random.RandomState(seed + 1)
    idx = rng_split.permutation(cfg["N"])
    n_lf, n_hf, n_cal = cfg["n_lf"], cfg["n_hf"], cfg["n_cal"]
    X2 = X.reshape(-1, 1)
    data = {
        "X_lf": X2[idx[:n_lf]], "Y_lf_train": Y_lf[idx[:n_lf]],
        "X_hf": X2[idx[n_lf:n_lf + n_hf]], "Y_hf_train": Y_hf[idx[n_lf:n_lf + n_hf]],
        "Y_lf_at_hf": Y_lf[idx[n_lf:n_lf + n_hf]],
        "X_test": X2[idx[n_lf + n_hf + n_cal:]], "Y_hf_test": Y_hf[idx[n_lf + n_hf + n_cal:]],
        "Y_lf_test": Y_lf[idx[n_lf + n_hf + n_cal:]],
        "X_cal": X2[idx[n_lf + n_hf:n_lf + n_hf + n_cal]], "Y_hf_cal": Y_hf[idx[n_lf + n_hf:n_lf + n_hf + n_cal]],
        "Y_lf_cal": Y_lf[idx[n_lf + n_hf:n_lf + n_hf + n_cal]],
    }
    standardise_separately(data, standardize_x=True)
    data["_fit_cache"] = {}
    r = np.corrcoef(Y_lf, Y_hf)[0, 1]
    return data, X, Y_hf, Y_lf, r


# ── Run CQR experiment ────────────────────────────────────────────
def run_experiment(cfg, outdir):
    data, X, Y_hf, Y_lf, r = generate_data(cfg)
    seed = cfg["seed"]
    tag = cfg["tag"]
    mu_hf, sig_hf = data["mu_hf"], data["sig_hf"]
    mu_x, sig_x = data.get("mu_x", 0), data.get("sig_x", 1)

    methods = build_methods(
        alpha=ALPHA, backend="rf", random_state=seed,
        rf_cdf_kwargs={"min_samples_leaf": 10},
        rf_mean_min_leaf=cfg["rf_mean_min_leaf"],
        rf_crossfit_min_leaf=cfg["rf_crossfit_min_leaf"],
        one_step_crossfit=True, one_step_tune_gamma=True,
        one_step_gamma_grid=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        one_step_density_source="hf", one_step_density_floor=0.1,
    )

    results = {}
    w_hf = None
    print(f"\n{'='*60}")
    print(f"  {tag} | seed={seed} | r={r:.3f}")
    print(f"{'='*60}")

    for mn in cfg["methods"]:
        m = methods[mn]
        m.fit(data)
        raw_lo, raw_up = m.predict_std(data["X_test"])
        lo_c, up_c = m.predict_std(data["X_cal"])
        q = conformal_calibrate(lo_c, up_c, data["Y_hf_cal_std"], ALPHA)
        cqr_lo, cqr_up = raw_lo - q, raw_up + q
        cov = float(np.mean((data["Y_hf_test_std"] >= cqr_lo) & (data["Y_hf_test_std"] <= cqr_up)))
        wid = float(np.mean(cqr_up - cqr_lo))
        if mn == "HF-Only":
            w_hf = wid
        pct = (wid - w_hf) / w_hf * 100
        g = getattr(m, "one_step_gamma_", "-")
        print(f"  {SHORT[mn]:<12} cov={cov:.3f} wid={wid:.4f} {pct:+.1f}%  g={g}")
        results[mn] = {
            "cov": cov, "wid": wid, "pct": pct,
            "x_test": (data["X_test"] * sig_x + mu_x).ravel(),
            "y_test": data["Y_hf_test"],
            "cqr_lo": cqr_lo * sig_hf + mu_hf,
            "cqr_up": cqr_up * sig_hf + mu_hf,
        }

    # Save results JSON
    os.makedirs(outdir, exist_ok=True)
    summary = {mn: {"cov": results[mn]["cov"], "wid": results[mn]["wid"], "pct": results[mn]["pct"]}
               for mn in cfg["methods"]}
    with open(os.path.join(outdir, f"{tag}_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return results, data, methods


# ── Plotting ──────────────────────────────────────────────────────
def plot_cqr(cfg, results, methods, outdir):
    tag = cfg["tag"]
    method_list = cfg["methods"]
    xl, xr = cfg["xl"], cfg["xr"]
    X_grid = np.linspace(xl, xr, 1000)
    f_grid = cfg["f_fn"](X_grid)

    for mn in method_list:
        short_name = SHORT[mn]
        res = results[mn]
        idx2 = np.argsort(res["x_test"])

        fig, ax = plt.subplots(1, 1, figsize=(8, 3.3))
        ax.fill_between(res["x_test"][idx2], res["cqr_lo"][idx2], res["cqr_up"][idx2],
                        alpha=0.25, color=COLORS[mn])
        ax.scatter(res["x_test"][idx2], res["y_test"][idx2], alpha=0.3, s=8, c="black", marker=".", zorder=3)
        ax.plot(X_grid, f_grid, "k--", lw=1, alpha=0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.tight_layout()
        safe_name = short_name.replace(" ", "_").replace("+", "plus")
        plt.savefig(os.path.join(outdir, f"{tag}_cqr_{safe_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {tag}_cqr_{safe_name}.png saved")


def plot_data(cfg, outdir):
    tag = cfg["tag"]
    seed = cfg["seed"]
    xl, xr = cfg["xl"], cfg["xr"]
    X_grid = np.linspace(xl, xr, 1000)
    f_grid = cfg["f_fn"](X_grid)

    # Generate 2500 points each for display
    rng_data = np.random.RandomState(seed + 50)
    X_data = rng_data.uniform(xl, xr, 2500)
    f_data = cfg["f_fn"](X_data)
    Y_hf_data = cfg["hf_gen"](np.random.RandomState(seed + 50), X_data, f_data)
    Y_lf_data = cfg["lf_gen"](np.random.RandomState(seed + 51), X_data, f_data)
    r = np.corrcoef(Y_lf_data, Y_hf_data)[0, 1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 3.3))
    ax.scatter(X_data, Y_lf_data, alpha=0.35, s=8, c="tab:blue", label="LF Response")
    ax.scatter(X_data, Y_hf_data, alpha=0.25, s=8, c="tab:red", label="HF Response", zorder=3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # ax.set_title(f"{tag} (seed={seed}, r={r:.3f})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_data.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {tag}_data.png saved")


def plot_transform(cfg, data, methods, outdir):
    tag = cfg["tag"]
    seed = cfg["seed"]
    xl, xr = cfg["xl"], cfg["xr"]
    mu_x, sig_x = data.get("mu_x", 0), data.get("sig_x", 1)
    mu_hf, sig_hf = data["mu_hf"], data["sig_hf"]
    X_grid = np.linspace(xl, xr, 1000)
    f_grid = cfg["f_fn"](X_grid)
    X_grid_std = ((X_grid - mu_x) / sig_x).reshape(-1, 1)

    m_troff = methods["Transfer (offset)"]

    # 10k fresh HF points
    rng2 = np.random.RandomState(999)
    X_u = rng2.uniform(xl, xr, 10000).reshape(-1, 1)
    f_u = cfg["f_fn"](X_u.ravel())
    Y_u = cfg["hf_gen"](rng2, X_u.ravel(), f_u)
    X_u_std = (X_u - mu_x) / sig_x
    Y_u_std = (Y_u - mu_hf) / sig_hf

    # Compute U using the actual Tr(off) pipeline
    hf_aligned = m_troff._invert_link(Y_u_std, m_troff.residual_ab_)
    hf_resid = hf_aligned - m_troff.lf_mean_model_.predict(X_u_std)
    U_plot = np.clip(m_troff.lf_resid_cdf_.cdf_batch(X_u_std, hf_resid), 1e-3, 1 - 1e-3)

    # Smoother lines
    u_lo_raw = np.clip(m_troff._z_to_u(m_troff.u_lo_model_.predict(X_grid_std)), 0.001, 0.999)
    u_up_raw = np.clip(m_troff._z_to_u(m_troff.u_up_model_.predict(X_grid_std)), 0.001, 0.999)
    u_lo_s = uniform_filter1d(u_lo_raw, size=SMOOTH_SIZE)
    u_up_s = uniform_filter1d(u_up_raw, size=SMOOTH_SIZE)

    # --- Raw HF outcome (individual plot) ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.scatter(X_u.ravel(), Y_u, alpha=0.15, s=3, c="tab:red")
    ax.plot(X_grid, f_grid, "k--", lw=1.5, alpha=0.5, label=r"$\mu(x)$")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_raw_hf.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {tag}_raw_hf.png saved")

    # --- Transformed outcome (individual plot) ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.3))
    ax.scatter(X_u.ravel(), U_plot, alpha=0.2, s=3, c="grey")
    ax.plot(X_grid, u_lo_s, "-", color="forestgreen", lw=2, label="0.05-quantile")
    ax.plot(X_grid, u_up_s, "-", color="forestgreen", lw=2, label="0.95-quantile")
    ax.axhline(0.05, color="gray", ls=":", lw=1, alpha=0.5)
    ax.axhline(0.95, color="gray", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("U")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_transform.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {tag}_transform.png saved")


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run synthetic MF-CQR experiments.")
    parser.add_argument("--experiment", choices=["P1", "Break2", "all"], default="all")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots and results.")
    args = parser.parse_args()

    if args.experiment == "all":
        exp_names = ["P1", "Break2"]
    else:
        exp_names = [args.experiment]

    outdir = args.output_dir or os.path.join(os.path.dirname(__file__), "synthetic_results")
    os.makedirs(outdir, exist_ok=True)

    for name in exp_names:
        cfg = EXPERIMENTS[name]
        results, data, methods = run_experiment(cfg, outdir)
        plot_cqr(cfg, results, methods, outdir)
        plot_data(cfg, outdir)

        # Transform plot uses larger n_hf=250 for visual clarity
        # (per draft caption); the table results above use cfg["n_hf"].
        cfg_viz = dict(cfg)
        cfg_viz["n_hf"] = cfg.get("n_hf_viz", 250)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            _, data_viz, methods_viz = run_experiment(cfg_viz, tmpdir)
        plot_transform(cfg, data_viz, methods_viz, outdir)
        del methods, data, methods_viz, data_viz


if __name__ == "__main__":
    main()
