"""Section 5.2.3 Misinformative regime: data + CQR plots.

Generates:
  - Misinformative_data.png            (scatter of LF/HF responses)
  - Misinformative_cqr_HF-Only.png
  - Misinformative_cqr_MFQR.png
  - Misinformative_cqr_MFQR_plus_OS.png
  - Misinformative_cqr_MFQR_plus_MS.png   (multi-step: m chosen by CV pinball)

The one-step correction is run in *raw HF response space* (one_step_space='raw'),
matching the §5.2.3 protocol from the paper: the bias is in the affine offset
itself, so residual-space iteration would propagate the offset bias.

Multi-step m is selected by minimum cross-fit pinball loss over m=2..5 with
gamma=1.  MSE against the closed-form oracle quantile is also reported for
diagnostic transparency.
"""
import argparse
import os
import sys

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from final_scripts/figures/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mfqr import build_methods, standardise_separately, conformal_calibrate

ALPHA = 0.10
TAU_LO, TAU_UP = ALPHA / 2, 1.0 - ALPHA / 2
SEED = 300
N = 5000
N_LF, N_HF, N_CAL = 2500, 350, 500
SIGMA_H = 0.3
SIGMA_L = 0.3
M_GRID = (2, 3, 4, 5)


def mu_hf_fn(X):  return np.sin(2 * np.pi * X)
def mu_lf_fn(X):  return 0.5 * mu_hf_fn(X) + 1.5 * np.sin(4 * np.pi * X)
def g_fn(X):      return 0.5 + (2.0 * X - 1.0) ** 2


def hf_gen(rng, X):
    return mu_hf_fn(X) + SIGMA_H * g_fn(X) * rng.normal(0, 1, len(X))


def lf_gen(rng, X):
    return mu_lf_fn(X) + SIGMA_L * g_fn(X) * rng.normal(0, 1, len(X))


def pinball(y, q, tau):
    err = y - q
    return float(np.mean(np.maximum(tau * err, (tau - 1.0) * err)))


def build_data():
    """Reproduces plot_d7v1_nhf350.py's seed/split exactly."""
    rng = np.random.RandomState(SEED)
    X = rng.uniform(0, 1, N)
    Y_hf = hf_gen(rng, X)
    rng_lf = np.random.RandomState(SEED + 99)
    Y_lf = lf_gen(rng_lf, X)

    rng_split = np.random.RandomState(SEED + 1)
    idx = rng_split.permutation(N)
    X2 = X.reshape(-1, 1)
    s_lf, s_hf, s_cal = N_LF, N_HF, N_CAL
    data = {
        "X_lf": X2[idx[:s_lf]], "Y_lf_train": Y_lf[idx[:s_lf]],
        "X_hf": X2[idx[s_lf:s_lf + s_hf]], "Y_hf_train": Y_hf[idx[s_lf:s_lf + s_hf]],
        "Y_lf_at_hf": Y_lf[idx[s_lf:s_lf + s_hf]],
        "X_test": X2[idx[s_lf + s_hf + s_cal:]],
        "Y_hf_test": Y_hf[idx[s_lf + s_hf + s_cal:]],
        "Y_lf_test": Y_lf[idx[s_lf + s_hf + s_cal:]],
        "X_cal": X2[idx[s_lf + s_hf:s_lf + s_hf + s_cal]],
        "Y_hf_cal": Y_hf[idx[s_lf + s_hf:s_lf + s_hf + s_cal]],
        "Y_lf_cal": Y_lf[idx[s_lf + s_hf:s_lf + s_hf + s_cal]],
    }
    standardise_separately(data, standardize_x=True)
    data["_fit_cache"] = {}
    return data


def plot_data_figure(outdir):
    rng_data = np.random.RandomState(SEED + 50)
    X_data = rng_data.uniform(0, 1, 2500)
    Y_hf_data = hf_gen(np.random.RandomState(SEED + 50), X_data)
    Y_lf_data = lf_gen(np.random.RandomState(SEED + 51), X_data)

    fig, ax = plt.subplots(1, 1, figsize=(8, 3.3))
    ax.scatter(X_data, Y_lf_data, alpha=0.5, s=5, c="tab:blue", label="LF response")
    ax.scatter(X_data, Y_hf_data, alpha=0.2, s=8, c="tab:red", label="HF response", zorder=3)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    out = os.path.join(outdir, "Misinformative_data.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def iterate_raw_hf(m_os, cf_models, X_input, max_m, gamma):
    """m steps of Newton in raw HF outcome space, averaged across cross-fit folds."""
    lo_all, up_all = [], []
    for obj in cf_models:
        info = m_os._transfer_residual_pilot_info(X_input, model=obj["base"])
        q_lo = info["q0_lo"].copy(); q_up = info["q0_up"].copy()
        for _ in range(max_m):
            q_lo = m_os._one_step_update(X_input, q_lo, m_os.tau_lo,
                                          hf_cdf_model=obj["hf_cdf"], gamma=gamma)
            q_up = m_os._one_step_update(X_input, q_up, m_os.tau_up,
                                          hf_cdf_model=obj["hf_cdf"], gamma=gamma)
        lo_all.append(q_lo); up_all.append(q_up)
    return np.mean(np.vstack(lo_all), axis=0), np.mean(np.vstack(up_all), axis=0)


def cv_pinball_for_m(m_os, fold_objects, data, m_value, gamma):
    """Sum of lo+up pinball loss across the K=5 OS cross-fit folds."""
    losses = []
    for obj in fold_objects:
        va = obj["va"]
        X_va = data["X_hf"][va]
        y_va = data["Y_hf_std"][va]
        info = m_os._transfer_residual_pilot_info(X_va, model=obj["base"])
        q_lo = info["q0_lo"].copy(); q_up = info["q0_up"].copy()
        for _ in range(m_value):
            q_lo = m_os._one_step_update(X_va, q_lo, m_os.tau_lo,
                                          hf_cdf_model=obj["hf_cdf"], gamma=gamma)
            q_up = m_os._one_step_update(X_va, q_up, m_os.tau_up,
                                          hf_cdf_model=obj["hf_cdf"], gamma=gamma)
        losses.append(pinball(y_va, q_lo, m_os.tau_lo) + pinball(y_va, q_up, m_os.tau_up))
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None,
                        help="Output directory.  Default: alongside this script.")
    args = parser.parse_args()
    outdir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(outdir, exist_ok=True)

    print("\n=== §5.2.3 Misinformative regime ===\n")
    plot_data_figure(outdir)

    data = build_data()
    mu_hf, sig_hf = data["mu_hf"], data["sig_hf"]
    mu_x, sig_x = data.get("mu_x", 0), data.get("sig_x", 1)
    X_test_orig = (data["X_test"].ravel() * sig_x + mu_x)

    # ---- Build & fit methods (raw-HF-space OS) ----
    methods = build_methods(
        alpha=ALPHA, backend="rf", random_state=SEED,
        rf_cdf_kwargs={"min_samples_leaf": 10},
        one_step_crossfit=True, one_step_tune_gamma=True,
        one_step_gamma_grid=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        one_step_density_source="hf", one_step_density_floor=0.1,
        one_step_space="raw",
    )
    for m in methods.values():
        m.rf_mean_min_leaf = 30
        m.rf_crossfit_min_leaf = 30

    m_hf = methods["HF-Only"]
    m_mq = methods["Transfer (offset)"]
    m_os = methods["Transfer (offset) + 1Step"]
    for m in [m_hf, m_mq, m_os]:
        m.fit(data)

    cf_models = m_os.one_step_cf_models_
    fold_objects = m_os._prepare_one_step_fold_objects(data)

    # ---- CV-select m for MFQR + MS ----
    print("\n  m-selection by CV pinball loss (over m=2..5, gamma=1):")
    cv_losses = {}
    for m_val in M_GRID:
        cv_losses[m_val] = cv_pinball_for_m(m_os, fold_objects, data, m_val, 1.0)
        print(f"    m={m_val}: CV pinball = {cv_losses[m_val]:.6f}")
    m_hat = min(cv_losses, key=cv_losses.get)
    print(f"\n  CV-selected m_hat = {m_hat} (lowest pinball)")

    # ---- Oracle MSE (HF noise is Normal so q*_tau is closed-form) ----
    oracle_lo = mu_hf_fn(X_test_orig) + SIGMA_H * g_fn(X_test_orig) * norm.ppf(TAU_LO)
    oracle_up = mu_hf_fn(X_test_orig) + SIGMA_H * g_fn(X_test_orig) * norm.ppf(TAU_UP)

    # ---- Collect entries to plot (in HF outcome space, post-CQR) ----
    def finalize(lo, up, lc, uc):
        q = conformal_calibrate(lc, uc, data["Y_hf_cal_std"], ALPHA)
        return (lo - q) * sig_hf + mu_hf, (up + q) * sig_hf + mu_hf

    def mse_vs_oracle_raw(raw_lo_std, raw_up_std):
        """MSE on the raw (pre-CQR) quantile estimate, after de-standardising."""
        lo_orig = raw_lo_std * sig_hf + mu_hf
        up_orig = raw_up_std * sig_hf + mu_hf
        return 0.5 * (np.mean((lo_orig - oracle_lo) ** 2)
                      + np.mean((up_orig - oracle_up) ** 2))

    lo_hf, up_hf = m_hf.predict_std(data["X_test"])
    lo_hf_c, up_hf_c = m_hf.predict_std(data["X_cal"])
    lo_mq, up_mq = m_mq.predict_std(data["X_test"])
    lo_mq_c, up_mq_c = m_mq.predict_std(data["X_cal"])
    lo_os, up_os = m_os.predict_std(data["X_test"])
    lo_os_c, up_os_c = m_os.predict_std(data["X_cal"])

    # Multi-step at m_hat
    lo_ms, up_ms = iterate_raw_hf(m_os, cf_models, data["X_test"], m_hat, 1.0)
    lo_ms_c, up_ms_c = iterate_raw_hf(m_os, cf_models, data["X_cal"], m_hat, 1.0)

    entries = [
        ("HF-Only",         lo_hf, up_hf, lo_hf_c, up_hf_c, "#d62728", "MFQR_plus_MS"),  # placeholder
    ]
    # we use a structured plan instead:
    # Filename for the multi-step plot reflects the CV-selected m so it matches
    # the paper's referenced figure name (paper currently uses m=5).
    ms_filename = f"Misinformative_cqr_Iter_m{m_hat}.png"
    plot_plan = [
        ("Misinformative_cqr_HF-Only.png",       lo_hf, up_hf, lo_hf_c, up_hf_c, "#d62728"),
        ("Misinformative_cqr_MFQR.png",          lo_mq, up_mq, lo_mq_c, up_mq_c, "#2ca02c"),
        ("Misinformative_cqr_MFQR_plus_OS.png",  lo_os, up_os, lo_os_c, up_os_c, "#1f77b4"),
        (ms_filename,                         lo_ms, up_ms, lo_ms_c, up_ms_c, "#9467bd"),
    ]

    # Pre-compute final intervals for shared ylim
    finalized = []
    all_vals = [data["Y_hf_test"].min(), data["Y_hf_test"].max()]
    for name, lo, up, lc, uc, col in plot_plan:
        cqr_lo, cqr_up = finalize(lo, up, lc, uc)
        finalized.append((name, cqr_lo, cqr_up, col, lo, up))
        all_vals.extend([cqr_lo.min(), cqr_up.max()])
    ymin, ymax = min(all_vals) - 0.1, max(all_vals) + 0.1

    print("\n  metrics (CQR after conformal calibration):")
    print(f"    {'method':28s}  {'MSE-oracle':>11s}  {'CQR cov':>8s}  {'CQR wid':>8s}")
    idx2 = np.argsort(X_test_orig)
    y_test = data["Y_hf_test"]

    for name, cqr_lo, cqr_up, col, raw_lo, raw_up in finalized:
        cov = float(np.mean((y_test >= cqr_lo) & (y_test <= cqr_up)))
        wid = float(np.mean(cqr_up - cqr_lo))
        mse = mse_vs_oracle_raw(raw_lo, raw_up)
        label = name.replace("Misinformative_cqr_", "").replace(".png", "")
        print(f"    {label:28s}  {mse:11.6f}  {cov * 100:7.2f}%  {wid:8.4f}")

        fig, ax = plt.subplots(1, 1, figsize=(8, 3.3))
        ax.fill_between(X_test_orig[idx2], cqr_lo[idx2], cqr_up[idx2], alpha=0.25, color=col)
        ax.scatter(X_test_orig[idx2], y_test[idx2], alpha=0.3, s=8, c="black", marker=".", zorder=3)
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_ylim(ymin, ymax)
        plt.tight_layout()
        out = os.path.join(outdir, name)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"      saved {out}")

    print(f"\n  Done.  CV-selected m for MFQR+MS = {m_hat}.")


if __name__ == "__main__":
    main()
