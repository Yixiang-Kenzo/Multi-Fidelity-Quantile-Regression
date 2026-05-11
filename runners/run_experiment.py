import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

# Allow running this script directly from final_scripts/runners/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mfqr.models import (
    DEFAULT_GP_KWARGS,
    DEFAULT_RF_CDF_KWARGS,
    METHOD_SPECS,
    build_methods,
    create_mf_split,
    run_one,
    standardise_separately,
)


def _parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _print_dataset_summary(X, y_lf, y_hf):
    print(f"Loaded dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"LF:  mean={np.mean(y_lf):.4f}, std={np.std(y_lf):.6f}")
    print(f"HF:  mean={np.mean(y_hf):.4f}, std={np.std(y_hf):.6f}")
    print(f"LF-HF correlation: {np.corrcoef(y_lf, y_hf)[0, 1]:.4f}")


def _print_summary(results, method_names, hf_fracs, alpha):
    print(f"\n\n{'=' * 80}")
    print(f"  SUMMARY -- CQR metrics (standardised units), mean +/- std over {len(next(iter(results.values()))[hf_fracs[0]])} seeds")
    print(f"  Target coverage: {1 - alpha:.0%}")
    print(f"{'=' * 80}")

    col_w = 24
    header = f"  {'Method':<20}" + "".join(
        f"  {'n2=' + f'{f:.0%}':>{col_w}}" for f in hf_fracs
    )
    print(header)
    print(f"  {'-' * 20}" + "".join(f"  {'-' * col_w}" for _ in hf_fracs))

    for name in method_names:
        row = f"  {name:<20}"
        for hf_frac in hf_fracs:
            vals = results[name][hf_frac]
            covs = [v[0] for v in vals]
            wids = [v[1] for v in vals]
            row += f"  cov={np.mean(covs):.3f} wid={np.mean(wids):.4f}+/-{np.std(wids):.4f}"
        print(row)

    print("\n  Width improvements vs HF-Only (negative = method is narrower):")
    for hf_frac in hf_fracs:
        w_hf = np.mean([v[1] for v in results['HF-Only'][hf_frac]])
        for name in method_names:
            w = np.mean([v[1] for v in results[name][hf_frac]])
            pct = (w - w_hf) / w_hf * 100
            print(f"    n2={hf_frac:.0%}  {name} vs HF-Only : {pct:+.1f}%")
    print(f"\n{'=' * 80}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Run multi-fidelity conformal quantile regression experiments.")
    parser.add_argument("--backend", choices=["rf", "gp"], default="rf", help="Base learner backend.")
    parser.add_argument("--dataset", default="QeMFi_acrolein.npz", help="Path to dataset (.npz).")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--hf-fracs", default="0.05", help="Comma-separated HF fractions (ignored, kept for compatibility).")
    parser.add_argument("--seeds", default="42", help="Comma-separated integer seeds.")
    parser.add_argument("--test-frac", type=float, default=0.4)
    parser.add_argument("--n-hf-override", type=int, default=None, help="Explicit n_hf; overrides 12-section default.")
    parser.add_argument("--n-lf-override", type=int, default=None, help="Explicit n_lf; overrides 12-section default.")
    parser.add_argument("--n-cal-override", type=int, default=None, help="Explicit n_cal; overrides 12-section default (use to decouple cal from hf).")
    parser.add_argument("--standardize-x", dest="standardize_x", action="store_true", help="Standardize X before fitting.")
    parser.add_argument("--no-standardize-x", dest="standardize_x", action="store_false")
    parser.set_defaults(standardize_x=True)
    parser.add_argument("--stage2-fixed-rho", type=float, default=0.0)
    parser.add_argument("--stage2-cv", action="store_true", help="Tune stage-2 rho by CV.")

    # One-step correction arguments
    parser.add_argument("--one-step-gamma", type=float, default=1.0)
    parser.add_argument("--one-step-tune-gamma", action="store_true")
    parser.add_argument("--one-step-gamma-grid", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--one-step-density-source", choices=["lf", "hf"], default="hf")
    parser.add_argument("--one-step-density-floor", type=float, default=0.1)
    parser.add_argument("--one-step-max-update", type=float, default=1.0)
    parser.add_argument("--one-step-crossfit", dest="one_step_crossfit", action="store_true")
    parser.add_argument("--no-one-step-crossfit", dest="one_step_crossfit", action="store_false")
    parser.add_argument("--one-step-cf-folds", type=int, default=5)
    parser.set_defaults(one_step_crossfit=True)

    # GP hyperparameters (fixed)
    parser.add_argument("--gp-length-scale", type=float, default=DEFAULT_GP_KWARGS["length_scale"])
    parser.add_argument("--gp-n-features", type=int, default=DEFAULT_GP_KWARGS["n_features"])
    parser.add_argument("--gp-noise-var", type=float, default=DEFAULT_GP_KWARGS["noise_var"])
    parser.add_argument("--gp-prior-var", type=float, default=DEFAULT_GP_KWARGS["prior_var"])

    # RF hyperparameters (fixed)
    parser.add_argument("--rf-n-estimators", type=int, default=DEFAULT_RF_CDF_KWARGS["n_estimators"])
    parser.add_argument("--rf-min-leaf", type=int, default=DEFAULT_RF_CDF_KWARGS["min_samples_leaf"])
    parser.add_argument("--rf-moment-order", type=int, default=DEFAULT_RF_CDF_KWARGS["moment_order"])
    parser.add_argument("--rf-split-mode", default=DEFAULT_RF_CDF_KWARGS["split_mode"])
    parser.add_argument("--rf-dist-mode", default=DEFAULT_RF_CDF_KWARGS["dist_mode"])

    parser.add_argument("--output-dir", default="results", help="Directory to save JSON results.")
    parser.add_argument("--methods", default=None, help="Comma-separated method names to run (default: all).")

    args = parser.parse_args()

    hf_fracs = _parse_float_list(args.hf_fracs)
    seeds = _parse_int_list(args.seeds)
    one_step_gamma_grid = _parse_float_list(args.one_step_gamma_grid)
    all_method_names = list(METHOD_SPECS.keys())
    if args.methods:
        method_names = [m.strip() for m in args.methods.split(",")]
        for m in method_names:
            if m not in all_method_names:
                raise ValueError(f"Unknown method: {m!r}. Available: {all_method_names}")
    else:
        method_names = all_method_names

    gp_kwargs = {
        "length_scale": args.gp_length_scale,
        "n_features": args.gp_n_features,
        "noise_var": args.gp_noise_var,
        "prior_var": args.gp_prior_var,
    }
    rf_cdf_kwargs = {
        "n_estimators": args.rf_n_estimators,
        "min_samples_leaf": args.rf_min_leaf,
        "split_mode": args.rf_split_mode,
        "moment_order": args.rf_moment_order,
        "dist_mode": args.rf_dist_mode,
    }
    stage2_fixed_rho = None if args.stage2_cv else args.stage2_fixed_rho
    standardize_x = args.standardize_x

    print("=" * 72)
    print(f"  Backend={args.backend.upper()}, Seeds={seeds}, alpha={args.alpha}")
    print(f"  RF leaf={args.rf_min_leaf}, GP ls={args.gp_length_scale}")
    print("=" * 72)

    # Load dataset.  Two conventions are supported:
    #   (a) generic multi-fidelity npz with keys X, Y_lf, Y_hf
    #       (e.g., Burgers, Materials Project Formation Energy).
    #   (b) QeMFi-format npz with keys R (geometries) and SCF (energy matrix);
    #       LF = STO-3G  (column 0 of SCF), HF = def2-TZVP (column 4).
    _d = np.load(args.dataset)
    if 'X' in _d and 'Y_lf' in _d and 'Y_hf' in _d:
        X, y_lf, y_hf = _d['X'], _d['Y_lf'], _d['Y_hf']
    elif 'R' in _d and 'SCF' in _d:
        X = _d['R'].reshape(_d['R'].shape[0], -1)
        y_lf = _d['SCF'][:, 0]
        y_hf = _d['SCF'][:, 4]
    else:
        raise ValueError(
            f"Dataset {args.dataset} has unrecognised keys: {list(_d.keys())}. "
            "Expected either (X, Y_lf, Y_hf) or QeMFi-format (R, SCF)."
        )

    _print_dataset_summary(X, y_lf, y_hf)

    results = {m: defaultdict(list) for m in method_names}
    all_rows = []

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    out_path = os.path.join(args.output_dir, f"{dataset_name}_{args.backend}_results.jsonl")

    for seed in seeds:
        print(f"\n\n{'#' * 72}")
        print(f"  SEED = {seed}")
        print(f"{'#' * 72}")

        for hf_frac in hf_fracs:
            data = create_mf_split(
                X, y_lf, y_hf,
                test_frac=args.test_frac,
                seed=seed,
                n_hf=args.n_hf_override,
                n_lf=args.n_lf_override,
                n_cal=args.n_cal_override,
            )
            standardise_separately(data, standardize_x=standardize_x)
            n2_actual = len(data["X_hf"])

            print(f"\n  {'=' * 60}")
            print(f"  n2={hf_frac:.0%} ({n2_actual} HF pts), seed={seed}")
            print(f"  {'=' * 60}")

            data["_fit_cache"] = {}

            methods = build_methods(
                alpha=args.alpha,
                backend=args.backend,
                random_state=seed,
                gp_kwargs=gp_kwargs,
                rf_cdf_kwargs=rf_cdf_kwargs,
                rf_mean_min_leaf=args.rf_min_leaf,
                rf_crossfit_min_leaf=args.rf_min_leaf,
                stage2_fixed_rho=stage2_fixed_rho,
                one_step_crossfit=args.one_step_crossfit,
                one_step_cf_folds=args.one_step_cf_folds,
                one_step_gamma=args.one_step_gamma,
                one_step_tune_gamma=args.one_step_tune_gamma,
                one_step_gamma_grid=one_step_gamma_grid,
                one_step_density_source=args.one_step_density_source,
                one_step_density_floor=args.one_step_density_floor,
                one_step_max_update=args.one_step_max_update,
            )

            for name in method_names:
                print(f"\n  [{name}]")
                t0 = time.time()
                raw_cov, raw_wid, cqr_cov, cqr_wid = run_one(methods[name], data, args.alpha)
                elapsed = time.time() - t0
                print(
                    f"    raw: cov={raw_cov:.4f} wid={raw_wid:.5f} | "
                    f"CQR: cov={cqr_cov:.4f} wid={cqr_wid:.5f}  [{elapsed:.0f}s]"
                )
                results[name][hf_frac].append((cqr_cov, cqr_wid))
                row = {
                    "dataset": dataset_name,
                    "backend": args.backend,
                    "method": name,
                    "seed": seed,
                    "hf_frac": hf_frac,
                    "n2": n2_actual,
                    "cqr_cov": round(cqr_cov, 6),
                    "cqr_wid": round(cqr_wid, 6),
                    "raw_cov": round(raw_cov, 6),
                    "raw_wid": round(raw_wid, 6),
                }
                with open(out_path, "a") as f:
                    f.write(json.dumps(row) + "\n")

    _print_summary(results, method_names, hf_fracs, args.alpha)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
