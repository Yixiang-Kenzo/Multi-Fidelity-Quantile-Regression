from typing import Dict

import numpy as np
from sklearn.model_selection import train_test_split


def standardise_separately(data: Dict, standardize_x: bool = True):
    """
    Standardize LF and HF responses separately.

    Parameters
    ----------
    data : dict
        Split dictionary from create_mf_split.
    standardize_x : bool, default=True
        If True, standardize covariates using LF-training statistics.
    """
    if standardize_x:
        mu_x = np.mean(data["X_lf"], axis=0)
        sig_x = np.std(data["X_lf"], axis=0)
        sig_x = np.where(sig_x < 1e-12, 1.0, sig_x)
        for key in ["X_lf", "X_hf", "X_test", "X_cal"]:
            data[key] = (data[key] - mu_x) / sig_x
        data["mu_x"], data["sig_x"] = mu_x, sig_x

    mu_lf = float(np.mean(data["Y_lf_train"]))
    sig_lf = float(np.std(data["Y_lf_train"]))
    mu_hf = float(np.mean(data["Y_hf_train"]))
    sig_hf = float(np.std(data["Y_hf_train"]))

    sig_lf = sig_lf if sig_lf > 1e-12 else 1.0
    sig_hf = sig_hf if sig_hf > 1e-12 else 1.0

    data["mu_lf"], data["sig_lf"] = mu_lf, sig_lf
    data["mu_hf"], data["sig_hf"] = mu_hf, sig_hf

    data["Y_lf_std"] = (data["Y_lf_train"] - mu_lf) / sig_lf
    data["Y_hf_std"] = (data["Y_hf_train"] - mu_hf) / sig_hf
    data["Y_hf_test_std"] = (data["Y_hf_test"] - mu_hf) / sig_hf
    data["Y_hf_cal_std"] = (data["Y_hf_cal"] - mu_hf) / sig_hf
    return mu_lf, sig_lf, mu_hf, sig_hf


def conformal_calibrate(lower, upper, y_cal, alpha):
    scores = np.maximum(lower - y_cal, y_cal - upper)
    n = len(scores)
    q_level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    try:
        return np.quantile(scores, q_level, method="higher")
    except TypeError:
        return np.quantile(scores, q_level, interpolation="higher")


def create_mf_split(
    X,
    y_lf,
    y_hf,
    test_frac=0.40,
    seed=42,
    n_hf=None,
    n_lf=None,
    n_cal=None,
):
    """Split data into disjoint LF training, HF training, HF calibration, and test sets.

    Default behaviour (no override) — 12 equal sections of training data:
      - Test  = test_frac (default 40%)
      - Train = 1 - test_frac, divided into 12 equal sections:
          - LF  = 10 sections, HF = 1 section, Cal = 1 section.

    Override behaviour — if any of n_hf/n_lf/n_cal is provided, those values are
    used directly (other unset slots fill in with 12-section defaults so the
    common case of "just bump cal" works). Useful when n_hf and n_cal need to
    differ (e.g., small-HF stress test with bigger cal for stable conformal
    calibration). Caller must ensure n_hf + n_lf + n_cal <= n_train.
    """
    # Step 1: Split into train and test
    X_train, X_test, ylf_tr, ylf_te, yhf_tr, yhf_te = train_test_split(
        X, y_lf, y_hf, test_size=test_frac, random_state=seed
    )

    # Step 2: Shuffle training, then carve out the requested sizes
    rng = np.random.RandomState(seed)
    n_train = len(X_train)
    idx = rng.permutation(n_train)
    section_size = n_train // 12

    # Resolve sizes: explicit overrides take priority, else fall back to 12-section defaults
    n_lf_use = int(n_lf) if n_lf is not None else 10 * section_size
    n_hf_use = int(n_hf) if n_hf is not None else section_size
    n_cal_use = int(n_cal) if n_cal is not None else section_size
    total = n_lf_use + n_hf_use + n_cal_use
    if total > n_train:
        raise ValueError(f"requested n_lf+n_hf+n_cal={total} exceeds available train budget {n_train}")

    lf_idx = idx[:n_lf_use]
    hf_idx = idx[n_lf_use:n_lf_use + n_hf_use]
    cal_idx = idx[n_lf_use + n_hf_use:n_lf_use + n_hf_use + n_cal_use]

    data = {
        "X_lf": X_train[lf_idx],
        "Y_lf_train": ylf_tr[lf_idx],
        "X_hf": X_train[hf_idx],
        "Y_hf_train": yhf_tr[hf_idx],
        "Y_lf_at_hf": ylf_tr[hf_idx],
        "X_test": X_test,
        "Y_hf_test": yhf_te,
        "Y_lf_test": ylf_te,
        "X_cal": X_train[cal_idx],
        "Y_hf_cal": yhf_tr[cal_idx],
        "Y_lf_cal": ylf_tr[cal_idx],
    }

    print("\n--- Data Split ---")
    print(f"  LF training (n1): {len(data['X_lf'])}")
    print(f"  HF training (n2): {len(data['X_hf'])}")
    print(f"  Calibration:      {len(data['X_cal'])}")
    print(f"  Test:             {len(data['X_test'])}")
    return data
