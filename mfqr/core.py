import contextlib
import io
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from .cdf_models import (
    DEFAULT_GP_KWARGS,
    DEFAULT_RF_CDF_KWARGS,
    ConditionalCDF_QRF,
    ConditionalCDF_RFFGP,
    RFFMeanRegressor,
)
from .data_utils import conformal_calibrate


def _freeze_for_key(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return tuple(sorted((k, _freeze_for_key(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple, np.ndarray)):
        return tuple(_freeze_for_key(v) for v in obj)
    return obj

METHOD_SPECS = {
    "HF-Only": {"mode": "hf", "hf_strategy": "raw"},
    "HF-Only (augment)": {"mode": "hf", "hf_strategy": "augment"},
    "HF-Only (offset)": {"mode": "hf", "hf_strategy": "offset"},
    "Transfer": {"mode": "transfer", "transfer_target": "outcome", "hf_strategy": "raw"},
    "Transfer (augment)": {"mode": "transfer", "transfer_target": "outcome", "hf_strategy": "augment"},
    "Transfer (offset)": {"mode": "transfer", "transfer_target": "residual", "hf_strategy": "raw"},
    "Transfer + 1Step": {
        "mode": "transfer",
        "transfer_target": "outcome",
        "hf_strategy": "raw",
        "one_step_correction": True,
    },
    "Transfer (augment) + 1Step": {
        "mode": "transfer",
        "transfer_target": "outcome",
        "hf_strategy": "augment",
        "one_step_correction": True,
    },
    "Transfer (offset) + 1Step": {
        "mode": "transfer",
        "transfer_target": "residual",
        "hf_strategy": "raw",
        "one_step_correction": True,
    },
}

class PosteriorQuantileSmoother:
    """
    Stage-2 quantile smoother on U in [0,1].

    It fits a backend-specific conditional CDF model on (X_hf, U_hf), then either:
      - uses the pure backend quantile model when rho = 0, or
      - shrinks the backend CDF toward Uniform(0,1):
            F_post(u | x) = rho * u + (1-rho) * F_base(u | x).
    """

    def __init__(
        self,
        tau,
        cdf_builder,
        random_state=42,
        n_cv_folds=5,
        rho_grid=None,
        fixed_rho=None,
        report_param=True,
        quiet_base_fit=True,
        clip_eps=1e-3,
        n_bisect_iter=30,
    ):
        if not (0 < tau < 1):
            raise ValueError("tau must lie in (0,1).")
        self.tau = float(tau)
        self.cdf_builder = cdf_builder
        self.random_state = int(random_state)
        self.n_cv_folds = int(n_cv_folds)
        self.rho_grid = (
            np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0], dtype=float)
            if rho_grid is None
            else np.array(rho_grid, dtype=float)
        )
        self.fixed_rho = fixed_rho
        self.report_param = report_param
        self.quiet_base_fit = quiet_base_fit
        self.clip_eps = float(clip_eps)
        self.n_bisect_iter = int(n_bisect_iter)


    def fit(self, X, U):
        X = np.asarray(X)
        U = np.asarray(U).reshape(-1)
        U = np.clip(U, self.clip_eps, 1.0 - self.clip_eps)

        self.sorted_U_ = np.sort(U.copy())
        self.n_U_ = len(U)


        if self.fixed_rho is None:
            self.rho_, self.cv_loss_ = self._tune_rho(X, U)
        else:
            self.rho_ = float(self.fixed_rho)
            self.cv_loss_ = np.nan
        self.base_cdf_ = self._fit_base(X, U)
        if self.report_param:
            if np.isnan(self.cv_loss_):
                if self.rho_ > 0.0:
                    print(f"    Stage-2: fixed shrinkage rho = {self.rho_:.6f}")
            else:
                tag = "pure model on U selected by CV" if self.rho_ == 0.0 else f"shrinkage rho = {self.rho_:.6f}"
                print(f"    Stage-2: {tag} (CV pinball = {self.cv_loss_:.6f})")
        return self

    def _global_cdf(self, u):
        u = np.asarray(u).reshape(-1)
        return np.searchsorted(self.sorted_U_, u, side="right") / self.n_U_

    def _global_quantile(self, tau):
        tau = np.asarray(tau)
        idx = np.clip((tau * (self.n_U_ - 1)).astype(int), 0, self.n_U_ - 1)
        return self.sorted_U_[idx]


    def predict(self, X_new):
        X_new = np.asarray(X_new)
        if self.rho_ == 0.0:
            tau_mat = np.full((len(X_new), 1), self.tau, dtype=float)
            q = self.base_cdf_.quantile_varying_multi(X_new, tau_mat).reshape(-1)
            return np.clip(q, self.clip_eps, 1.0 - self.clip_eps)
        q = self._posterior_quantile(self.base_cdf_, X_new, self.tau, self.rho_)
        return np.clip(q, self.clip_eps, 1.0 - self.clip_eps)

    def _fit_base(self, X, U):
        model = self.cdf_builder()
        if self.quiet_base_fit:
            with contextlib.redirect_stdout(io.StringIO()):
                model.fit(X, U)
        else:
            model.fit(X, U)
        return model

    def _posterior_cdf(self, model, X, u, rho):
        u = np.asarray(u).reshape(-1)
        g = self._global_cdf(u)
        f_rf = model.cdf_batch(X, u)
        return rho * g + (1.0 - rho) * f_rf

    def _posterior_quantile(self, model, X_new, tau, rho):
        n = X_new.shape[0]
        lo = np.full(n, self.clip_eps, dtype=float)
        hi = np.full(n, 1.0 - self.clip_eps, dtype=float)
        for _ in range(self.n_bisect_iter):
            mid = 0.5 * (lo + hi)
            fmid = self._posterior_cdf(model, X_new, mid, rho)
            go_right = fmid < tau
            lo[go_right] = mid[go_right]
            hi[~go_right] = mid[~go_right]
        return 0.5 * (lo + hi)

    def _pinball_loss(self, y_true, y_pred):
        err = np.asarray(y_true) - np.asarray(y_pred)
        return np.mean(np.maximum(self.tau * err, (self.tau - 1.0) * err))

    def _tune_rho(self, X, U):
        n = len(U)
        if n < 2:
            return 0.0, 0.0
        n_splits = min(self.n_cv_folds, n)
        if n_splits < 2:
            return 0.0, 0.0
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        best_rho = None
        best_loss = np.inf
        for rho in self.rho_grid:
            losses = []
            for tr, va in kf.split(X):
                model = self._fit_base(X[tr], U[tr])
                if float(rho) == 0.0:
                    tau_mat = np.full((len(va), 1), self.tau, dtype=float)
                    pred = model.quantile_varying_multi(X[va], tau_mat).reshape(-1)
                else:
                    pred = self._posterior_quantile(model, X[va], self.tau, float(rho))
                pred = np.clip(pred, self.clip_eps, 1.0 - self.clip_eps)
                losses.append(self._pinball_loss(U[va], pred))
            mean_loss = float(np.mean(losses))
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_rho = float(rho)
        return best_rho, best_loss


# =============================================================================
# Shared experiment model
# =============================================================================

@dataclass
class BackendConfig:
    mean_type: str
    cdf_type: str
    standardize_x: bool
    rf_mean_min_leaf: int = 10
    rf_crossfit_min_leaf: int = 10
    gp_kwargs: Optional[Dict] = None
    cdf_kwargs: Optional[Dict] = None


class TransferQuantileRegressor:
    VALID_BACKENDS = {"rf", "gp"}
    VALID_MODES = {"hf", "transfer"}
    VALID_TRANSFER_TARGETS = {"outcome", "residual"}
    VALID_HF_STRATEGIES = {"raw", "offset", "augment"}

    def __init__(
        self,
        alpha=0.1,
        backend="rf",
        mode="transfer",
        transfer_target="outcome",
        hf_strategy="raw",
        random_state=42,
        n_jobs=-1,
        rf_mean_min_leaf=10,
        rf_crossfit_min_leaf=10,
        rf_cdf_kwargs=None,
        hf_cdf_kwargs=None,
        smoother_cdf_kwargs=None,
        gp_kwargs=None,
        hf_gp_kwargs=None,
        smoother_gp_kwargs=None,
        stage2_fixed_rho=0.0,
        stage2_rho_grid=None,
        stage2_cv_folds=5,
        augment_with_quantiles=False,
        augment_quantile_levels=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8, 0.9),
        augment_include_mean=True,
        one_step_correction=False,
        one_step_crossfit=False,
        one_step_cf_folds=5,
        one_step_gamma=1.0,
        one_step_tune_gamma=False,
        one_step_gamma_grid=None,
        one_step_density_source="hf",
        one_step_density_floor=0.1,
        one_step_max_update=1.0,
        one_step_space="residual",
        probit_transform=False,
    ):
        if backend not in self.VALID_BACKENDS:
            raise ValueError(f"backend must be one of {self.VALID_BACKENDS}")
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}")
        if transfer_target not in self.VALID_TRANSFER_TARGETS:
            raise ValueError(f"transfer_target must be one of {self.VALID_TRANSFER_TARGETS}")
        if hf_strategy not in self.VALID_HF_STRATEGIES:
            raise ValueError(f"hf_strategy must be one of {self.VALID_HF_STRATEGIES}")
        self.alpha = float(alpha)
        self.tau_lo = self.alpha / 2
        self.tau_up = 1.0 - self.alpha / 2
        self.backend = backend
        self.mode = mode
        self.transfer_target = transfer_target
        self.hf_strategy = hf_strategy
        self.random_state = int(random_state)
        self.n_jobs = n_jobs
        self.rf_mean_min_leaf = int(rf_mean_min_leaf)
        self.rf_crossfit_min_leaf = int(rf_crossfit_min_leaf)
        self.rf_cdf_kwargs = DEFAULT_RF_CDF_KWARGS.copy()
        self.augment_with_quantiles = augment_with_quantiles
        self.augment_quantile_levels = tuple(augment_quantile_levels)
        self.augment_include_mean = augment_include_mean
        if rf_cdf_kwargs is not None:
            self.rf_cdf_kwargs.update(rf_cdf_kwargs)
        self.hf_cdf_kwargs = None
        if hf_cdf_kwargs is not None:
            self.hf_cdf_kwargs = DEFAULT_RF_CDF_KWARGS.copy()
            self.hf_cdf_kwargs.update(hf_cdf_kwargs)
        self.smoother_cdf_kwargs = None
        if smoother_cdf_kwargs is not None:
            self.smoother_cdf_kwargs = DEFAULT_RF_CDF_KWARGS.copy()
            self.smoother_cdf_kwargs.update(smoother_cdf_kwargs)
        self.gp_kwargs = DEFAULT_GP_KWARGS.copy()
        if gp_kwargs is not None:
            self.gp_kwargs.update(gp_kwargs)
        self.hf_gp_kwargs = None
        if hf_gp_kwargs is not None:
            self.hf_gp_kwargs = DEFAULT_GP_KWARGS.copy()
            self.hf_gp_kwargs.update(hf_gp_kwargs)
        self.smoother_gp_kwargs = None
        if smoother_gp_kwargs is not None:
            self.smoother_gp_kwargs = DEFAULT_GP_KWARGS.copy()
            self.smoother_gp_kwargs.update(smoother_gp_kwargs)
        self.probit_transform = probit_transform
        self.stage2_fixed_rho = stage2_fixed_rho
        self.stage2_rho_grid = stage2_rho_grid
        self.stage2_cv_folds = int(stage2_cv_folds)
        self.one_step_correction = bool(one_step_correction)
        self.one_step_crossfit = bool(one_step_crossfit)
        self.one_step_cf_folds = int(one_step_cf_folds)
        self.one_step_gamma = float(one_step_gamma)
        self.one_step_tune_gamma = bool(one_step_tune_gamma)
        if one_step_gamma_grid is None:
            self.one_step_gamma_grid = [self.one_step_gamma]
        else:
            self.one_step_gamma_grid = [float(g) for g in one_step_gamma_grid]
            if len(self.one_step_gamma_grid) == 0:
                self.one_step_gamma_grid = [self.one_step_gamma]
        self.one_step_density_source = str(one_step_density_source)
        if self.one_step_density_source not in {"lf", "hf"}:
            raise ValueError("one_step_density_source must be 'lf' or 'hf'.")
        self.one_step_density_floor = float(one_step_density_floor)
        self.one_step_max_update = None if one_step_max_update is None else float(one_step_max_update)
        self.one_step_space = str(one_step_space)
        if self.one_step_space not in {"residual", "raw"}:
            raise ValueError("one_step_space must be 'residual' or 'raw'.")
        self.seed_ = self.random_state

    @staticmethod
    def _clip_u(u, eps=1e-3):
        return np.clip(np.asarray(u), eps, 1 - eps)

    def _u_to_z(self, u):
        """Apply probit transform: U ∈ (0,1) → Z ∈ ℝ via Φ⁻¹(u)."""
        if not self.probit_transform:
            return u
        from scipy.stats import norm
        return norm.ppf(self._clip_u(u))

    def _z_to_u(self, z):
        """Inverse probit transform: Z ∈ ℝ → U ∈ (0,1) via Φ(z)."""
        if not self.probit_transform:
            return z
        from scipy.stats import norm
        return norm.cdf(z)

    @staticmethod
    def _tau_matrix(n, tau_lo, tau_up):
        return np.repeat(np.array([[tau_lo, tau_up]]), repeats=n, axis=0)

    @staticmethod
    def _with_intercept(x):
        x = np.asarray(x).reshape(-1)
        return np.column_stack([np.ones(len(x)), x])

    @staticmethod
    def _apply_link(x, ab):
        a, b = ab
        return a + b * np.asarray(x)

    @staticmethod
    def _invert_link(y, ab):
        a, b = ab
        return (np.asarray(y) - a) / b

    def _lf_quantile_summary_features(self, X, cdf_model, include_mean=True):
        feats = []

        if include_mean:
            mean_hat = self.lf_mean_model_.predict(X)
            feats.append(mean_hat.reshape(-1, 1))

        if self.augment_with_quantiles:
            taus = np.tile(np.array([self.augment_quantile_levels], dtype=float), (len(X), 1))
            q = cdf_model.quantile_varying_multi(X, taus)
            feats.append(q)

            if q.shape[1] >= 2:
                spread = (q[:, -1] - q[:, 0]).reshape(-1, 1)
                feats.append(spread)

        if not feats:
            raise ValueError("augment produced no features; enable mean and/or quantile summaries.")

        return np.column_stack(feats)



    def _stage2_features(self, X, cdf_model=None):
        X = np.asarray(X)

        if self.hf_strategy != "augment":
            return X

        if cdf_model is None:
            raise ValueError("cdf_model must be provided when hf_strategy='augment'.")

        lf_aug = self._lf_quantile_summary_features(
            X,
            cdf_model=cdf_model,
            include_mean=self.augment_include_mean,
        )
        return np.column_stack([X,lf_aug])



    def _stabilize_link(self, a, b, label):
        b = float(b)
        if not np.isfinite(b) or abs(b) < 1e-8:
            print(f"    Warning: {label} slope nearly zero; using slope=1.")
            b = 1.0
        elif b < 0:
            print(f"    Warning: {label} slope negative ({b:.6f}); using abs(slope).")
            b = abs(b)
        return np.array([float(a), b])

    def _fit_link(self, x, y, label):
        raw_ab = np.linalg.lstsq(self._with_intercept(x), y, rcond=None)[0]
        ab = self._stabilize_link(raw_ab[0], raw_ab[1], label)
        a, b = ab
        print(f"    {label} fitted: a={a:.6f}, b={b:.6f}")
        return ab

    def _make_mean_model(self, crossfit=False):
        if self.backend == "rf":
            min_leaf = self.rf_crossfit_min_leaf if crossfit else self.rf_mean_min_leaf
            return RandomForestRegressor(
                min_samples_leaf=min_leaf,
                random_state=self.seed_,
                n_jobs=self.n_jobs,
            )
        return RFFMeanRegressor(random_state=self.seed_, gp_kwargs=self.gp_kwargs)

    def _make_conditional_cdf(self):
        """CDF builder for LF models (outcome CDF, residual CDF)."""
        if self.backend == "rf":
            return ConditionalCDF_QRF(random_state=self.seed_, **self.rf_cdf_kwargs)
        return ConditionalCDF_RFFGP(random_state=self.seed_, **self.gp_kwargs)

    def _make_hf_cdf(self):
        """CDF builder for HF models (HF-Only, HF offset, 1-step correction)."""
        if self.backend == "rf":
            kwargs = self.hf_cdf_kwargs if self.hf_cdf_kwargs is not None else self.rf_cdf_kwargs
            return ConditionalCDF_QRF(random_state=self.seed_, **kwargs)
        gp_kw = self.hf_gp_kwargs if self.hf_gp_kwargs is not None else self.gp_kwargs
        return ConditionalCDF_RFFGP(random_state=self.seed_, **gp_kw)

    def _make_smoother_cdf(self):
        """CDF builder for the U stage-2 smoother."""
        if self.backend == "rf":
            kwargs = self.smoother_cdf_kwargs if self.smoother_cdf_kwargs is not None else self.rf_cdf_kwargs
            return ConditionalCDF_QRF(random_state=self.seed_, **kwargs)
        gp_kw = self.smoother_gp_kwargs if self.smoother_gp_kwargs is not None else self.gp_kwargs
        return ConditionalCDF_RFFGP(random_state=self.seed_, **gp_kw)

    def _make_stage2_smoother(self, tau):
        return PosteriorQuantileSmoother(
            tau=tau,
            cdf_builder=self._make_smoother_cdf,
            random_state=self.seed_,
            n_cv_folds=self.stage2_cv_folds,
            rho_grid=self.stage2_rho_grid,
            fixed_rho=self.stage2_fixed_rho,
            report_param=True,
            quiet_base_fit=True,
        )

    def _get_fit_cache(self, data):
        if "_fit_cache" not in data or data["_fit_cache"] is None:
            data["_fit_cache"] = {}
        return data["_fit_cache"]

    def _transfer_pilot_cache_key(self, kind):
        return (
            "transfer_pilot",
            kind,
            self.backend,
            self.seed_,
            self.hf_strategy,
            self.transfer_target,
            self.rf_mean_min_leaf,
            self.rf_crossfit_min_leaf,
            _freeze_for_key(self.rf_cdf_kwargs),
            _freeze_for_key(self.smoother_cdf_kwargs),
            _freeze_for_key(self.gp_kwargs),
            _freeze_for_key(self.smoother_gp_kwargs),
            self.stage2_fixed_rho,
            _freeze_for_key(self.stage2_rho_grid),
            self.stage2_cv_folds,
            self.augment_with_quantiles,
            _freeze_for_key(self.augment_quantile_levels),
            self.augment_include_mean,
        )

    def _hf_method_cache_key(self):
        return (
            "hf_method",
            self.backend,
            self.seed_,
            self.hf_strategy,
            self.rf_mean_min_leaf,
            self.rf_crossfit_min_leaf,
            _freeze_for_key(self.rf_cdf_kwargs),
            _freeze_for_key(self.hf_cdf_kwargs),
            _freeze_for_key(self.gp_kwargs),
            _freeze_for_key(self.hf_gp_kwargs),
            self.augment_with_quantiles,
            _freeze_for_key(self.augment_quantile_levels),
            self.augment_include_mean,
        )

    def _hf_correction_cache_key(self):
        return (
            "hf_correction_cdf",
            self.backend,
            self.seed_,
            self.one_step_space,
            _freeze_for_key(self.hf_cdf_kwargs),
            _freeze_for_key(self.gp_kwargs),
            _freeze_for_key(self.hf_gp_kwargs),
        )

    def _one_step_fold_cache_key(self):
        return (
            "one_step_fold_objects",
            self._transfer_pilot_cache_key(f"{self.transfer_target}:{self.hf_strategy}"),
            self.one_step_cf_folds,
            self.one_step_space,
        )

    @staticmethod
    def _save_state(attrs, obj):
        return {name: getattr(obj, name) for name in attrs if hasattr(obj, name)}

    @staticmethod
    def _load_state(state, obj):
        for name, value in state.items():
            setattr(obj, name, value)


    def _fit_hf_correction_cdf(self, data):
        cache = self._get_fit_cache(data)
        key = self._hf_correction_cache_key()
        if key in cache:
            self.hf_correction_cdf_ = cache[key]
            return
        self.hf_correction_cdf_ = self._make_hf_cdf()
        with contextlib.redirect_stdout(io.StringIO()):
            if (
                self.transfer_target == "residual"
                and self.one_step_space == "residual"
                and hasattr(self, "lf_mean_model_")
            ):
                # Residual-space OS: fit HF correction CDF on mean-subtracted residuals.
                hf_mean_pred = self.lf_mean_model_.predict(data["X_hf"])
                hf_offset = self._apply_link(hf_mean_pred, self.residual_ab_)
                hf_resid = data["Y_hf_std"] - hf_offset
                self.hf_correction_cdf_.fit(data["X_hf"], hf_resid)
            else:
                # Raw-HF-space OS: fit on raw HF responses (one_step_space='raw'
                # path used by §5.3 Misinformative regime).
                self.hf_correction_cdf_.fit(data["X_hf"], data["Y_hf_std"])
        cache[key] = self.hf_correction_cdf_

    @staticmethod
    def _subset_data_hf(data, hf_idx):
        sub = dict(data)
        sub.pop("_fit_cache", None)
        hf_idx = np.asarray(hf_idx, dtype=int)
        for key in ["X_hf", "Y_hf_train", "Y_hf_std", "Y_lf_at_hf"]:
            if key in sub:
                sub[key] = np.asarray(sub[key])[hf_idx]
        return sub

    def _spawn_base_model_no_onestep(self, seed_offset=0):
        return TransferQuantileRegressor(
            alpha=self.alpha,
            backend=self.backend,
            mode=self.mode,
            transfer_target=self.transfer_target,
            hf_strategy=self.hf_strategy,
            random_state=self.seed_ + int(seed_offset),
            n_jobs=self.n_jobs,
            rf_mean_min_leaf=self.rf_mean_min_leaf,
            rf_crossfit_min_leaf=self.rf_crossfit_min_leaf,
            rf_cdf_kwargs=self.rf_cdf_kwargs.copy(),
            hf_cdf_kwargs=self.hf_cdf_kwargs.copy() if self.hf_cdf_kwargs else None,
            smoother_cdf_kwargs=self.smoother_cdf_kwargs.copy() if self.smoother_cdf_kwargs else None,
            gp_kwargs=self.gp_kwargs.copy(),
            hf_gp_kwargs=self.hf_gp_kwargs.copy() if self.hf_gp_kwargs else None,
            smoother_gp_kwargs=self.smoother_gp_kwargs.copy() if self.smoother_gp_kwargs else None,
            stage2_fixed_rho=self.stage2_fixed_rho,
            stage2_rho_grid=self.stage2_rho_grid,
            stage2_cv_folds=self.stage2_cv_folds,
            augment_with_quantiles=self.augment_with_quantiles,
            augment_quantile_levels=self.augment_quantile_levels,
            augment_include_mean=self.augment_include_mean,
            one_step_correction=False,
            one_step_crossfit=False,
            one_step_cf_folds=self.one_step_cf_folds,
            one_step_gamma=self.one_step_gamma,
            one_step_tune_gamma=False,
            one_step_gamma_grid=[self.one_step_gamma],
            one_step_density_source=self.one_step_density_source,
            one_step_density_floor=self.one_step_density_floor,
            one_step_max_update=self.one_step_max_update,
            one_step_space=self.one_step_space,
            probit_transform=self.probit_transform,
        )


    @staticmethod
    def _pinball_loss_tau(y_true, y_pred, tau):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        err = y_true - y_pred
        return np.mean(np.maximum(tau * err, (tau - 1.0) * err))

    def _prepare_one_step_fold_objects(self, data):
        cache = self._get_fit_cache(data)
        key = self._one_step_fold_cache_key()
        if key in cache:
            return cache[key]
        n_hf = len(data["X_hf"])
        n_splits = min(max(2, self.one_step_cf_folds), n_hf)
        if n_splits < 2 or n_hf < 2:
            cache[key] = []
            return cache[key]
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed_)
        fold_objects = []
        use_residual_correction = (
            self.transfer_target == "residual"
            and self.one_step_space == "residual"
        )
        for fold_id, (tr, va) in enumerate(kf.split(data["X_hf"])):
            fold_data = self._subset_data_hf(data, tr)
            base = self._spawn_base_model_no_onestep(seed_offset=5000 + fold_id)
            with contextlib.redirect_stdout(io.StringIO()):
                base.fit(fold_data, seed=self.seed_ + 5000 + fold_id)
            hf_cdf = base._make_hf_cdf()
            with contextlib.redirect_stdout(io.StringIO()):
                if use_residual_correction and hasattr(self, "lf_mean_model_"):
                    # Fit CDF on HF residuals (mean-subtracted)
                    hf_mean_pred = self.lf_mean_model_.predict(fold_data["X_hf"])
                    hf_offset = self._apply_link(hf_mean_pred, self.residual_ab_)
                    hf_resid = fold_data["Y_hf_std"] - hf_offset
                    hf_cdf.fit(fold_data["X_hf"], hf_resid)
                else:
                    hf_cdf.fit(fold_data["X_hf"], fold_data["Y_hf_std"])
            fold_objects.append({
                "base": base,
                "hf_cdf": hf_cdf,
                "va": np.asarray(va, dtype=int),
                "use_residual": use_residual_correction,
            })
        cache[key] = fold_objects
        return fold_objects

    def _one_step_gamma_eval_cache_key(self):
        return (
            "one_step_gamma_eval",
            self._one_step_fold_cache_key(),
            self.one_step_density_source,
            float(self.one_step_density_floor),
            None if self.one_step_max_update is None else float(self.one_step_max_update),
        )

    def _one_step_apply_gamma_from_raw(self, q0, raw_delta, gamma):
        q0 = np.asarray(q0).reshape(-1)
        raw_delta = np.asarray(raw_delta).reshape(-1)
        step = float(gamma) * raw_delta
        if self.one_step_max_update is not None:
            step = np.clip(step, -self.one_step_max_update, self.one_step_max_update)
        return q0 - step

    def _prepare_one_step_gamma_eval(self, data):
        cache = self._get_fit_cache(data)
        key = self._one_step_gamma_eval_cache_key()
        if key in cache:
            return cache[key]
        fold_objects = self._prepare_one_step_fold_objects(data)
        eval_objects = []
        for obj in fold_objects:
            va = obj["va"]
            base = obj["base"]
            hf_cdf = obj["hf_cdf"]
            X_va = np.asarray(data["X_hf"])[va]
            y_va = np.asarray(data["Y_hf_std"])[va]
            if self.transfer_target == "outcome":
                info = base._transfer_outcome_pilot_info(X_va)
            else:
                info = base._transfer_residual_pilot_info(X_va)

            q0_lo = np.asarray(info["q0_lo"]).reshape(-1)
            q0_up = np.asarray(info["q0_up"]).reshape(-1)

            # If using residual correction, convert pilot to residual space
            use_residual = obj.get("use_residual", False)
            if use_residual and hasattr(self, "lf_mean_model_"):
                mean_va = self.lf_mean_model_.predict(X_va)
                offset_va = self._apply_link(mean_va, self.residual_ab_)
                r0_lo = q0_lo - offset_va
                r0_up = q0_up - offset_va
            else:
                r0_lo = q0_lo
                r0_up = q0_up

            numer_lo = hf_cdf.cdf_batch(X_va, r0_lo) - float(self.tau_lo)
            numer_up = hf_cdf.cdf_batch(X_va, r0_up) - float(self.tau_up)

            if self.one_step_density_source == "hf":
                denom_lo = hf_cdf.density_batch(X_va, r0_lo)
                denom_up = hf_cdf.density_batch(X_va, r0_up)
            else:
                link_slope = max(float(info["link_slope"]), 1e-8)
                denom_lo = np.asarray(info["lf_dens_lo"]).reshape(-1) / link_slope
                denom_up = np.asarray(info["lf_dens_up"]).reshape(-1) / link_slope

            denom_lo = np.maximum(np.asarray(denom_lo).reshape(-1), self.one_step_density_floor)
            denom_up = np.maximum(np.asarray(denom_up).reshape(-1), self.one_step_density_floor)

            eval_objects.append({
                "y_va": y_va,
                "q0_lo": q0_lo,
                "q0_up": q0_up,
                "raw_delta_lo": numer_lo / denom_lo,
                "raw_delta_up": numer_up / denom_up,
            })
        cache[key] = eval_objects
        return eval_objects

    def _one_step_cv_loss_for_gamma(self, data, gamma):
        eval_objects = self._prepare_one_step_gamma_eval(data)
        if len(eval_objects) == 0:
            return np.inf
        fold_losses = []
        for obj in eval_objects:
            y_va = obj["y_va"]
            lo = self._one_step_apply_gamma_from_raw(obj["q0_lo"], obj["raw_delta_lo"], gamma)
            up = self._one_step_apply_gamma_from_raw(obj["q0_up"], obj["raw_delta_up"], gamma)
            loss = self._pinball_loss_tau(y_va, lo, self.tau_lo) + self._pinball_loss_tau(y_va, up, self.tau_up)
            fold_losses.append(loss)
        return float(np.mean(fold_losses))

    def _tune_one_step_gamma(self, data):
        grid = [float(g) for g in self.one_step_gamma_grid]
        if len(grid) == 1:
            self.one_step_gamma_ = grid[0]
            self.one_step_gamma_cv_loss_ = np.nan
            self.one_step_gamma = grid[0]
            return grid[0], np.nan
        best_gamma = grid[0]
        best_loss = np.inf
        for gamma in grid:
            loss = self._one_step_cv_loss_for_gamma(data, gamma)
            if loss < best_loss:
                best_loss = loss
                best_gamma = gamma
        self.one_step_gamma_ = best_gamma
        self.one_step_gamma_cv_loss_ = float(best_loss)
        self.one_step_gamma = float(best_gamma)
        print(f"    One-step gamma tuned by CV: gamma={best_gamma:.3f} (CV pinball={best_loss:.6f})")
        return best_gamma, best_loss

    def _fit_one_step_crossfit_models(self, data):
        fold_objects = self._prepare_one_step_fold_objects(data)
        self.one_step_cf_models_ = [{"base": obj["base"], "hf_cdf": obj["hf_cdf"], "use_residual": obj.get("use_residual", False)} for obj in fold_objects]

    def _one_step_update(self, X_new, q0_hf, tau, lf_density=None, link_slope=1.0, hf_cdf_model=None, gamma=None):
        q0_hf = np.asarray(q0_hf).reshape(-1)
        hf_model = self.hf_correction_cdf_ if hf_cdf_model is None else hf_cdf_model
        numer = hf_model.cdf_batch(X_new, q0_hf) - float(tau)

        if self.one_step_density_source == "hf" or lf_density is None:
            denom = hf_model.density_batch(X_new, q0_hf)
        else:
            denom = np.asarray(lf_density).reshape(-1) / max(float(link_slope), 1e-8)

        denom = np.maximum(denom, self.one_step_density_floor)
        gamma_use = self.one_step_gamma if gamma is None else float(gamma)
        step = gamma_use * numer / denom
        if self.one_step_max_update is not None:
            step = np.clip(step, -self.one_step_max_update, self.one_step_max_update)
        q1 = q0_hf - step
        return q1

    def _transfer_outcome_pilot_info(self, X_new, model=None):
        m = self if model is None else model
        X_stage2 = m._stage2_features(X_new, cdf_model=m.lf_outcome_cdf_)
        u_lo = m._clip_u(m._z_to_u(m.u_lo_model_.predict(X_stage2)))
        u_up = m._clip_u(m._z_to_u(m.u_up_model_.predict(X_stage2)))

        tau = np.column_stack([u_lo, u_up])
        q_lf = m.lf_outcome_cdf_.quantile_varying_multi(X_new, tau)
        q0_lo = m._apply_link(q_lf[:, 0], m.outcome_ab_)
        q0_up = m._apply_link(q_lf[:, 1], m.outcome_ab_)
        dens_lo = m.lf_outcome_cdf_.density_batch(X_new, q_lf[:, 0])
        dens_up = m.lf_outcome_cdf_.density_batch(X_new, q_lf[:, 1])
        return {
            "q0_lo": q0_lo,
            "q0_up": q0_up,
            "lf_dens_lo": dens_lo,
            "lf_dens_up": dens_up,
            "link_slope": m.outcome_ab_[1],
        }

    def _transfer_residual_pilot_info(self, X_new, model=None):
        m = self if model is None else model
        if m.hf_strategy == "augment":
            X_stage2 = m._stage2_features(X_new, cdf_model=m.lf_outcome_cdf_aux_)
        else:
            X_stage2 = X_new

        u_lo = m._clip_u(m._z_to_u(m.u_lo_model_.predict(X_stage2)))
        u_up = m._clip_u(m._z_to_u(m.u_up_model_.predict(X_stage2)))

        tau = np.column_stack([u_lo, u_up])
        lf_mean = m.lf_mean_model_.predict(X_new)
        q_resid = m.lf_resid_cdf_.quantile_varying_multi(X_new, tau)
        y_lf_lo = lf_mean + q_resid[:, 0]
        y_lf_up = lf_mean + q_resid[:, 1]
        q0_lo = m._apply_link(y_lf_lo, m.residual_ab_)
        q0_up = m._apply_link(y_lf_up, m.residual_ab_)
        dens_lo = m.lf_resid_cdf_.density_batch(X_new, q_resid[:, 0])
        dens_up = m.lf_resid_cdf_.density_batch(X_new, q_resid[:, 1])
        return {
            "q0_lo": q0_lo,
            "q0_up": q0_up,
            "lf_dens_lo": dens_lo,
            "lf_dens_up": dens_up,
            "link_slope": m.residual_ab_[1],
        }

    def _cross_fit_residuals(self, X, y, n_folds=5):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed_)
        residual = np.zeros(len(y), dtype=float)
        label = "RF" if self.backend == "rf" else "GP"
        print(f"    Cross-fitting {label} mean ({n_folds} folds)...")
        for tr, va in kf.split(X):
            model = self._make_mean_model(crossfit=True)
            model.fit(X[tr], y[tr])
            residual[va] = y[va] - model.predict(X[va])
        print(f"    Cross-fitted residuals: mean={residual.mean():.6f} std={residual.std():.6f}")
        return residual

    def fit(self, data, seed=None):
        self.seed_ = self.random_state if seed is None else int(seed)
        self._get_fit_cache(data)

        # Always define these
        self.offset_ab_ = np.array([0.0, 1.0])
        self.outcome_ab_ = np.array([0.0, 1.0])
        self.residual_ab_ = np.array([0.0, 1.0])

        if self.mode == "hf":
            return self._fit_hf(data)
        if self.transfer_target == "outcome":
            return self._fit_transfer_outcome(data)
        return self._fit_transfer_residual(data)


    def _fit_hf(self, data):
        cache = self._get_fit_cache(data)
        key = self._hf_method_cache_key()
        if key in cache:
            self._load_state(cache[key], self)
            return self

        self.hf_cdf_ = self._make_hf_cdf()
        self.offset_ab_ = np.array([0.0, 1.0])
        attrs = ["hf_cdf_", "offset_ab_"]
        if self.hf_strategy == "raw":
            self.hf_cdf_.fit(data["X_hf"], data["Y_hf_std"])
            cache[key] = self._save_state(attrs, self)
            return self

        self.lf_mean_model_ = self._make_mean_model(crossfit=False)
        self.lf_mean_model_.fit(data["X_lf"], data["Y_lf_std"])
        attrs.append("lf_mean_model_")
        lf_mean_on_hf = self.lf_mean_model_.predict(data["X_hf"])
        if self.hf_strategy == "augment":
            self.lf_outcome_cdf_ = self._make_conditional_cdf()
            self.lf_outcome_cdf_.fit(data["X_lf"], data["Y_lf_std"])
            attrs.append("lf_outcome_cdf_")

            X_aug = self._stage2_features(data["X_hf"], cdf_model=self.lf_outcome_cdf_)
            self.hf_cdf_.fit(X_aug, data["Y_hf_std"])
            cache[key] = self._save_state(attrs, self)
            return self
        self.offset_ab_ = self._fit_link(lf_mean_on_hf, data["Y_hf_std"], "HF offset link")
        hf_offset = self._apply_link(lf_mean_on_hf, self.offset_ab_)
        self.hf_cdf_.fit(data["X_hf"], data["Y_hf_std"] - hf_offset)
        cache[key] = self._save_state(attrs, self)
        return self

    def _fit_transfer_outcome(self, data):
        cache = self._get_fit_cache(data)
        key = self._transfer_pilot_cache_key(f"outcome:{self.hf_strategy}")
        if key in cache:
            self._load_state(cache[key], self)
        else:
            self.lf_mean_model_ = self._make_mean_model(crossfit=False)
            self.lf_mean_model_.fit(data["X_lf"], data["Y_lf_std"])

            self.lf_outcome_cdf_ = self._make_conditional_cdf()
            self.lf_outcome_cdf_.fit(data["X_lf"], data["Y_lf_std"])

            lf_mean_on_hf = self.lf_mean_model_.predict(data["X_hf"])
            self.outcome_ab_ = self._fit_link(lf_mean_on_hf, data["Y_hf_std"], "Outcome link")

            hf_aligned_to_lf = self._invert_link(data["Y_hf_std"], self.outcome_ab_)
            u_hf = self._clip_u(self.lf_outcome_cdf_.cdf_batch(data["X_hf"], hf_aligned_to_lf))
            z_hf = self._u_to_z(u_hf)

            X_stage2 = self._stage2_features(data["X_hf"], cdf_model=self.lf_outcome_cdf_)
            self.u_lo_model_ = self._make_stage2_smoother(self.tau_lo).fit(X_stage2, z_hf)
            self.u_up_model_ = self._make_stage2_smoother(self.tau_up).fit(X_stage2, z_hf)
            cache[key] = self._save_state(["lf_mean_model_", "lf_outcome_cdf_", "outcome_ab_", "u_lo_model_", "u_up_model_"], self)
        if self.one_step_correction:
            self._fit_hf_correction_cdf(data)
            if self.one_step_tune_gamma:
                self._tune_one_step_gamma(data)
            else:
                self.one_step_gamma_ = self.one_step_gamma
                self.one_step_gamma_cv_loss_ = np.nan
            if self.one_step_crossfit:
                self._fit_one_step_crossfit_models(data)
            print(
                f"    One-step correction enabled: gamma={self.one_step_gamma:.3f}, "
                f"density_source={self.one_step_density_source}, "
                f"density_floor={self.one_step_density_floor:.4g}, "
                f"crossfit={'on' if self.one_step_crossfit else 'off'}"
            )
        return self

    def _fit_transfer_residual(self, data):
        cache = self._get_fit_cache(data)
        key = self._transfer_pilot_cache_key(f"residual:{self.hf_strategy}")
        if key in cache:
            self._load_state(cache[key], self)
        else:
            lf_resid_crossfit = self._cross_fit_residuals(data["X_lf"], data["Y_lf_std"], n_folds=5)

            self.lf_resid_cdf_ = self._make_conditional_cdf()
            self.lf_resid_cdf_.fit(data["X_lf"], lf_resid_crossfit)

            self.lf_mean_model_ = self._make_mean_model(crossfit=False)
            self.lf_mean_model_.fit(data["X_lf"], data["Y_lf_std"])

            lf_mean_on_hf = self.lf_mean_model_.predict(data["X_hf"])
            self.residual_ab_ = self._fit_link(lf_mean_on_hf, data["Y_hf_std"], "Residual link")

            hf_aligned_to_lf = self._invert_link(data["Y_hf_std"], self.residual_ab_)
            hf_resid_on_lf_scale = hf_aligned_to_lf - lf_mean_on_hf
            u_hf = self._clip_u(self.lf_resid_cdf_.cdf_batch(data["X_hf"], hf_resid_on_lf_scale))
            z_hf = self._u_to_z(u_hf)

            if self.hf_strategy == "augment":
                self.lf_outcome_cdf_aux_ = self._make_conditional_cdf()
                self.lf_outcome_cdf_aux_.fit(data["X_lf"], data["Y_lf_std"])
                X_stage2 = self._stage2_features(data["X_hf"], cdf_model=self.lf_outcome_cdf_aux_)
            else:
                X_stage2 = data["X_hf"]

            self.u_lo_model_ = self._make_stage2_smoother(self.tau_lo).fit(X_stage2, z_hf)
            self.u_up_model_ = self._make_stage2_smoother(self.tau_up).fit(X_stage2, z_hf)
            cache[key] = self._save_state(["lf_resid_cdf_", "lf_mean_model_", "residual_ab_", "u_lo_model_", "u_up_model_", "lf_outcome_cdf_aux_"], self)
        if self.one_step_correction:
            self._fit_hf_correction_cdf(data)
            if self.one_step_tune_gamma:
                self._tune_one_step_gamma(data)
            else:
                self.one_step_gamma_ = self.one_step_gamma
                self.one_step_gamma_cv_loss_ = np.nan
            if self.one_step_crossfit:
                self._fit_one_step_crossfit_models(data)
            print(
                f"    One-step correction enabled: gamma={self.one_step_gamma:.3f}, "
                f"density_source={self.one_step_density_source}, "
                f"density_floor={self.one_step_density_floor:.4g}, "
                f"crossfit={'on' if self.one_step_crossfit else 'off'}"
            )
        return self

    def predict_std(self, X_new):
        if self.mode == "hf":
            return self._predict_hf(X_new)
        if self.transfer_target == "outcome":
            return self._predict_transfer_outcome(X_new)
        return self._predict_transfer_residual(X_new)

    def _predict_hf(self, X_new):
        tau = self._tau_matrix(len(X_new), self.tau_lo, self.tau_up)

        if self.hf_strategy == "augment":
            X_use = self._stage2_features(X_new, cdf_model=self.lf_outcome_cdf_)
        else:
            X_use = X_new

        q = self.hf_cdf_.quantile_varying_multi(X_use, tau)
        lo, up = q[:, 0], q[:, 1]

        if self.hf_strategy == "offset":
            lf_mean = self.lf_mean_model_.predict(X_new)
            hf_offset = self._apply_link(lf_mean, self.offset_ab_)
            lo = hf_offset + lo
            up = hf_offset + up

        return lo, up

    def _predict_transfer_outcome(self, X_new):
        if self.one_step_correction and self.one_step_crossfit and getattr(self, "one_step_cf_models_", None):
            lo_list, up_list = [], []
            for obj in self.one_step_cf_models_:
                info = self._transfer_outcome_pilot_info(X_new, model=obj["base"])
                lo_k = self._one_step_update(
                    X_new,
                    info["q0_lo"],
                    self.tau_lo,
                    lf_density=info["lf_dens_lo"],
                    link_slope=info["link_slope"],
                    hf_cdf_model=obj["hf_cdf"],
                )
                up_k = self._one_step_update(
                    X_new,
                    info["q0_up"],
                    self.tau_up,
                    lf_density=info["lf_dens_up"],
                    link_slope=info["link_slope"],
                    hf_cdf_model=obj["hf_cdf"],
                )
                lo_list.append(lo_k)
                up_list.append(up_k)
            lo = np.mean(np.vstack(lo_list), axis=0)
            up = np.mean(np.vstack(up_list), axis=0)
        else:
            info = self._transfer_outcome_pilot_info(X_new)
            lo = info["q0_lo"]
            up = info["q0_up"]
            if self.one_step_correction:
                lo = self._one_step_update(
                    X_new,
                    lo,
                    self.tau_lo,
                    lf_density=info["lf_dens_lo"],
                    link_slope=info["link_slope"],
                )
                up = self._one_step_update(
                    X_new,
                    up,
                    self.tau_up,
                    lf_density=info["lf_dens_up"],
                    link_slope=info["link_slope"],
                )

        lo_final = np.minimum(lo, up)
        up_final = np.maximum(lo, up)
        return lo_final, up_final

    def _predict_transfer_residual(self, X_new):
        if self.one_step_correction and self.one_step_crossfit and getattr(self, "one_step_cf_models_", None):
            lo_list, up_list = [], []
            for obj in self.one_step_cf_models_:
                info = self._transfer_residual_pilot_info(X_new, model=obj["base"])
                use_residual = obj.get("use_residual", False)

                if use_residual and hasattr(self, "lf_mean_model_"):
                    # Convert pilot to residual space for CDF evaluation
                    mean_new = self.lf_mean_model_.predict(X_new)
                    offset_new = self._apply_link(mean_new, self.residual_ab_)
                    r0_lo = info["q0_lo"] - offset_new
                    r0_up = info["q0_up"] - offset_new
                    # Correction in residual space
                    lo_k = self._one_step_update(
                        X_new, r0_lo, self.tau_lo,
                        lf_density=info["lf_dens_lo"],
                        link_slope=info["link_slope"],
                        hf_cdf_model=obj["hf_cdf"],
                    )
                    up_k = self._one_step_update(
                        X_new, r0_up, self.tau_up,
                        lf_density=info["lf_dens_up"],
                        link_slope=info["link_slope"],
                        hf_cdf_model=obj["hf_cdf"],
                    )
                    # Convert correction back: the update subtracted from r0,
                    # add offset back to get Y space
                    lo_k = lo_k + offset_new
                    up_k = up_k + offset_new
                else:
                    lo_k = self._one_step_update(
                        X_new,
                        info["q0_lo"],
                        self.tau_lo,
                        lf_density=info["lf_dens_lo"],
                        link_slope=info["link_slope"],
                        hf_cdf_model=obj["hf_cdf"],
                    )
                    up_k = self._one_step_update(
                        X_new,
                        info["q0_up"],
                        self.tau_up,
                        lf_density=info["lf_dens_up"],
                        link_slope=info["link_slope"],
                        hf_cdf_model=obj["hf_cdf"],
                    )
                lo_list.append(lo_k)
                up_list.append(up_k)
            lo = np.mean(np.vstack(lo_list), axis=0)
            up = np.mean(np.vstack(up_list), axis=0)
        else:
            info = self._transfer_residual_pilot_info(X_new)
            lo = info["q0_lo"]
            up = info["q0_up"]
            if self.one_step_correction:
                if (
                    hasattr(self, "lf_mean_model_")
                    and self.one_step_space == "residual"
                ):
                    mean_new = self.lf_mean_model_.predict(X_new)
                    offset_new = self._apply_link(mean_new, self.residual_ab_)
                    r0_lo = lo - offset_new
                    r0_up = up - offset_new
                    lo = self._one_step_update(
                        X_new, r0_lo, self.tau_lo,
                        lf_density=info["lf_dens_lo"],
                        link_slope=info["link_slope"],
                    ) + offset_new
                    up = self._one_step_update(
                        X_new, r0_up, self.tau_up,
                        lf_density=info["lf_dens_up"],
                        link_slope=info["link_slope"],
                    ) + offset_new
                else:
                    # Raw-HF-space update (one_step_space='raw' path).
                    lo = self._one_step_update(
                        X_new, lo, self.tau_lo,
                        lf_density=info["lf_dens_lo"],
                        link_slope=info["link_slope"],
                    )
                    up = self._one_step_update(
                        X_new, up, self.tau_up,
                        lf_density=info["lf_dens_up"],
                        link_slope=info["link_slope"],
                    )

        lo_final = np.minimum(lo, up)
        up_final = np.maximum(lo, up)
        return lo_final, up_final




def build_methods(
    alpha,
    backend,
    random_state=42,
    gp_kwargs=None,
    rf_cdf_kwargs=None,
    hf_cdf_kwargs=None,
    smoother_cdf_kwargs=None,
    hf_gp_kwargs=None,
    smoother_gp_kwargs=None,
    rf_mean_min_leaf=10,
    rf_crossfit_min_leaf=10,
    stage2_fixed_rho=0.0,
    stage2_rho_grid=None,
    stage2_cv_folds=5,
    one_step_crossfit=False,
    one_step_cf_folds=5,
    one_step_gamma=1.0,
    one_step_tune_gamma=False,
    one_step_gamma_grid=None,
    one_step_density_source="hf",
    one_step_density_floor=0.1,
    one_step_max_update=1.0,
    one_step_space="residual",
    probit_transform=False,
):
    methods = {}
    for name, spec in METHOD_SPECS.items():
        kwargs = dict(
            alpha=alpha,
            backend=backend,
            random_state=random_state,
            gp_kwargs=gp_kwargs,
            rf_cdf_kwargs=rf_cdf_kwargs,
            hf_cdf_kwargs=hf_cdf_kwargs,
            smoother_cdf_kwargs=smoother_cdf_kwargs,
            hf_gp_kwargs=hf_gp_kwargs,
            smoother_gp_kwargs=smoother_gp_kwargs,
            rf_mean_min_leaf=rf_mean_min_leaf,
            rf_crossfit_min_leaf=rf_crossfit_min_leaf,
            stage2_fixed_rho=stage2_fixed_rho,
            stage2_rho_grid=stage2_rho_grid,
            stage2_cv_folds=stage2_cv_folds,
            one_step_crossfit=one_step_crossfit,
            one_step_cf_folds=one_step_cf_folds,
            one_step_gamma=one_step_gamma,
            one_step_tune_gamma=one_step_tune_gamma,
            one_step_gamma_grid=one_step_gamma_grid,
            one_step_density_source=one_step_density_source,
            one_step_density_floor=one_step_density_floor,
            one_step_max_update=one_step_max_update,
            one_step_space=one_step_space,
            probit_transform=probit_transform,
        )
        kwargs.update(spec)
        methods[name] = TransferQuantileRegressor(**kwargs)
    return methods


def run_one(method, data, alpha):
    method.fit(data)
    lo, up = method.predict_std(data["X_test"])
    y_test = data["Y_hf_test_std"]
    raw_cov = np.mean((y_test >= lo) & (y_test <= up))
    raw_wid = np.mean(up - lo)
    lo_c, up_c = method.predict_std(data["X_cal"])
    q = conformal_calibrate(lo_c, up_c, data["Y_hf_cal_std"], alpha)
    cqr_cov = np.mean((y_test >= lo - q) & (y_test <= up + q))
    cqr_wid = np.mean((up + q) - (lo - q))
    return raw_cov, raw_wid, cqr_cov, cqr_wid
