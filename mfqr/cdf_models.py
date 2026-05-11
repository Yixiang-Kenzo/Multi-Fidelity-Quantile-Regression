import numpy as np
from scipy.stats import norm as scipy_norm
from sklearn.ensemble import RandomForestRegressor

# ridge lambda = noise_var / prior_var = 0.001
DEFAULT_GP_KWARGS = {
    "length_scale": 5,
    "n_features": 1000,
    "noise_var": 0.0001,
    "prior_var": 0.1,
}

DEFAULT_RF_CDF_KWARGS = {
    "n_estimators": 100,
    "min_samples_leaf": 10,
    "criterion": "squared_error",
    "dist_mode": "empirical",
    "split_mode": "raw",
    "moment_order": 1,
}

class RFFGaussianRegressor:
    """Random Fourier Features approximation to an RBF GP via Bayesian linear regression."""

    def __init__(self, length_scale=1.0, n_features=100, noise_var=1e-3, prior_var=1.0, seed=0):
        self.length_scale = float(length_scale)
        self.n_features = int(n_features)
        self.noise_var = float(noise_var)
        self.prior_var = float(prior_var)
        self.seed = int(seed)

    def _init_rff(self, d):
        rng = np.random.RandomState(self.seed)
        self.W_ = rng.normal(0.0, 1.0 / self.length_scale, size=(d, self.n_features))
        self.b_ = rng.uniform(0.0, 2 * np.pi, size=(self.n_features,))

    def _phi(self, X):
        Z = X @ self.W_ + self.b_
        return np.sqrt(2.0 / self.n_features) * np.cos(Z)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, d = X.shape
        self._init_rff(d)

        Phi = self._phi(X)
        lam = 1.0 / self.prior_var
        A = lam * np.eye(self.n_features) + (Phi.T @ Phi) / self.noise_var
        self.L_ = np.linalg.cholesky(A)

        rhs = (Phi.T @ y) / self.noise_var
        v = np.linalg.solve(self.L_, rhs)
        self.m_ = np.linalg.solve(self.L_.T, v)
        return self

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        Phi = self._phi(X)
        mean = Phi @ self.m_

        if not return_std:
            return mean

        V = np.linalg.solve(self.L_, Phi.T)
        var = np.sum(V ** 2, axis=0) + self.noise_var
        std = np.sqrt(np.maximum(var, 1e-12))
        return mean, std


class RFFMeanRegressor:
    """Mean regressor wrapper around RFFGaussianRegressor with internal X/y scaling."""

    def __init__(self, random_state=42, gp_kwargs=None):
        self.random_state = int(random_state)
        self.gp_kwargs = DEFAULT_GP_KWARGS.copy()
        if gp_kwargs is not None:
            self.gp_kwargs.update(gp_kwargs)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.x_mean_ = np.mean(X, axis=0)
        self.x_std_ = np.std(X, axis=0)
        self.x_std_ = np.where(self.x_std_ < 1e-12, 1.0, self.x_std_)
        self.y_mean_ = float(np.mean(y))
        self.y_std_ = float(np.std(y))
        self.y_std_ = self.y_std_ if self.y_std_ > 1e-12 else 1.0

        Xs = (X - self.x_mean_) / self.x_std_
        ys = (y - self.y_mean_) / self.y_std_
        self.model_ = RFFGaussianRegressor(seed=self.random_state, **self.gp_kwargs)
        self.model_.fit(Xs, ys)
        return self

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        Xs = (X - self.x_mean_) / self.x_std_
        if not return_std:
            mean_s = self.model_.predict(Xs, return_std=False)
            return self.y_mean_ + self.y_std_ * mean_s
        mean_s, std_s = self.model_.predict(Xs, return_std=True)
        return self.y_mean_ + self.y_std_ * mean_s, self.y_std_ * std_s


# =============================================================================
# Conditional CDF models
# =============================================================================

class ConditionalCDF_QRF:
    """Conditional CDF / quantile estimator from RF neighborhood weights."""

    def __init__(
        self,
        n_estimators=100,
        min_samples_leaf=5,
        criterion="squared_error",
        dist_mode="empirical",
        random_state=42,
        split_mode="raw",
        moment_order=1,
    ):
        assert dist_mode in {"empirical", "gaussian"}
        assert split_mode in {"raw", "moments"}
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.dist_mode = dist_mode
        self.random_state = random_state
        self.split_mode = split_mode
        self.moment_order = moment_order
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state,
            n_jobs=-1,
        )

    def _build_rf_target(self, R):
        R = np.asarray(R).reshape(-1)
        if self.split_mode == "raw":
            return R
        mu = np.mean(R)
        sd = max(np.std(R), 1e-8)
        z = (R - mu) / sd
        cols = []
        for k in range(1, self.moment_order + 1):
            v = z ** k
            v = v - np.mean(v)
            cols.append(v)
        return np.column_stack(cols)

    def fit(self, X, R, verbose=True):
        X = np.asarray(X)
        R = np.asarray(R).reshape(-1)
        Y_rf = self._build_rf_target(R)
        self.rf.fit(X, Y_rf)
        self.R_train = R.copy()
        self.train_leaves = self.rf.apply(X)
        self.n_train, self.n_trees = self.train_leaves.shape
        self.leaf_members = []
        for t in range(self.n_trees):
            tree_dict = {}
            for i, leaf_id in enumerate(self.train_leaves[:, t]):
                tree_dict.setdefault(leaf_id, []).append(i)
            for key in tree_dict:
                tree_dict[key] = np.array(tree_dict[key], dtype=np.int32)
            self.leaf_members.append(tree_dict)
        self.sorted_idx = np.argsort(self.R_train)
        self.sorted_R = self.R_train[self.sorted_idx]
        if verbose:
            print(
                f"    CDF fitted: {self.n_train} samples, {self.n_trees} trees, "
                f"min_leaf={self.min_samples_leaf}, criterion={self.criterion}, "
                f"split_mode={self.split_mode}, moment_order={self.moment_order}, "
                f"dist_mode={self.dist_mode}, R std={np.std(R):.4f}"
            )
        return self

    def _compute_weights_batch(self, X_new):
        X_new = np.asarray(X_new)
        test_leaves = self.rf.apply(X_new)
        weights = np.zeros((len(X_new), self.n_train), dtype=float)
        for t in range(self.n_trees):
            tree_dict = self.leaf_members[t]
            test_leaves_t = test_leaves[:, t]
            for leaf_id in np.unique(test_leaves_t):
                members = tree_dict.get(leaf_id, None)
                if members is None or len(members) == 0:
                    continue
                test_mask = np.where(test_leaves_t == leaf_id)[0]
                weights[np.ix_(test_mask, members)] += 1.0 / len(members)
        weights /= self.n_trees
        return weights

    def gaussian_moments_batch(self, X_new, batch_size=500):
        X_new = np.asarray(X_new)
        n = len(X_new)
        mu = np.zeros(n)
        sig = np.zeros(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            W = self._compute_weights_batch(X_new[start:end])
            mu[start:end] = W @ self.R_train
            second = W @ (self.R_train ** 2)
            var = np.maximum(second - mu[start:end] ** 2, 1e-10)
            sig[start:end] = np.sqrt(var)
        return mu, sig

    def cdf_batch(self, X_new, R_new, batch_size=500):
        X_new = np.asarray(X_new)
        R_new = np.asarray(R_new)
        n = len(X_new)
        U = np.zeros(n)
        if self.dist_mode == "gaussian":
            mu, sig = self.gaussian_moments_batch(X_new, batch_size=batch_size)
            return scipy_norm.cdf((R_new - mu) / sig)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            W = self._compute_weights_batch(X_new[start:end])
            indicators = self.R_train[None, :] <= R_new[start:end, None]
            U[start:end] = np.sum(W * indicators, axis=1)
        return U

    def density_batch(self, X_new, R_new, batch_size=500):
        X_new = np.asarray(X_new)
        R_new = np.asarray(R_new).reshape(-1)
        n = len(X_new)
        D = np.zeros(n)
        if self.dist_mode == "gaussian":
            mu, sig = self.gaussian_moments_batch(X_new, batch_size=batch_size)
            z = (R_new - mu) / sig
            return scipy_norm.pdf(z) / sig

        if not hasattr(self, "kernel_bw_"):
            sd = float(np.std(self.R_train))
            iqr = float(np.subtract(*np.percentile(self.R_train, [75, 25])))
            scale = sd if iqr <= 0 else min(sd, iqr / 1.34)
            if not np.isfinite(scale) or scale < 1e-8:
                scale = max(sd, 1.0)
            self.kernel_bw_ = max(0.9 * scale * (self.n_train ** (-1 / 5)), 1e-3)

        h = self.kernel_bw_
        const = 1.0 / np.sqrt(2.0 * np.pi)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            W = self._compute_weights_batch(X_new[start:end])
            Z = (R_new[start:end, None] - self.R_train[None, :]) / h
            K = const * np.exp(-0.5 * Z ** 2) / h
            D[start:end] = np.sum(W * K, axis=1)
        return np.maximum(D, 1e-12)

    def quantile_varying_multi(self, X_new, tau_matrix, batch_size=500):
        X_new = np.asarray(X_new)
        tau_matrix = np.asarray(tau_matrix)
        n, K = tau_matrix.shape
        Q = np.zeros((n, K))
        if self.dist_mode == "gaussian":
            mu, sig = self.gaussian_moments_batch(X_new, batch_size=batch_size)
            return mu[:, None] + sig[:, None] * scipy_norm.ppf(tau_matrix)
        n_R = len(self.sorted_R)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            W = self._compute_weights_batch(X_new[start:end])
            W_sorted = W[:, self.sorted_idx]
            cum_W = np.cumsum(W_sorted, axis=1)
            for j in range(end - start):
                for k in range(K):
                    idx = np.searchsorted(cum_W[j], tau_matrix[start + j, k], side="left")
                    idx = min(idx, n_R - 1)
                    Q[start + j, k] = self.sorted_R[idx]
        return Q


class ConditionalCDF_RFFGP:
    """Conditional CDF / quantile estimator from a Gaussian predictive distribution."""

    def __init__(
        self,
        length_scale=1.0,
        n_features=256,
        noise_var=5e-3,
        prior_var=1.0,
        random_state=42,
        **kwargs,
    ):
        self.length_scale = float(length_scale)
        self.n_features = int(n_features)
        self.noise_var = float(noise_var)
        self.prior_var = float(prior_var)
        self.random_state = int(random_state)
        self.extra_kwargs = kwargs

    def fit(self, X, R, verbose=True):
        X = np.asarray(X, dtype=float)
        R = np.asarray(R, dtype=float).reshape(-1)
        self.gp_ = RFFMeanRegressor(
            random_state=self.random_state,
            gp_kwargs={
                "length_scale": self.length_scale,
                "n_features": self.n_features,
                "noise_var": self.noise_var,
                "prior_var": self.prior_var,
            },
        )
        self.gp_.fit(X, R)
        self.n_train = len(R)
        if verbose:
            print(
                f"    GP CDF fitted: {self.n_train} samples, RFF features={self.n_features}, "
                f"length_scale={self.length_scale:.3f}, noise_var={self.noise_var:.4g}, "
                f"prior_var={self.prior_var:.4g}, R std={np.std(R):.4f}"
            )
        return self

    def gaussian_moments_batch(self, X_new, batch_size=500):
        del batch_size
        mu, sig = self.gp_.predict(X_new, return_std=True)
        sig = np.maximum(sig, 1e-8)
        return mu, sig

    def cdf_batch(self, X_new, R_new, batch_size=500):
        del batch_size
        X_new = np.asarray(X_new, dtype=float)
        R_new = np.asarray(R_new, dtype=float).reshape(-1)
        mu, sig = self.gaussian_moments_batch(X_new)
        return scipy_norm.cdf((R_new - mu) / sig)

    def density_batch(self, X_new, R_new, batch_size=500):
        del batch_size
        X_new = np.asarray(X_new, dtype=float)
        R_new = np.asarray(R_new, dtype=float).reshape(-1)
        mu, sig = self.gaussian_moments_batch(X_new)
        z = (R_new - mu) / sig
        return np.maximum(scipy_norm.pdf(z) / sig, 1e-12)

    def quantile_varying_multi(self, X_new, tau_matrix, batch_size=500):
        del batch_size
        X_new = np.asarray(X_new, dtype=float)
        tau_matrix = np.asarray(tau_matrix, dtype=float)
        mu, sig = self.gaussian_moments_batch(X_new)
        return mu[:, None] + sig[:, None] * scipy_norm.ppf(tau_matrix)


# =============================================================================
# Stage-2 smoother on U-space
# =============================================================================
