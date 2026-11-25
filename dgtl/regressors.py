"""Second-stage regressors on frozen features.

Each regressor opens a *session* per training fold: the expensive, hyperparameter-
independent work (eigendecomposition, kernel matrices, neighbour search) is done once,
after which every grid point is cheap. This is what makes the paper's full grids run on
a laptop CPU instead of needing a GPU SVR.
"""
from __future__ import annotations

import numpy as np
from sklearn.svm import SVR


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    np.maximum(d, 0.0, out=d)
    return d


# --------------------------------------------------------------------------- ridge
class RidgeSession:
    """Exact ridge path from one eigendecomposition (intercept unpenalised, like sklearn)."""

    def __init__(self, Xtr, ytr, queries):
        self.mu, self.ybar = Xtr.mean(0), ytr.mean()
        Xc = Xtr - self.mu
        n, d = Xc.shape
        if n >= d:
            s, V = np.linalg.eigh(Xc.T @ Xc)
            self.V = V
        else:                                   # dual form: fewer samples than features
            s, U = np.linalg.eigh(Xc @ Xc.T)
            with np.errstate(divide="ignore", invalid="ignore"):
                V = Xc.T @ (U / np.sqrt(np.maximum(s, 1e-300)))
            self.V = V
        self.s = np.maximum(s, 0.0)
        self.c = self.V.T @ (Xc.T @ (ytr - self.ybar))
        self.tol = self.s.max() * max(n, d) * np.finfo(float).eps
        self.Q = {k: (Xq - self.mu) @ self.V for k, Xq in queries.items()}

    def _inv(self, alpha):
        # alpha = 0 gives the minimum-norm least-squares solution (what LinearRegression returns)
        return np.where(self.s > self.tol, 1.0 / (np.maximum(self.s, 1e-300) + alpha), 0.0)

    def predict(self, p):
        w = self._inv(p["alpha"]) * self.c
        return {k: Q @ w + self.ybar for k, Q in self.Q.items()}

    def coef(self, p):
        return self.V @ (self._inv(p["alpha"]) * self.c)

    def diagnostic(self, p, Xq):
        """max_j |x_j w_j|: the largest single-feature contribution (paper Sec. IV E)."""
        return np.abs(Xq * self.coef(p)[None, :]).max(1)


class Ridge:
    name = "ridge"

    def __init__(self, alpha_max=10.0, alpha_step=0.05):
        self.grid = [{"alpha": float(a)} for a in np.arange(0.0, alpha_max + 1e-9, alpha_step)]

    def session(self, Xtr, ytr, queries):
        return RidgeSession(Xtr, ytr, queries)


# --------------------------------------------------------------------------- SVR
class SVRSession:
    def __init__(self, Xtr, ytr, queries, kernel, degree, max_iter):
        self.ytr, self.kernel, self.max_iter = ytr, kernel, max_iter
        self.gamma = 1.0 / (Xtr.shape[1] * Xtr.var())          # sklearn / cuML gamma='scale'
        if kernel == "rbf":
            f = lambda A: np.exp(-self.gamma * _sqdist(A, Xtr))
        elif kernel == "linear":
            f = lambda A: A @ Xtr.T
        elif kernel == "poly":
            f = lambda A: (self.gamma * (A @ Xtr.T)) ** degree
        else:
            raise ValueError(kernel)
        self.Ktt = f(Xtr)
        self.Kq = {k: f(Xq) for k, Xq in queries.items()}

    def predict(self, p):
        m = SVR(kernel="precomputed", C=p["C"], epsilon=p["epsilon"], cache_size=2000, max_iter=self.max_iter)
        m.fit(self.Ktt, self.ytr)
        return {k: m.predict(K) for k, K in self.Kq.items()}

    def diagnostic(self, p, key="test"):
        """1 - max_i K(x, x_i): distance to the nearest training point in RBF kernel space."""
        if self.kernel != "rbf":
            raise ValueError("dissimilarity diagnostic is defined for the RBF kernel")
        return 1.0 - self.Kq[key].max(1)


class KernelSVR:
    def __init__(self, kernel="rbf", grid="paper", degree=3, max_iter=-1):
        self.kernel, self.degree, self.max_iter = kernel, degree, max_iter
        self.name = "svr" if kernel == "rbf" else f"svr_{kernel}"
        if grid == "paper":      # C: 10..1000 in 5 geometric steps; eps: 0..0.06 step 0.02
            self.grid = [{"C": float(c), "epsilon": float(e)}
                         for c in np.geomspace(10, 1000, 5) for e in np.arange(0, 0.0601, 0.02)]
        elif grid == "medium":   # 6 of the paper's 20 points; used for the ablation runs
            self.grid = [{"C": float(c), "epsilon": float(e)} for c in (10.0, 100.0, 1000.0) for e in (0.0, 0.02)]
        elif grid == "fast":
            self.grid = [{"C": float(c), "epsilon": 0.02} for c in (10.0, 100.0)]
        else:
            self.grid = list(grid)

    def session(self, Xtr, ytr, queries):
        return SVRSession(Xtr, ytr, queries, self.kernel, self.degree, self.max_iter)


# --------------------------------------------------------------------------- similarity-based (local) ridge
class LocalRidgeSession:
    """For each query: ridge on its n nearest training points (Lee et al. 2025 style).

    Solved in dual form from one global Gram matrix; the local centring is applied
    algebraically, so each query costs one n x n solve.
    """

    def __init__(self, Xtr, ytr, queries, n_max):
        self.ytr = ytr
        self.Gtt = Xtr @ Xtr.T
        self.Gq, self.order = {}, {}
        sq_t = np.einsum("ij,ij->i", Xtr, Xtr)
        for k, Xq in queries.items():
            G = Xq @ Xtr.T
            d = (Xq * Xq).sum(1)[:, None] + sq_t[None, :] - 2 * G
            self.Gq[k] = G
            kk = min(n_max, len(ytr))
            part = np.argpartition(d, kk - 1, axis=1)[:, :kk]
            self.order[k] = np.take_along_axis(part, np.argsort(np.take_along_axis(d, part, 1), 1), 1)

    def predict(self, p, keys=None):
        n, alpha = int(p["n_neighbors"]), p["alpha"]
        out = {}
        for k in (keys or self.Gq):
            G = self.Gq[k]
            pred = np.empty(len(G))
            for i in range(len(G)):
                nb = self.order[k][i, :n]
                K = self.Gtt[np.ix_(nb, nb)]
                kq = G[i, nb]
                yb = self.ytr[nb]
                rm = K.mean(0)
                Kc = K - rm[None, :] - rm[:, None] + K.mean()         # centre on the neighbour mean
                kc = kq - rm - kq.mean() + K.mean()
                a = np.linalg.solve(Kc + max(alpha, 1e-10) * np.eye(len(nb)), yb - yb.mean())
                pred[i] = kc @ a + yb.mean()
            out[k] = pred
        return out


class LocalRidge:
    name = "local_ridge"

    def __init__(self, neighbors=range(10, 201, 10), alphas=np.arange(0.0, 1.0001, 0.05), alpha0=0.05,
                 val_subsample=300, seed=0):
        self.neighbors, self.alphas, self.alpha0 = list(neighbors), [float(a) for a in alphas], alpha0
        self.val_subsample, self.seed = val_subsample, seed
        self.grid = None           # tuned in two stages by the pipeline (n first, then alpha)

    def session(self, Xtr, ytr, queries):
        return LocalRidgeSession(Xtr, ytr, queries, max(self.neighbors))


# --------------------------------------------------------------------------- small MLP head
class MLPSession:
    def __init__(self, Xtr, ytr, queries, seed):
        self.Xtr, self.ytr, self.queries, self.seed = Xtr, ytr, queries, seed

    def predict(self, p):
        from sklearn.neural_network import MLPRegressor
        m = MLPRegressor(hidden_layer_sizes=(512, 128, 128), learning_rate_init=1e-4, batch_size=32,
                         early_stopping=True, validation_fraction=0.25, n_iter_no_change=5, max_iter=200,
                         random_state=self.seed).fit(self.Xtr, self.ytr)
        return {k: m.predict(Xq) for k, Xq in self.queries.items()}


class MLP:
    """Neural-network head on the frozen features (512-128-128, as in the paper's comparison;
    ReLU here instead of softplus because scikit-learn has no softplus)."""
    name = "mlp"

    def __init__(self, seed=0):
        self.seed, self.grid = seed, [{}]

    def session(self, Xtr, ytr, queries):
        return MLPSession(Xtr, ytr, queries, self.seed)


def get_regressor(name: str, svr_grid="paper"):
    if name == "ridge":
        return Ridge()
    if name == "svr":
        return KernelSVR("rbf", svr_grid)
    if name in ("svr_linear", "svr_poly"):       # capped: unbounded kernels can stall libsvm at large C
        return KernelSVR(name.split("_")[1], svr_grid, max_iter=500_000)
    if name == "local_ridge":
        return LocalRidge()
    if name == "mlp":
        return MLP()
    raise ValueError(f"unknown regressor {name!r}")


REGRESSORS = ("ridge", "svr", "svr_linear", "svr_poly", "local_ridge", "mlp")
