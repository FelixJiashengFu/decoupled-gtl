"""Run one experiment: scale -> cross-validated model selection -> fold-ensemble test prediction."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler

from .data import Dataset
from .regressors import LocalRidge
from .splits import Split


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return {"n": 0}
    ss = float(((y - y.mean()) ** 2).sum())
    return {"n": int(len(y)), "mae": float(np.abs(y - p).mean()), "rmse": float(np.sqrt(((y - p) ** 2).mean())),
            "r2": float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")}


@dataclass
class Result:
    dataset: str
    split: str
    regressor: str
    features: list
    best_params: dict
    oof: dict
    test: dict
    test_inside: dict            # test rows whose true y lies inside the training range
    test_outside: dict           # ... outside it: the property-extrapolation region
    train_range: tuple
    frac_true_outside: float     # share of test rows that require property extrapolation
    frac_pred_outside: float     # share of test predictions that actually leave the training range
    slope_outside: float         # d(pred)/d(true) over the outside rows: 1 = no compression toward the mean
    seconds: float
    y_test: np.ndarray = field(repr=False, default=None)
    pred_test: np.ndarray = field(repr=False, default=None)
    y_oof: np.ndarray = field(repr=False, default=None)
    pred_oof: np.ndarray = field(repr=False, default=None)
    diagnostic: dict = field(repr=False, default=None)

    def summary(self) -> dict:
        keep = ("dataset", "split", "regressor", "features", "best_params", "oof", "test", "test_inside",
                "test_outside", "train_range", "frac_true_outside", "frac_pred_outside", "slope_outside", "seconds")
        d = {k: getattr(self, k) for k in keep}
        if self.diagnostic:
            d["diagnostic"] = {k: v for k, v in self.diagnostic.items() if k != "score"}
        return d


def _oof_rmse(sessions, masks, z, params, n_pool):
    se = 0.0
    for s, va in zip(sessions, masks):
        se += float(((s.predict(params)["val"] - z[va]) ** 2).sum())
    return np.sqrt(se / n_pool)


def run(ds: Dataset, split: Split, reg, *, gnns=None, verbose=False) -> Result:
    t0 = time.time()
    X = ds.X
    if gnns is not None:                                   # feature-block ablation
        X = np.concatenate([X[:, slice(*ds.feature_blocks[g])] for g in gnns], 1)
    feats = list(gnns) if gnns is not None else list(ds.feature_blocks)
    pool, test = split.pool, split.test
    X = MinMaxScaler().fit(X[pool]).transform(X)           # statistics from the non-test pool only
    mu, sd = ds.y[pool].mean(), ds.y[pool].std() or 1.0
    z = (ds.y - mu) / sd

    sessions, masks = [], []
    for k in range(split.n_folds):
        tr, va = pool & (split.fold != k), split.fold == k
        q = {"val": X[va], "test": X[test]}
        if isinstance(reg, LocalRidge) and reg.val_subsample and va.sum() > reg.val_subsample:
            rng = np.random.default_rng(reg.seed + k)      # paper: tune on 300 validation rows per fold
            sub = np.sort(rng.choice(np.where(va)[0], reg.val_subsample, replace=False))
            q["tune"] = X[sub]
            masks.append((va, sub))
        else:
            masks.append((va, None))
        sessions.append(reg.session(X[tr], z[tr], q))

    if isinstance(reg, LocalRidge):                        # two-stage search: n at fixed alpha, then alpha
        def score(p):
            se, n = 0.0, 0
            for s, (va, sub) in zip(sessions, masks):
                key, idx = ("tune", sub) if sub is not None else ("val", np.where(va)[0])
                pr = s.predict(p, keys=[key])[key]
                se += float(((pr - z[idx]) ** 2).sum()); n += len(idx)
            return np.sqrt(se / n)
        best_n = min(reg.neighbors, key=lambda n: score({"n_neighbors": n, "alpha": reg.alpha0}))
        best = min(({"n_neighbors": best_n, "alpha": a} for a in reg.alphas), key=score)
    else:
        n_pool, best, best_s = int(pool.sum()), None, np.inf
        for p in reg.grid:
            s = _oof_rmse(sessions, [m[0] for m in masks], z, p, n_pool)
            if verbose:
                print(f"  {p} -> oof rmse(z) {s:.4f}", flush=True)
            if s < best_s:
                best, best_s = p, s

    oof = np.full(len(z), np.nan)
    test_preds = []
    for s, (va, _) in zip(sessions, masks):
        out = s.predict(best)
        oof[va] = out["val"]
        test_preds.append(out["test"])
    pt = np.mean(test_preds, 0) * sd + mu                  # ensemble of the fold models
    po, yt, yo = oof[pool] * sd + mu, ds.y[test], ds.y[pool]
    lo, hi = float(yo.min()), float(yo.max())
    outside = (yt < lo) | (yt > hi)
    slope = float(np.polyfit(yt[outside], pt[outside], 1)[0]) if outside.sum() > 2 and np.ptp(yt[outside]) > 0 \
        else float("nan")

    diag = None
    err = np.abs(pt - yt)
    try:
        if reg.name == "svr":
            score = np.mean([s.diagnostic(best) for s in sessions], 0)
            diag = {"name": "1 - max RBF similarity to training set"}
        elif reg.name == "ridge":
            score = np.mean([s.diagnostic(best, X[test]) for s in sessions], 0)
            diag = {"name": "max |x_j w_j|"}
        if diag:
            diag.update(score=score, pearson=float(pearsonr(score, err)[0]), spearman=float(spearmanr(score, err)[0]),
                        mean_score_inside=float(score[~outside].mean()) if (~outside).any() else None,
                        mean_score_outside=float(score[outside].mean()) if outside.any() else None)
    except Exception:
        diag = None

    return Result(ds.name, split.kind, reg.name, feats, best, metrics(yo, po), metrics(yt, pt),
                  metrics(yt[~outside], pt[~outside]), metrics(yt[outside], pt[outside]), (lo, hi),
                  float(outside.mean()), float(((pt < lo) | (pt > hi)).mean()), slope, time.time() - t0,
                  yt, pt, yo, po, diag)
