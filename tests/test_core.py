import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge as SkRidge
from sklearn.svm import SVR

from dgtl.data import Dataset, count_atoms
from dgtl.pipeline import metrics, run
from dgtl.regressors import KernelSVR, LocalRidge, Ridge
from dgtl.splits import make_split


def toy(n=240, d=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X @ rng.normal(size=d) + 0.05 * rng.normal(size=n)
    meta = pd.DataFrame({"group": rng.integers(0, 30, n).astype(str), "dataset": np.where(np.arange(n) < 150, "MP18", "MP21"),
                         "nelements": rng.integers(1, 5, n)})
    return Dataset("toy", X, y, meta, "y", "", {"a": (0, 5), "b": (5, d)})


def test_count_atoms():
    assert count_atoms("Ag2Sn2") == 4 and count_atoms("C8") == 8 and count_atoms("NbSe2") == 3


@pytest.mark.parametrize("shape", [(80, 10), (15, 40)])           # primal and dual (n < d) branches
@pytest.mark.parametrize("alpha", [0.0, 0.05, 3.0])
def test_ridge_path_matches_sklearn(shape, alpha):
    rng = np.random.default_rng(1)
    X, y, Xq = rng.normal(size=shape), rng.normal(size=shape[0]), rng.normal(size=(7, shape[1]))
    ref = (SkRidge(alpha=alpha) if alpha else LinearRegression()).fit(X, y).predict(Xq)
    got = Ridge().session(X, y, {"q": Xq}).predict({"alpha": alpha})["q"]
    assert np.allclose(ref, got, atol=1e-7)


@pytest.mark.parametrize("kernel", ["rbf", "linear", "poly"])
def test_precomputed_svr_matches_sklearn(kernel):
    rng = np.random.default_rng(2)
    X, y, Xq = rng.uniform(size=(60, 8)), rng.normal(size=60), rng.uniform(size=(9, 8))
    ref = SVR(kernel=kernel, C=10, epsilon=0.02, coef0=0.0).fit(X, y).predict(Xq)
    got = KernelSVR(kernel).session(X, y, {"q": Xq}).predict({"C": 10, "epsilon": 0.02})["q"]
    assert np.allclose(ref, got, atol=2e-3)       # both stop at libsvm's 1e-3 tolerance


def test_local_ridge_matches_explicit_fit():
    rng = np.random.default_rng(3)
    X, y, Xq = rng.normal(size=(50, 6)), rng.normal(size=50), rng.normal(size=(4, 6))
    got = LocalRidge(neighbors=[12]).session(X, y, {"q": Xq}).predict({"n_neighbors": 12, "alpha": 0.3})["q"]
    for i, q in enumerate(Xq):
        nb = np.argsort(((X - q) ** 2).sum(1))[:12]
        assert np.isclose(SkRidge(alpha=0.3).fit(X[nb], y[nb]).predict(q[None])[0], got[i], atol=1e-7)


def test_splits_do_not_leak():
    ds = toy()
    g = ds.meta["group"].to_numpy()
    sp = make_split(ds, "group")
    assert not set(g[sp.test]) & set(g[sp.pool])
    for k in range(sp.n_folds):                                    # validation folds are grouped too
        assert not set(g[sp.fold == k]) & set(g[sp.pool & (sp.fold != k)])
    sp = make_split(ds, "target", lower=-1.0, upper=1.0)
    assert ds.y[sp.pool].min() >= -1.0 and ds.y[sp.pool].max() <= 1.0
    assert ((ds.y[sp.test] < -1.0) | (ds.y[sp.test] > 1.0)).all()
    lo, hi = np.sort(ds.y)[1], np.sort(ds.y)[-2]                   # one sample beyond each cut
    sp = make_split(ds, "target_group", lower=lo, upper=hi)
    assert not set(g[sp.test]) & set(g[sp.pool])
    assert ds.y[sp.pool].min() >= lo and ds.y[sp.pool].max() <= hi
    assert (ds.meta["dataset"][make_split(ds, "temporal").test] == "MP21").all()
    assert (ds.meta["nelements"][make_split(ds, "nelements").pool] <= 2).all()
    assert (make_split(ds, "random").fold[make_split(ds, "random").test] == -1).all()


def test_pipeline_recovers_linear_trend_outside_training_range():
    ds = toy(n=400)
    sp = make_split(ds, "target", lower=np.quantile(ds.y, .1), upper=np.quantile(ds.y, .9))
    res = run(ds, sp, Ridge(alpha_max=1.0, alpha_step=0.25))
    assert res.frac_true_outside == 1.0 and res.frac_pred_outside > 0.9
    assert res.test["r2"] > 0.98 and abs(res.slope_outside - 1) < 0.05
    sub = run(ds, sp, Ridge(alpha_max=1.0, alpha_step=0.25), gnns=["a"])   # feature-block ablation runs
    assert sub.features == ["a"] and sub.test["rmse"] > res.test["rmse"]
    assert metrics([1, 2, 3], [1, 2, 3])["rmse"] == 0
