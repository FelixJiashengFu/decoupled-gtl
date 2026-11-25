"""Train / validation / test partitions that separate interpolation from extrapolation.

Every split first sets the test set aside according to the deployment scenario, then
cuts the remaining pool into validation folds. When the test set is defined by groups,
validation folds are grouped too, so model selection never sees a shared host.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold, KFold

from .data import Dataset

SPLITS = ("random", "group", "target", "target_group", "temporal", "nelements")


@dataclass
class Split:
    kind: str
    test: np.ndarray        # bool mask
    fold: np.ndarray        # validation fold id for pool rows, -1 for test rows
    n_folds: int
    description: str

    @property
    def pool(self) -> np.ndarray:
        return ~self.test


def _folds(ds: Dataset, test: np.ndarray, n_folds: int, grouped: bool, seed: int) -> np.ndarray:
    fold = np.full(len(ds), -1)
    pool = np.where(~test)[0]
    if grouped:
        it = GroupKFold(n_folds).split(pool, groups=ds.meta["group"].to_numpy()[pool])
    else:
        it = KFold(n_folds, shuffle=True, random_state=seed).split(pool)
    for k, (_, va) in enumerate(it):
        fold[pool[va]] = k
    return fold


def make_split(ds: Dataset, kind: str, *, seed: int = 3776, n_folds: int = 4, test_frac_folds: int = 5,
               lower: float | None = None, upper: float | None = None, max_train_elements: int = 2) -> Split:
    """Build a split.

    random        interpolation baseline: random test set
    group         structural extrapolation: whole groups (hosts) held out
    target        property extrapolation: y < lower or y > upper is the test set
    target_group  coupled: any group containing an out-of-range y is held out whole
    temporal      rows whose `dataset` column is MP21 are the test set
    nelements     compositional complexity: train on <= max_train_elements element types
    """
    y, g = ds.y, ds.meta["group"].to_numpy() if "group" in ds.meta else None
    test = np.zeros(len(ds), bool)
    grouped = False
    if kind == "random":
        for k, (_, va) in enumerate(KFold(test_frac_folds, shuffle=True, random_state=seed).split(y)):
            if k == test_frac_folds - 1:
                test[va] = True
        desc = f"random 1/{test_frac_folds} test"
    elif kind == "group":
        for k, (_, va) in enumerate(GroupKFold(test_frac_folds).split(y, groups=g)):
            if k == test_frac_folds - 1:
                test[va] = True
        grouped, desc = True, f"1/{test_frac_folds} of groups held out"
    elif kind in ("target", "target_group"):
        if lower is None and upper is None:
            raise ValueError(f"split {kind!r} needs lower and/or upper")
        lo = -np.inf if lower is None else lower
        hi = np.inf if upper is None else upper
        out = (y < lo) | (y > hi)                         # strict, as in the paper
        if kind == "target":
            test = out
            desc = f"y < {lo:g} or y > {hi:g} is test"
        else:
            test = np.isin(g, np.unique(g[out]))
            grouped, desc = True, f"groups with any y < {lo:g} or y > {hi:g} held out"
    elif kind == "temporal":
        test = (ds.meta["dataset"] == "MP21").to_numpy()
        desc = "MP18 -> MP21"
    elif kind == "nelements":
        test = (ds.meta["nelements"] > max_train_elements).to_numpy()
        desc = f"train on <= {max_train_elements} element types"
    else:
        raise ValueError(f"unknown split {kind!r}; choose from {SPLITS}")
    if test.all() or not test.any():
        raise ValueError(f"split {kind!r} produced an empty {'pool' if test.all() else 'test set'}")
    return Split(kind, test, _folds(ds, test, n_folds, grouped, seed), n_folds, desc)
