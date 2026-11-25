"""The experiment suite: every scenario in the paper (and its supplement) as a named config."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np

from .data import ALL_GNNS, load_dataset
from .pipeline import run
from .regressors import get_regressor
from .splits import make_split

# LIC thresholds used in the paper (formation energy, eV/atom)
LIC_SPLITS = {
    "random": {},
    "group": {},
    "target": {"lower": -1.7, "upper": 0.1},
    "target_group": {"lower": -2.6, "upper": 0.4},
}

# Reference numbers printed in the paper, for the side-by-side check (test RMSE unless noted).
PAPER = {
    "lic/eform/random/svr/all": {"mae": 0.035, "rmse": 0.055, "r2": 0.995},
    "lic/eform/group/svr/all": {"mae": 0.076, "rmse": 0.099, "r2": 0.969},
    "lic/eform/target/svr/all": {"mae": 0.150, "rmse": 0.205, "r2": 0.975},
    "lic/eform/target_group/svr/all": {"mae": 0.155, "rmse": 0.199, "r2": 0.966},
    "aoi/eform/temporal/svr/all": {"mae": 0.143, "rmse": 0.296, "r2": 0.873, "rmse_inside": 0.151,
                                   "rmse_outside": 0.881},
    "aoi/eform/temporal/svr/matminer": {"mae": 0.297, "rmse": 0.702, "r2": 0.284, "rmse_inside": 0.240,
                                        "rmse_outside": 2.262},
    "lic/efermi/random/svr/all": {"rmse": 0.352},
    "lic/efermi/group/svr/all": {"rmse": 0.502},
    "lic/efermi/target/svr/all": {"rmse": 1.269},
    "lic/efermi/target_group/svr/all": {"rmse": 1.200},
}
# End-to-end baselines reported in the paper (not re-run here: they need GPU fine-tuning).
PAPER_BASELINES = {
    "lic/eform/random": {"End-to-end CGCNN": 0.063},
    "lic/eform/group": {"End-to-end CGCNN": 0.120},
    "lic/eform/target": {"End-to-end CGCNN": 0.378},
    "lic/eform/target_group": {"End-to-end CGCNN": 0.308},
    "aoi/eform/temporal": {"End-to-end CGCNN": 0.876, "ALIGNN": 0.747, "XGBoost": 0.537, "Random forest": 0.879,
                           "Linear forest": 0.606},
}


def _cfg(dataset, split, reg, gnns=ALL_GNNS, target="eform", features="gnn", tag=None, **split_kw):
    feat = "matminer" if features == "matminer" else ("all" if tuple(gnns) == ALL_GNNS else "+".join(gnns))
    key = f"{dataset}/{target}/{tag or split}/{reg}/{feat}"
    return {"key": key, "dataset": dataset, "split": split, "regressor": reg, "gnns": list(gnns), "target": target,
            "features": features, "split_kw": split_kw}


def suite(name: str = "main") -> list[dict]:
    """main: headline tables. full: + ablations, kernels, threshold scans, Fermi energy."""
    c = []
    for sp, kw in LIC_SPLITS.items():
        for reg in ("svr", "ridge"):
            c.append(_cfg("lic", sp, reg, **kw))
    for reg in ("svr", "ridge"):
        c.append(_cfg("aoi", "temporal", reg))
    c.append(_cfg("aoi", "temporal", "svr", features="matminer"))
    c.append(_cfg("aoi", "temporal", "ridge", features="matminer"))
    if name == "main":
        return c
    # --- which GNNs matter (single and pairwise)
    subsets = [s for r in (1, 2) for s in combinations(ALL_GNNS, r)]
    for s in subsets:
        for sp, kw in LIC_SPLITS.items():
            c.append(_cfg("lic", sp, "svr", gnns=s, **kw))
        c.append(_cfg("aoi", "temporal", "svr", gnns=s))
    # --- which regressor matters
    for reg in ("svr_linear", "svr_poly", "local_ridge", "mlp"):
        for sp, kw in LIC_SPLITS.items():
            c.append(_cfg("lic", sp, reg, **kw))
        c.append(_cfg("aoi", "temporal", reg))
    c.append(_cfg("aoi", "temporal", "local_ridge", features="matminer"))
    # --- other extrapolation settings
    for reg in ("svr", "ridge"):
        c.append(_cfg("aoi", "nelements", reg))
    for lo, hi in ((-1.4, -0.1), (-2.0, 0.3)):
        c.append(_cfg("lic", "target", "svr", tag=f"target[{lo},{hi}]", lower=lo, upper=hi))
    for lo, hi in ((-2.3, 0.3), (-2.9, 0.5)):
        c.append(_cfg("lic", "target_group", "svr", tag=f"target_group[{lo},{hi}]", lower=lo, upper=hi))
    for hi in (0.5, 1.0):
        c.append(_cfg("aoi", "target", "svr", tag=f"target[<{hi}]", upper=hi))
    # --- a second property: Fermi energy (thresholds are quantiles; the paper does not state its cuts)
    for reg in ("svr", "ridge"):
        for sp in ("random", "group"):
            c.append(_cfg("lic", sp, reg, target="efermi"))
        c.append(_cfg("lic", "target", reg, target="efermi", tag="target", lower="q05", upper="q95"))
        c.append(_cfg("lic", "target_group", reg, target="efermi", tag="target_group", lower="q01", upper="q99"))
    seen, out = set(), []
    for x in c:
        if x["key"] not in seen:
            seen.add(x["key"]); out.append(x)
    return out


def run_config(cfg: dict, data_dir: str, out_dir: str, svr_grid="paper", force=False) -> dict:
    out = Path(out_dir) / (cfg["key"].replace("/", "__") + ".json")
    if out.exists() and not force:
        return json.loads(out.read_text())
    kw = {"gnns": ALL_GNNS} if cfg["features"] == "gnn" else {"features": "matminer"}
    if cfg["dataset"] == "lic":
        kw["target"] = cfg["target"]
    ds = load_dataset(cfg["dataset"], data_dir, **kw)
    skw = dict(cfg["split_kw"])
    for b in ("lower", "upper"):                       # 'q05' -> 5th percentile of the target
        if isinstance(skw.get(b), str):
            skw[b] = float(np.quantile(ds.y, int(skw[b][1:]) / 100))
    sp = make_split(ds, cfg["split"], **skw)
    res = run(ds, sp, get_regressor(cfg["regressor"], svr_grid),
              gnns=cfg["gnns"] if cfg["features"] == "gnn" else None)
    d = res.summary()
    d.update(key=cfg["key"], split_description=sp.description, split_kw=skw, n_pool=int(sp.pool.sum()),
             n_test=int(sp.test.sum()), target=cfg["target"], unit=ds.unit,
             svr_grid=svr_grid if cfg["regressor"].startswith("svr") else None)
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if res.diagnostic:
        extra["diag_score"] = res.diagnostic["score"]
    if "group" in ds.meta:
        extra["group_test"] = ds.meta["group"].to_numpy()[sp.test].astype(str)
    np.savez_compressed(out.with_suffix(".npz"), y_test=res.y_test, pred_test=res.pred_test, y_oof=res.y_oof,
                        pred_oof=res.pred_oof, **extra)
    out.write_text(json.dumps(d, indent=1))
    return d


def run_suite(name, data_dir, out_dir, jobs=1, svr_grid="paper", force=False, only=None):
    cfgs = [c for c in suite(name) if not only or only in c["key"]]
    done = []
    if jobs <= 1:
        for c in cfgs:
            d = run_config(c, data_dir, out_dir, svr_grid, force)
            print(f"{d['key']:55s} test rmse {d['test']['rmse']:.3f}  ({d['seconds']:.0f}s)", flush=True)
            done.append(d)
        return done
    with ProcessPoolExecutor(jobs) as ex:
        futs = {ex.submit(run_config, c, data_dir, out_dir, svr_grid, force): c for c in cfgs}
        for f in as_completed(futs):
            try:
                d = f.result()
                print(f"{d['key']:55s} test rmse {d['test']['rmse']:.3f}  ({d['seconds']:.0f}s)", flush=True)
                done.append(d)
            except Exception as e:                       # keep the rest of the suite going
                print(f"FAILED {futs[f]['key']}: {e!r}", flush=True)
    return done
