"""Command line: `python -m dgtl run|suite|report`."""
from __future__ import annotations

import argparse
import json

from .data import ALL_GNNS, load_custom, load_dataset
from .regressors import REGRESSORS, get_regressor
from .splits import SPLITS, make_split


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dgtl", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="one experiment")
    r.add_argument("--data-dir", default="data")
    r.add_argument("--dataset", default="lic", help="lic | aoi | custom")
    r.add_argument("--target", default="eform", help="lic: eform | efermi; custom: CSV column")
    r.add_argument("--features", default="gnn", choices=["gnn", "matminer"])
    r.add_argument("--gnns", default=",".join(ALL_GNNS), help="comma list drawn from " + ",".join(ALL_GNNS))
    r.add_argument("--split", default="random", choices=SPLITS)
    r.add_argument("--lower", type=float); r.add_argument("--upper", type=float)
    r.add_argument("--regressor", default="svr", choices=REGRESSORS)
    r.add_argument("--svr-grid", default="paper", choices=["paper", "medium", "fast"])
    r.add_argument("--seed", type=int, default=3776)
    r.add_argument("--csv", help="custom dataset: CSV path"); r.add_argument("--embeddings", nargs="+")
    r.add_argument("--group", help="custom dataset: group column")
    r.add_argument("--plot", help="write a parity plot to this PNG")
    r.add_argument("-v", "--verbose", action="store_true")

    s = sub.add_parser("suite", help="run a whole named suite (resumable)")
    s.add_argument("name", choices=["main", "full"])
    s.add_argument("--data-dir", default="data"); s.add_argument("--out-dir", default="results/runs")
    s.add_argument("--jobs", type=int, default=1); s.add_argument("--svr-grid", default="paper")
    s.add_argument("--only", help="substring filter on experiment keys"); s.add_argument("--force", action="store_true")

    p = sub.add_parser("report", help="build results/RESULTS.md and figures from finished runs")
    p.add_argument("--runs", default="results/runs"); p.add_argument("--out", default="results")

    a = ap.parse_args(argv)
    if a.cmd == "run":
        from .pipeline import run
        gnns = a.gnns.split(",")
        if a.dataset == "custom":
            ds = load_custom(a.csv, a.embeddings, a.target, a.group); gnns = None
        elif a.dataset == "lic":
            ds = load_dataset("lic", a.data_dir, gnns=ALL_GNNS, target=a.target)
        else:
            ds = load_dataset("aoi", a.data_dir, features=a.features)
            gnns = gnns if a.features == "gnn" else None
        sp = make_split(ds, a.split, seed=a.seed, lower=a.lower, upper=a.upper)
        res = run(ds, sp, get_regressor(a.regressor, a.svr_grid), gnns=gnns, verbose=a.verbose)
        print(json.dumps(res.summary(), indent=1))
        if a.plot:
            from .report import parity
            parity(res.y_oof, res.pred_oof, res.y_test, res.pred_test, res.train_range, a.plot,
                   title=f"{ds.name} · {sp.description} · {res.regressor}", unit=ds.unit)
    elif a.cmd == "suite":
        from .experiments import run_suite
        run_suite(a.name, a.data_dir, a.out_dir, a.jobs, a.svr_grid, a.force, a.only)
    else:
        from .report import build
        build(a.runs, a.out)


if __name__ == "__main__":
    main()
