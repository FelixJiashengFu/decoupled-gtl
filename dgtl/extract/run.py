"""Extract embeddings for a table of structures.

    python -m dgtl.extract.run --dataset lic --data-dir data --checkpoint-dir checkpoints --out work/emb_lic
    python -m dgtl.extract.run --csv my.csv --structure-col cif --checkpoint-dir checkpoints --out work/emb

Writes one `<gnn>.npy` per model (rows in CSV order), usable with `dgtl run --dataset custom`.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .graph import structure_to_graph
from .models import CHECKPOINTS, Extractor


def parse_structure(text):
    """CIF text, or a pymatgen Structure dict (as a Python or JSON literal)."""
    from pymatgen.core import Structure
    s = str(text).lstrip()
    if s.startswith("{"):
        try:
            return Structure.from_dict(ast.literal_eval(s))
        except (ValueError, SyntaxError):
            return Structure.from_dict(json.loads(s))
    return Structure.from_str(s, fmt="cif")


def extract(structures, gnns, checkpoint_dir=None, cutoff=6.0, max_neighbors=50, progress=True) -> dict:
    models = {}
    for g in gnns:
        m = Extractor(g)
        if checkpoint_dir is not None:
            m.load_checkpoint(Path(checkpoint_dir) / CHECKPOINTS[g])
        models[g] = m
    out = {g: [] for g in gnns}
    it = structures
    if progress:
        from tqdm import tqdm
        it = tqdm(structures)
    for s in it:
        graph = structure_to_graph(s, cutoff, max_neighbors)
        for g, m in models.items():
            out[g].append(m(graph).numpy())
    return {g: np.stack(v) for g, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["lic", "aoi"]); ap.add_argument("--data-dir", default="data")
    ap.add_argument("--csv"); ap.add_argument("--structure-col")
    ap.add_argument("--checkpoint-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gnns", default="cgcnn,schnet,dimenetpp"); ap.add_argument("--max-rows", type=int)
    a = ap.parse_args(argv)
    if a.dataset == "lic":
        csv, col = Path(a.data_dir) / "lic" / "LC_database_rev_with_cif.csv", "cif"
    elif a.dataset == "aoi":
        csv, col = Path(a.data_dir) / "aoi" / "AoI_with_timestamp.csv", "structure_as_dict"
    else:
        csv, col = a.csv, a.structure_col
    col_data = pd.read_csv(csv, usecols=[col], nrows=a.max_rows)[col]
    torch.set_grad_enabled(False)
    emb = extract((parse_structure(t) for t in col_data), a.gnns.split(","), a.checkpoint_dir)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for g, x in emb.items():
        np.save(out / f"{g}.npy", x.astype(np.float32))
        print(g, x.shape)


if __name__ == "__main__":
    main()
