"""Dataset loading: targets and metadata from CSV, features from embedding .npy files."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# short name -> file stem used by the published embeddings
GNN_STEMS = {
    "cgcnn": "CGCNN",
    "schnet": "schnet_mean",
    "dimenetpp": "DimeNetPlusPlus_multi_mean",
}
ALL_GNNS = tuple(GNN_STEMS)

# AoI columns that are not matminer descriptors
_AOI_META = [
    "material_id", "formula_pretty", "formula_anonymous", "formation_energy_per_atom",
    "energy_above_hull", "is_metal", "is_stable", "ordering", "uncorrected_energy_per_atom",
    "energy_per_atom", "nelements", "nsites", "elements", "task_ids", "space_group_number",
    "k_voigt", "g_voigt", "structure_as_dict", "dataset",
]


@dataclass
class Dataset:
    name: str
    X: np.ndarray            # (n, d) raw features, float64
    y: np.ndarray            # (n,) target
    meta: pd.DataFrame       # one row per sample, aligned with X / y
    target: str
    unit: str
    feature_blocks: dict = field(default_factory=dict)  # block name -> (start, stop)

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "Dataset":
        return Dataset(self.name, self.X[mask], self.y[mask], self.meta.loc[mask].reset_index(drop=True),
                       self.target, self.unit, self.feature_blocks)


def count_atoms(formula: str) -> int:
    """Number of atoms in a plain formula such as 'Ag2Sn2' (no brackets)."""
    return sum(int(n) if n else 1 for _, n in re.findall(r"([A-Z][a-z]*)(\d*)", str(formula)))


def _load_embeddings(emb_dir: Path, stems: list[str], gnns, sort_key: np.ndarray):
    blocks, parts, start = {}, [], 0
    for g in gnns:
        if g not in GNN_STEMS:
            raise ValueError(f"unknown GNN {g!r}; choose from {ALL_GNNS}")
        stem = stems[g]
        x = np.load(emb_dir / f"{stem}_emb_x.npy")
        order = np.load(emb_dir / f"{stem}_sort.npy").ravel()
        # align embedding rows to the CSV by the shared integer key
        pos = {int(k): i for i, k in enumerate(order)}
        try:
            idx = np.fromiter((pos[int(k)] for k in sort_key), dtype=np.int64, count=len(sort_key))
        except KeyError as e:
            raise RuntimeError(f"{stem}: CSV row key {e} has no embedding") from None
        parts.append(x[idx].astype(np.float64))
        blocks[g] = (start, start + x.shape[1])
        start += x.shape[1]
    return np.concatenate(parts, axis=1), blocks


def load_lic(data_dir: Path | str, gnns=ALL_GNNS, target: str = "eform") -> Dataset:
    """Layered intercalation compounds (9024 host + intercalant structures).

    target: 'eform' (formation energy per atom) or 'efermi' (Fermi energy, eV).
    """
    d = Path(data_dir) / "lic"
    meta = pd.read_csv(d / "LC_database_rev_with_cif.csv", usecols=lambda c: c != "cif")
    meta["n_atoms"] = meta["Host_composition"].map(count_atoms) + 1   # host + one intercalant atom
    meta["eform"] = meta["Ef (eV/f.u.)"] / meta["n_atoms"]
    meta = meta.rename(columns={"efermi (eV)": "efermi"})
    if target not in ("eform", "efermi"):
        raise ValueError("LIC target must be 'eform' or 'efermi'")
    meta = meta.loc[meta[target].notna()].sort_values("sort").reset_index(drop=True)
    stems = {g: f"{GNN_STEMS[g]}_lc_data_10000" for g in GNN_STEMS}
    X, blocks = _load_embeddings(d / "embeddings", stems, gnns, meta["sort"].to_numpy())
    meta["group"] = meta["Host_composition"]
    unit = "eV/atom" if target == "eform" else "eV"
    return Dataset("lic", X, meta[target].to_numpy(float), meta, target, unit, blocks)


def load_aoi(data_dir: Path | str, gnns=ALL_GNNS, features: str = "gnn") -> Dataset:
    """Alloys of Interest from the Materials Project (MP18 train, MP21 additions as test).

    features: 'gnn' (pretrained embeddings) or 'matminer' (the 273 hand-built descriptors
    shipped in the CSV, used as the classical-descriptor baseline).
    """
    d = Path(data_dir) / "aoi"
    cols = None if features == "matminer" else ["material_id", "formula_pretty", "formation_energy_per_atom",
                                                "nelements", "nsites", "dataset"]
    meta = pd.read_csv(d / "AoI_with_timestamp.csv", usecols=cols)
    meta["sort"] = np.arange(1, len(meta) + 1)
    meta["group"] = meta["formula_pretty"]
    if features == "gnn":
        stems = {g: f"{GNN_STEMS[g]}_mp_aoi" for g in GNN_STEMS}
        X, blocks = _load_embeddings(d / "embeddings", stems, gnns, meta["sort"].to_numpy())
    elif features == "matminer":
        desc = [c for c in meta.columns if c not in _AOI_META + ["sort", "group"]]
        Xd = meta[desc].apply(pd.to_numeric, errors="coerce").astype(float)
        keep = Xd.notna().all(axis=1).to_numpy()          # paper drops the 26 rows with missing descriptors
        meta, Xd = meta.loc[keep].reset_index(drop=True), Xd.loc[keep]
        X, blocks = Xd.to_numpy(), {"matminer": (0, Xd.shape[1])}
        meta = meta[["material_id", "formula_pretty", "formation_energy_per_atom", "nelements", "nsites",
                     "dataset", "sort", "group"]]
    else:
        raise ValueError("features must be 'gnn' or 'matminer'")
    return Dataset("aoi", X, meta["formation_energy_per_atom"].to_numpy(float), meta, "eform", "eV/atom", blocks)


def load_dataset(name: str, data_dir: Path | str, **kw) -> Dataset:
    if name == "lic":
        return load_lic(data_dir, **kw)
    if name == "aoi":
        return load_aoi(data_dir, **kw)
    raise ValueError(f"unknown dataset {name!r}")


def load_custom(csv: Path | str, embeddings: list[Path | str], target: str, group: str | None = None,
                unit: str = "") -> Dataset:
    """Your own data: a CSV (row order = embedding row order) plus one .npy per feature block."""
    meta = pd.read_csv(csv)
    parts, blocks, start = [], {}, 0
    for p in embeddings:
        x = np.load(p).astype(np.float64)
        if len(x) != len(meta):
            raise ValueError(f"{p}: {len(x)} rows but CSV has {len(meta)}")
        parts.append(x)
        blocks[Path(p).stem] = (start, start + x.shape[1])
        start += x.shape[1]
    if group:
        meta["group"] = meta[group]
    return Dataset(Path(csv).stem, np.concatenate(parts, 1), meta[target].to_numpy(float), meta, target, unit, blocks)
