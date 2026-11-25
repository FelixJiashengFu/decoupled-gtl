"""Periodic neighbour graphs and DimeNet triplets."""
from __future__ import annotations

import numpy as np
import torch


def structure_to_graph(structure, cutoff: float = 6.0, max_neighbors: int = 50) -> dict:
    """pymatgen Structure -> edges within `cutoff` under periodic boundary conditions.

    Edge e runs from a source atom j (possibly in a neighbouring cell) to a centre atom i.
    Each centre keeps its `max_neighbors` nearest sources, as the OC20 models were trained.
    """
    centre, source, image, dist = structure.get_neighbor_list(cutoff)
    keep = dist > 1e-8
    centre, source, image, dist = centre[keep], source[keep], image[keep], dist[keep]
    order = np.lexsort((dist, centre))                       # by centre, then by distance
    centre, source, image, dist = centre[order], source[order], image[order], dist[order]
    rank = np.arange(len(centre)) - np.searchsorted(centre, centre, side="left")
    k = rank < max_neighbors
    centre, source, image, dist = centre[k], source[k], image[k], dist[k]
    cell = np.asarray(structure.lattice.matrix)
    pos = np.asarray(structure.cart_coords)
    vec = pos[source] + image @ cell - pos[centre]           # centre -> source image
    return {
        "z": torch.tensor([s.specie.Z for s in structure], dtype=torch.long),
        "edge_index": torch.tensor(np.stack([source, centre]), dtype=torch.long),   # [source j, target i]
        "dist": torch.tensor(dist, dtype=torch.float32),
        "vec": torch.tensor(vec, dtype=torch.float32),
        "image": torch.tensor(image, dtype=torch.long),
    }


def triplets(g: dict):
    """All (k -> j -> i) edge pairs, excluding the edge that points straight back.

    Returns idx_kj, idx_ji (edge ids) and the angle between the two edge vectors.
    """
    src, dst = g["edge_index"]
    n_edges = src.numel()
    order = torch.argsort(dst, stable=True)                  # edges grouped by their target atom
    counts = torch.bincount(dst, minlength=g["z"].numel())
    start = torch.cumsum(counts, 0) - counts
    # for edge ji (j -> i): every edge kj whose target is j
    n_in = counts[src]                                       # number of edges arriving at j
    idx_ji = torch.repeat_interleave(torch.arange(n_edges), n_in)
    offs = torch.arange(int(n_in.sum())) - torch.repeat_interleave(torch.cumsum(n_in, 0) - n_in, n_in)
    idx_kj = order[start[src][idx_ji] + offs]
    # drop k == i in the same periodic image (the reverse of edge ji)
    back = (src[idx_kj] == dst[idx_ji]) & (g["image"][idx_kj] == -g["image"][idx_ji]).all(1)
    idx_kj, idx_ji = idx_kj[~back], idx_ji[~back]
    v_ji, v_kj = g["vec"][idx_ji], g["vec"][idx_kj]
    a = (v_ji * v_kj).sum(-1)
    b = torch.linalg.cross(v_ji, v_kj).norm(dim=-1)
    return idx_kj, idx_ji, torch.atan2(b, a)
