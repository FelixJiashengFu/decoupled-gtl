"""The three OC20 S2EF feature extractors, with parameter names that match the released checkpoints.

Embeddings are taken before each network's energy head and mean-pooled over atoms:
  cgcnn      512  node features after the 3 graph convolutions
  schnet    1024  node features after the 5 interaction blocks
  dimenetpp  768  edge messages after the embedding block and each of the 3 interaction
                  blocks (4 x 192), averaged onto atoms, then over atoms
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .graph import triplets

CHECKPOINTS = {"cgcnn": "cgcnn_all.pt", "schnet": "schnet_all_large.pt", "dimenetpp": "dimenetpp_all.pt"}
DIMS = {"cgcnn": 512, "schnet": 1024, "dimenetpp": 768}


def _scatter_mean(x, index, n):
    out = torch.zeros(n, x.shape[1], dtype=x.dtype).index_add_(0, index, x)
    cnt = torch.zeros(n, dtype=x.dtype).index_add_(0, index, torch.ones(len(index), dtype=x.dtype))
    return out / cnt.clamp(min=1).unsqueeze(1)


class GaussianSmearing(nn.Module):
    def __init__(self, start, stop, n):
        super().__init__()
        offset = torch.linspace(start, stop, n)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, d):
        return torch.exp(self.coeff * (d.view(-1, 1) - self.offset.view(1, -1)) ** 2)


class CGCNNConv(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.lin1 = nn.Linear(2 * node_dim + edge_dim, 2 * node_dim)
        self.bn1 = nn.BatchNorm1d(2 * node_dim)
        self.ln1 = nn.LayerNorm(node_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        z = self.bn1(self.lin1(torch.cat([x[dst], x[src], edge_attr], 1)))
        gate, core = z.chunk(2, 1)
        msg = torch.sigmoid(gate) * nn.functional.softplus(core)
        agg = torch.zeros_like(x).index_add_(0, dst, msg)
        return nn.functional.softplus(self.ln1(agg) + x)


class CGCNN(nn.Module):
    def __init__(self, atom_embedding_size=512, num_graph_conv_layers=3, fc_feat_size=128, num_fc_layers=3,
                 cutoff=6.0, num_gaussians=100):
        super().__init__()
        table = json.loads((Path(__file__).parent / "cgcnn_atom_init.json").read_text())
        emb = torch.zeros(100, len(table["1"]))
        for i in range(100):
            emb[i] = torch.tensor(table[str(i + 1)], dtype=torch.float32)
        self.register_buffer("embedding", emb, persistent=False)      # fixed k-hot element descriptors
        self.embedding_fc = nn.Linear(emb.shape[1], atom_embedding_size)
        self.convs = nn.ModuleList(CGCNNConv(atom_embedding_size, num_gaussians)
                                   for _ in range(num_graph_conv_layers))
        self.conv_to_fc = nn.Sequential(nn.Linear(atom_embedding_size, fc_feat_size), nn.Softplus())
        layers = []
        for _ in range(num_fc_layers - 1):
            layers += [nn.Linear(fc_feat_size, fc_feat_size), nn.Softplus()]
        self.fcs = nn.Sequential(*layers)
        self.fc_out = nn.Linear(fc_feat_size, 1)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

    def embed(self, g):
        if int(g["z"].max()) > 100:
            raise ValueError("CGCNN element table covers Z <= 100")
        h = self.embedding_fc(self.embedding[g["z"] - 1])
        ea = self.distance_expansion(g["dist"])
        for conv in self.convs:
            h = conv(h, g["edge_index"], ea)
        return h.mean(0)


def _schnet():
    from torch_geometric.nn import SchNet
    return SchNet(hidden_channels=1024, num_filters=256, num_interactions=5, num_gaussians=200, cutoff=6.0)


def _dimenetpp():
    from torch_geometric.nn.models import DimeNetPlusPlus
    return DimeNetPlusPlus(hidden_channels=192, out_channels=1, num_blocks=3, int_emb_size=64, basis_emb_size=8,
                           out_emb_channels=192, num_spherical=7, num_radial=6, cutoff=6.0, envelope_exponent=5,
                           num_before_skip=1, num_after_skip=2, num_output_layers=3)


class Extractor(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.net = {"cgcnn": CGCNN, "schnet": _schnet, "dimenetpp": _dimenetpp}[name]()
        self.eval()

    @torch.no_grad()
    def forward(self, g: dict) -> torch.Tensor:
        net, n = self.net, g["z"].numel()
        if self.name == "cgcnn":
            return net.embed(g)
        if self.name == "schnet":
            h = net.embedding(g["z"])
            ea = net.distance_expansion(g["dist"])
            for block in net.interactions:
                h = h + block(h, g["edge_index"], g["dist"], ea)
            return h.mean(0)
        src, dst = g["edge_index"]                         # dimenet++: messages live on edges j -> i
        idx_kj, idx_ji, angle = triplets(g)
        rbf = net.rbf(g["dist"])
        sbf = net.sbf(g["dist"], angle, idx_kj)
        x = net.emb(g["z"], rbf, dst, src)
        out = [_scatter_mean(x, dst, n).mean(0)]
        for block in net.interaction_blocks:
            x = block(x, rbf, sbf, idx_kj, idx_ji)
            out.append(_scatter_mean(x, dst, n).mean(0))
        return torch.cat(out)

    def load_checkpoint(self, path) -> dict:
        """Load OC20 weights. Returns the keys that did not line up, and raises if any learned
        parameter of the feature-extraction path is missing."""
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:                                   # OC20 files also pickle their config
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        clean = {}
        for k, v in state.items():
            while k.startswith("module."):
                k = k[len("module."):]
            clean[k] = v
        own = self.net.state_dict()
        shape_bad = [k for k, v in clean.items() if k in own and own[k].shape != v.shape]
        if shape_bad:
            raise RuntimeError(f"{self.name}: shape mismatch for {shape_bad[:5]} - wrong checkpoint?")
        res = self.net.load_state_dict(clean, strict=False)
        learned = {k for k, _ in self.net.named_parameters()}
        missing = [k for k in res.missing_keys if k in learned]
        if missing:
            raise RuntimeError(f"{self.name}: checkpoint lacks parameters {missing[:8]}")
        return {"missing_buffers": [k for k in res.missing_keys if k not in learned],
                "unexpected": list(res.unexpected_keys)}
