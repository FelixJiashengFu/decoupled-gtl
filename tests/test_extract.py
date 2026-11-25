"""Stage-1 checks that need no checkpoint: shapes and the physical invariances."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pmg = pytest.importorskip("pymatgen.core")

from dgtl.extract.graph import structure_to_graph, triplets
from dgtl.extract.models import DIMS, Extractor


def _structure():
    lat = pmg.Lattice.from_parameters(3.9, 4.1, 6.3, 90, 95, 110)
    return pmg.Structure(lat, ["Li", "Co", "O", "O"], [[0, 0, 0.5], [0, 0, 0], [0.3, 0.6, 0.25], [0.7, 0.4, 0.75]])


def test_graph_respects_cutoff_and_neighbour_cap():
    g = structure_to_graph(_structure(), cutoff=6.0, max_neighbors=50)
    assert float(g["dist"].max()) <= 6.0
    assert int(torch.bincount(g["edge_index"][1]).max()) <= 50
    assert torch.allclose(g["vec"].norm(dim=1), g["dist"], atol=1e-4)


def test_triplets_exclude_backtracking():
    g = structure_to_graph(_structure(), cutoff=4.0)
    kj, ji, angle = triplets(g)
    src, dst = g["edge_index"]
    assert (dst[kj] == src[ji]).all()                    # k -> j feeds j -> i
    assert float(angle.min()) >= 0 and float(angle.max()) <= np.pi + 1e-6
    back = (src[kj] == dst[ji]) & (g["image"][kj] == -g["image"][ji]).all(1)
    assert not back.any()


@pytest.mark.parametrize("name", ["cgcnn", "schnet", "dimenetpp"])
def test_embedding_shape_and_invariance(name):
    torch.manual_seed(0)
    m = Extractor(name)
    s = _structure()
    e = m(structure_to_graph(s))
    assert e.shape == (DIMS[name],) and torch.isfinite(e).all()
    sup = s.copy(); sup.make_supercell([2, 1, 1])        # same crystal, bigger cell
    e2 = m(structure_to_graph(sup))
    rot = s.copy()
    rot.apply_operation(pmg.SymmOp.from_axis_angle_and_translation([1, 2, 3], 37.0))   # rigid rotation
    e3 = m(structure_to_graph(rot))
    scale = float(e.abs().max()) + 1e-9
    assert float((e - e2).abs().max()) / scale < 1e-3
    assert float((e - e3).abs().max()) / scale < 1e-3
    far = s.copy(); far.scale_lattice(s.volume * 1.3)    # a different crystal must give a different vector
    assert float((e - m(structure_to_graph(far))).abs().max()) / scale > 1e-3
