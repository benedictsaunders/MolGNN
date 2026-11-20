import math

import pytest
import torch
from torch_geometric.data import Batch

from molgnn.graph import (
    BesselBasis,
    PolynomialCutoff,
    RadialEmbedding,
    RadiusGraph,
    edge_vectors,
    radius_graph,
)


def brute_force_edges(pos, r_cut):
    n = pos.shape[0]
    return {
        (j, i)
        for i in range(n)
        for j in range(n)
        if i != j and (pos[i] - pos[j]).norm().item() < r_cut
    }


@pytest.mark.parametrize("r_cut", [0.5, 1.5, 3.0, 100.0])
def test_radius_graph_matches_brute_force(r_cut):
    pos = torch.randn(12, 3, generator=torch.Generator().manual_seed(0))
    ei = radius_graph(pos, r_cut)
    assert set(map(tuple, ei.T.tolist())) == brute_force_edges(pos, r_cut)


def test_radius_graph_symmetric_and_no_self_loops():
    pos = torch.randn(10, 3)
    ei = radius_graph(pos, 2.0)
    edges = set(map(tuple, ei.T.tolist()))
    assert all((b, a) in edges for a, b in edges)
    assert not (ei[0] == ei[1]).any()


def test_single_atom_has_no_edges():
    assert radius_graph(torch.zeros(1, 3), 5.0).shape == (2, 0)


def test_edge_vectors_point_from_receiver_to_source():
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])
    ei = torch.tensor([[1], [0]])  # source 1 -> receiver 0
    vec, length = edge_vectors(pos, ei)
    assert torch.equal(vec, torch.tensor([[1.0, 2.0, 2.0]]))
    assert length.item() == pytest.approx(3.0)


def test_batching_offsets_edges(molecules):
    batch = Batch.from_data_list(molecules)
    offset = 0
    for k, mol in enumerate(molecules):
        sel = batch.batch[batch.edge_index[1]] == k
        assert torch.equal(batch.edge_index[:, sel] - offset, mol.edge_index)
        offset += mol.num_nodes
    # No edge crosses molecules.
    assert torch.equal(batch.batch[batch.edge_index[0]], batch.batch[batch.edge_index[1]])


def test_radius_graph_transform(molecules):
    mol = molecules[0].clone()
    del mol.edge_index
    assert RadiusGraph(5.0)(mol).edge_index.shape[0] == 2


def test_polynomial_cutoff_boundary_behaviour():
    cutoff = PolynomialCutoff(r_cut=5.0, p=6)
    r = torch.tensor([0.0, 2.5, 4.999, 5.0, 7.0], requires_grad=True)
    u = cutoff(r)
    assert u[0].item() == pytest.approx(1.0)
    assert 0 < u[1].item() < 1
    assert u[2].abs().item() < 1e-6
    assert u[3].item() == 0.0 and u[4].item() == 0.0
    (grad,) = torch.autograd.grad(u.sum(), r)
    assert grad[2].abs().item() < 1e-4  # smooth: derivative vanishes at r_cut


def test_polynomial_cutoff_is_monotone():
    u = PolynomialCutoff(5.0)(torch.linspace(0, 5, 200))
    assert (u[1:] <= u[:-1] + 1e-7).all()


def test_bessel_basis_values():
    basis = BesselBasis(r_cut=5.0, num_basis=4)
    r = torch.tensor([[1.3]])
    expected = [math.sqrt(2 / 5) * math.sin(n * math.pi * 1.3 / 5) / 1.3 for n in range(1, 5)]
    assert basis(r)[0].tolist() == pytest.approx(expected, rel=1e-5)


def test_radial_embedding_vanishes_at_cutoff():
    emb = RadialEmbedding(r_cut=4.0, num_basis=6)
    out = emb(torch.tensor([[1.0], [4.0], [6.0]]))
    assert out.shape == (3, 6)
    assert torch.all(out[1:] == 0)
    assert torch.isfinite(out).all()
