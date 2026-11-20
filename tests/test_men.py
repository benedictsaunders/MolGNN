import pytest
import torch
from e3nn import o3
from torch_geometric.data import Batch, Data

from molgnn.graph import RadialEmbedding, edge_vectors, radius_graph
from molgnn.train import gpu_usable
from molgnn.men import EquivariantInteraction, MiniMACE, ProductBasis, channelwise_tp_irreps

EXTENSIVE = [False, True, True, True, True]
ATOL = 1e-4


def make_model(**kwargs) -> MiniMACE:
    torch.manual_seed(0)
    defaults = dict(num_targets=5, extensive=EXTENSIVE, channels=8, num_layers=2, avg_num_neighbors=4.0)
    defaults.update(kwargs)
    return MiniMACE(**defaults).eval()


def random_rotation(seed: int = 0, proper: bool = True) -> torch.Tensor:
    torch.manual_seed(seed)
    R = o3.rand_matrix()
    return R if proper else -R


def transformed(batch: Batch, R: torch.Tensor, t: torch.Tensor | None = None) -> Batch:
    out = batch.clone()
    out.pos = batch.pos @ R.T
    if t is not None:
        out.pos = out.pos + t
    return out


# ---------------------------------------------------------------- building blocks


def test_channelwise_tp_irreps_only_keeps_targets():
    irreps_mid, instructions = channelwise_tp_irreps(
        o3.Irreps("4x0e+4x1o"), o3.Irreps.spherical_harmonics(2), o3.Irreps("4x0e+4x1o")
    )
    assert {ir for _, ir in irreps_mid} <= {o3.Irrep("0e"), o3.Irrep("1o")}
    assert all(mul == 4 for mul, _ in irreps_mid)
    assert len(instructions) == len(irreps_mid)
    # instructions must point at valid, distinct output slots
    assert sorted(k for _, _, k, _, _ in instructions) == list(range(len(irreps_mid)))


@pytest.mark.parametrize("proper", [True, False])
def test_interaction_is_equivariant(proper):
    torch.manual_seed(0)
    irreps_in = o3.Irreps("4x0e+4x1o+2x2e")
    irreps_sh = o3.Irreps.spherical_harmonics(3)
    irreps_out = o3.Irreps("4x0e+4x1o+4x2e")
    layer = EquivariantInteraction(irreps_in, irreps_sh, irreps_out, num_basis=6, avg_num_neighbors=3.0)
    sh = o3.SphericalHarmonics(irreps_sh, normalize=True, normalization="component")
    radial = RadialEmbedding(5.0, 6)

    pos = torch.randn(7, 3)
    ei = radius_graph(pos, 5.0)
    h = irreps_in.randn(7, -1)

    def run(p, feats):
        vec, length = edge_vectors(p, ei)
        return layer(feats, ei, sh(vec), radial(length))

    R = random_rotation(1, proper)
    out_rot = run(pos @ R.T, h @ irreps_in.D_from_matrix(R).T)
    rot_out = run(pos, h) @ irreps_out.D_from_matrix(R).T
    assert torch.allclose(out_rot, rot_out, atol=ATOL)


def test_product_basis_is_equivariant():
    torch.manual_seed(0)
    irreps = o3.Irreps("4x0e+4x1o+4x2e")
    block = ProductBasis(irreps, irreps)
    A = irreps.randn(10, -1)
    D = irreps.D_from_matrix(random_rotation(2))
    assert torch.allclose(block(A @ D.T), block(A) @ D.T, atol=ATOL)


def test_product_basis_is_nonlinear():
    torch.manual_seed(0)
    irreps = o3.Irreps("4x0e+4x1o")
    block = ProductBasis(irreps, irreps)
    A = irreps.randn(3, -1)
    assert not torch.allclose(block(2 * A), 2 * block(A), atol=1e-3)


# ---------------------------------------------------------------- full model


def test_output_shape_and_finite(batch):
    out = make_model()(batch)
    assert out.shape == (batch.num_graphs, 5)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("proper", [True, False], ids=["rotation", "improper-rotation"])
def test_model_is_e3_invariant(batch, proper):
    model = make_model()
    R = random_rotation(3, proper)
    t = torch.tensor([3.0, -1.0, 0.5])
    assert torch.allclose(model(transformed(batch, R, t)), model(batch), atol=ATOL)


@pytest.mark.parametrize("lmax_hidden, lmax_sh", [(0, 1), (1, 2), (2, 3)])
def test_invariance_across_hyperparameters(batch, lmax_hidden, lmax_sh):
    model = make_model(lmax_hidden=lmax_hidden, lmax_sh=lmax_sh, num_layers=3)
    R = random_rotation(4)
    assert torch.allclose(model(transformed(batch, R)), model(batch), atol=ATOL)


def test_model_is_permutation_invariant(molecules):
    model = make_model()
    mol = molecules[2]
    perm = torch.randperm(mol.num_nodes, generator=torch.Generator().manual_seed(0))
    inv = torch.argsort(perm)
    permuted = Data(z=mol.z[perm], pos=mol.pos[perm], edge_index=inv[mol.edge_index])
    assert torch.allclose(model(permuted), model(mol), atol=ATOL)


def test_batched_equals_individual(molecules, batch):
    model = make_model()
    batched = model(batch)
    individual = torch.cat([model(m) for m in molecules])
    assert torch.allclose(batched, individual, atol=ATOL)


def test_extensive_and_intensive_pooling(molecules):
    """Two far-apart copies of a molecule: extensive targets double, intensive ones stay put."""
    model = make_model()
    mol = molecules[1]
    far = mol.pos + torch.tensor([100.0, 0.0, 0.0])
    pos = torch.cat([mol.pos, far])
    dimer = Data(z=torch.cat([mol.z, mol.z]), pos=pos, edge_index=radius_graph(pos, 5.0))
    single, double = model(mol)[0], model(dimer)[0]
    ext = torch.tensor(EXTENSIVE)
    assert torch.allclose(double[ext], 2 * single[ext], atol=ATOL)
    assert torch.allclose(double[~ext], single[~ext], atol=ATOL)


def test_isolated_atoms_are_finite():
    model = make_model()
    mol = Data(z=torch.tensor([6, 1]), pos=torch.tensor([[0.0, 0, 0], [10.0, 0, 0]]))
    mol.edge_index = radius_graph(mol.pos, 5.0)
    assert mol.edge_index.shape[1] == 0
    assert torch.isfinite(model(mol)).all()


def test_output_is_smooth_across_cutoff():
    """Moving a neighbour across r_cut must not cause a jump (the envelope switches it off)."""
    model = make_model(r_cut=3.0)

    def energy(d):
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [d, 0.5, 0.0]])
        return model(Data(z=torch.tensor([6, 1, 8]), pos=pos, edge_index=radius_graph(pos, 3.0)))

    # Atom 2 is within 3 Å of atom 0 just below d≈2.958 and outside just above.
    d_c = (3.0**2 - 0.5**2) ** 0.5
    inside, outside = energy(d_c - 1e-4), energy(d_c + 1e-4)
    assert torch.allclose(inside, outside, atol=1e-4)


def test_depends_on_geometry_and_species(molecules):
    model = make_model()
    mol = molecules[1]
    base = model(mol)
    stretched = Data(z=mol.z, pos=1.1 * mol.pos, edge_index=radius_graph(1.1 * mol.pos, 5.0))
    swapped = Data(z=torch.where(mol.z == 1, 9, 1), pos=mol.pos, edge_index=mol.edge_index)
    assert not torch.allclose(model(stretched), base, atol=1e-4)
    assert not torch.allclose(model(swapped), base, atol=1e-4)


def test_gradients_reach_every_parameter(batch):
    model = make_model().train()
    model(batch).pow(2).sum().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert missing == []
    # species not present in the batch legitimately get zero embedding gradient; others must not
    present = batch.z.unique()
    assert (model.species_embedding.weight.grad[present].abs().sum(1) > 0).all()


def test_forces_are_equivariant(batch):
    """-dE/dr is a vector field, so it should rotate with the molecule."""
    model = make_model()
    R = random_rotation(5)

    def forces(b):
        b = b.clone()
        b.pos.requires_grad_(True)
        (g,) = torch.autograd.grad(model(b)[:, 1].sum(), b.pos)
        return -g

    assert torch.allclose(forces(transformed(batch, R)), forces(batch) @ R.T, atol=ATOL)


def test_rejects_mismatched_extensive():
    with pytest.raises(ValueError):
        MiniMACE(num_targets=3, extensive=[True])


@pytest.mark.skipif(not gpu_usable(), reason="no GPU, or this torch build has no kernels for its architecture")
def test_runs_on_gpu(batch):
    model = make_model().to("cuda")
    out = model(batch.to("cuda"))
    assert out.device.type == "cuda"
    assert torch.allclose(out.cpu(), make_model()(batch.cpu()), atol=1e-3)
