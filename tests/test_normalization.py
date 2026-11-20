import pytest
import torch
from torch_geometric.data import Batch, Data

from molgnn.normalization import TargetNormalizer, composition_matrix

E0 = {1: -0.5, 6: -37.8, 7: -54.5, 8: -75.0, 9: -99.7}


def synthetic(n: int, seed: int = 0, noise: float = 0.0) -> list[Data]:
    g = torch.Generator().manual_seed(seed)
    species = torch.tensor(list(E0))
    out = []
    for _ in range(n):
        z = species[torch.randint(0, 5, (int(torch.randint(2, 10, (1,), generator=g)),), generator=g)]
        energy = sum(E0[int(a)] for a in z) + noise * torch.randn(1, generator=g).item()
        gap = 0.25 + 0.05 * torch.randn(1, generator=g).item()
        out.append(Data(z=z, pos=torch.zeros(len(z), 3), y=torch.tensor([[gap, energy]], dtype=torch.float64)))
    return out


def test_composition_matrix():
    z = torch.tensor([1, 1, 6, 8, 1])
    batch = torch.tensor([0, 0, 0, 1, 1])
    C = composition_matrix(z, batch, num_graphs=2, num_species=10)
    assert C[0, 1] == 2 and C[0, 6] == 1 and C[1, 8] == 1 and C[1, 1] == 1
    assert C.sum() == 5


def test_recovers_atomic_reference_energies():
    norm = TargetNormalizer([False, True]).fit(synthetic(200))
    for z, e in E0.items():
        assert norm.e0[z, 1].item() == pytest.approx(e, abs=1e-4)
    assert torch.all(norm.e0[:, 0] == 0)  # intensive target gets no E0
    assert norm.mean[1] == 0
    assert norm.mean[0].item() == pytest.approx(0.25, abs=0.02)


def test_normalized_targets_are_standardised():
    data = synthetic(300, noise=0.01)
    norm = TargetNormalizer([False, True]).fit(data)
    y_norm = norm.normalize(Batch.from_data_list(data).y, Batch.from_data_list(data))
    assert y_norm.dtype == torch.float32
    # E0 is fitted without an intercept, so residuals are orthogonal to the composition
    # columns but need not average to exactly zero.
    assert y_norm.mean(0).abs().max() < 2e-2
    assert y_norm.std(0, unbiased=False).tolist() == pytest.approx([1.0, 1.0], abs=1e-3)


def test_roundtrip(molecules):
    norm = TargetNormalizer([False, True, True, True, True]).fit(molecules)
    b = Batch.from_data_list(molecules)
    assert torch.allclose(norm.denormalize(norm.normalize(b.y, b), b), b.y.double(), atol=1e-5)


def test_precision_survives_large_offsets():
    """A 1 meV (~4e-5 Ha) difference on a -400 Ha total must not be lost to rounding."""
    data = synthetic(100, noise=0.01)
    norm = TargetNormalizer([False, True]).fit(data)
    mol = data[0]
    shifted = mol.y.clone()
    shifted[0, 1] += 4e-5
    delta = (norm.normalize(shifted, mol) - norm.normalize(mol.y, mol))[0, 1] * norm.std[1]
    assert delta.item() == pytest.approx(4e-5, rel=1e-3)


def test_normalizer_buffers_stay_float64_on_move():
    norm = TargetNormalizer([False, True]).fit(synthetic(20)).to("cpu")
    assert norm.e0.dtype == norm.std.dtype == torch.float64


def test_unbatched_baseline_matches_batched(molecules):
    norm = TargetNormalizer([False, True, True, True, True]).fit(molecules)
    b = Batch.from_data_list(molecules)
    stacked = torch.cat([norm.baseline(m) for m in molecules])
    assert torch.allclose(norm.baseline(b), stacked)


def test_state_dict_roundtrip():
    a = TargetNormalizer([False, True]).fit(synthetic(50))
    b = TargetNormalizer([False, True])
    b.load_state_dict(a.state_dict())
    assert torch.equal(a.e0, b.e0) and torch.equal(a.std, b.std)
