from pathlib import Path

import pytest
import torch

from molgnn.configuration import Configuration
from molgnn.loader import (
    TARGET_EXTENSIVE,
    TARGETS,
    get_qm9,
    load_qm9_graphs,
    parse_kaggle_qm9,
)


def test_parse_methane(fake_qm9_dir):
    atoms, data = parse_kaggle_qm9(fake_qm9_dir / "dsgdb9nsd_000001.xyz")
    assert atoms.get_chemical_formula() == "CH4"
    assert data.z.tolist() == [6, 1, 1, 1, 1]
    assert data.pos.shape == (5, 3)
    assert torch.allclose(data.pos, torch.tensor(atoms.positions, dtype=torch.float))
    assert data.q.tolist() == pytest.approx([-0.535689, 0.133921, 0.133922, 0.133923, 0.133923])
    # gap, U0, U, H, G in Hartree
    expected = [0.5048, -40.47893, -40.476062, -40.475117, -40.498597]
    assert data.y.shape == (1, len(TARGETS))
    assert data.y.dtype == torch.float64
    assert data.y[0].tolist() == pytest.approx(expected)
    assert data.cv.item() == pytest.approx(6.469)
    assert data.idx == 1


def test_parse_mathematica_exponent(fake_qm9_dir):
    _, data = parse_kaggle_qm9(fake_qm9_dir / "dsgdb9nsd_000003.xyz")
    assert data.z.tolist() == [8, 1, 1]
    assert data.pos[2, 2].item() == pytest.approx(1.5e-5)


def test_targets_metadata():
    assert len(TARGETS) == len(TARGET_EXTENSIVE)
    assert TARGET_EXTENSIVE[TARGETS.index("gap")] is False


def test_get_qm9_uses_configured_path(fake_qm9_dir, no_kaggle):
    mols = get_qm9(limit=10)
    # Sorted by GDB index, and the combined dsgdb9nsd.xyz file is ignored.
    assert [d.idx for _, d in mols] == [1, 3, 4]
    assert len(get_qm9(limit=2)) == 2


def test_get_qm9_checks_full_dataset_size(fake_qm9_dir, no_kaggle):
    with pytest.raises(RuntimeError, match="Expected 133885"):
        get_qm9()


def test_missing_path_triggers_download(tmp_path, monkeypatch, fake_qm9_dir):
    import kagglehub

    Configuration.set("dataset_path", str(tmp_path / "does_not_exist"))
    calls = []
    monkeypatch.setattr(kagglehub, "dataset_download", lambda *a, **k: calls.append(a) or str(fake_qm9_dir))
    assert len(get_qm9(limit=1)) == 1
    assert calls == [("zaharch/quantum-machine-9-aka-qm9",)]
    assert Configuration.get("dataset_path") == str(fake_qm9_dir)


def test_load_graphs_builds_edges_and_caches(fake_qm9_dir, no_kaggle, tmp_path, monkeypatch):
    cache = tmp_path / "cache" / "graphs.pt"
    graphs = load_qm9_graphs(r_cut=5.0, limit=3, cache_path=cache)
    assert cache.exists()
    methane = graphs[0]
    assert methane.edge_index.shape == (2, 5 * 4)  # fully connected, no self loops

    # Second call must come from the cache, not the parser.
    import molgnn.loader as loader

    monkeypatch.setattr(loader, "get_qm9", lambda **k: pytest.fail("cache not used"))
    again = load_qm9_graphs(r_cut=5.0, limit=3, cache_path=cache)
    assert torch.equal(again[0].edge_index, methane.edge_index)


def test_cache_rebuilt_on_different_cutoff(fake_qm9_dir, no_kaggle, tmp_path):
    cache = tmp_path / "graphs.pt"
    load_qm9_graphs(r_cut=5.0, limit=3, cache_path=cache)
    small = load_qm9_graphs(r_cut=1.2, limit=3, cache_path=cache)
    # With a 1.2 Å cutoff only the C-H bonds survive in methane.
    assert small[0].edge_index.shape[1] == 8


QM9_PATH = Path.home() / ".cache/kagglehub/datasets/zaharch/quantum-machine-9-aka-qm9/versions/1"


@pytest.mark.qm9
@pytest.mark.skipif(not QM9_PATH.exists(), reason="real QM9 not downloaded")
def test_real_qm9_first_molecules(no_kaggle):
    Configuration.set("dataset_path", str(QM9_PATH))
    graphs = load_qm9_graphs(r_cut=5.0, limit=50)
    assert len(graphs) == 50
    for g in graphs:
        assert set(g.z.tolist()) <= {1, 6, 7, 8, 9}
        assert torch.isfinite(g.y).all() and torch.isfinite(g.pos).all()
        assert g.y[0, 0] > 0  # the HOMO-LUMO gap is positive
