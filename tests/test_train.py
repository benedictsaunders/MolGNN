import math

import pytest
import torch

from molgnn.configuration import Configuration
from molgnn.loader import TARGETS
from molgnn.train import (
    DEFAULTS,
    average_num_neighbors,
    build_model,
    evaluate,
    fit,
    load_checkpoint,
    main,
    save_checkpoint,
    setting,
    setup_training,
    split_indices,
    train_step,
)

from conftest import random_molecule


@pytest.fixture
def small_config():
    for k, v in {"channels": 8, "batch_size": 4, "num_layers": 2, "lr": 5e-3}.items():
        Configuration.set(k, v)


@pytest.fixture
def dataset():
    g = torch.Generator().manual_seed(7)
    return [random_molecule(int(n), g) for n in torch.randint(2, 9, (30,), generator=g)]


def test_split_is_disjoint_complete_and_deterministic():
    tr, va, te = split_indices(103, [0.8, 0.1, 0.1], seed=1)
    assert len(tr) == 82 and len(va) == 10 and len(te) == 11
    assert sorted(tr + va + te) == list(range(103))
    assert (tr, va, te) == split_indices(103, [0.8, 0.1, 0.1], seed=1)
    assert tr != split_indices(103, [0.8, 0.1, 0.1], seed=2)[0]


@pytest.mark.parametrize("bad", [[0.9, 0.2, 0.1], [0.5, 0.5], [-0.1, 0.5, 0.5]])
def test_split_rejects_invalid_fractions(bad):
    with pytest.raises(ValueError):
        split_indices(10, bad, seed=0)


def test_settings_fall_back_to_defaults_and_respect_config():
    assert setting("r_cut") == DEFAULTS["r_cut"]
    Configuration.set("r_cut", 4.0)
    assert setting("r_cut") == 4.0
    assert build_model().r_cut == 4.0


def test_average_num_neighbors(molecules):
    expected = sum(m.edge_index.shape[1] for m in molecules) / sum(m.num_nodes for m in molecules)
    assert average_num_neighbors(molecules) == pytest.approx(expected)


def test_setup_uses_training_split_only(dataset, small_config):
    s = setup_training(dataset)
    n = len(dataset)
    assert len(s.train_loader.dataset) + len(s.val_loader.dataset) + len(s.test_loader.dataset) == n
    expected_nn = average_num_neighbors(s.train_loader.dataset)
    assert s.model.interactions[0].avg_num_neighbors == pytest.approx(expected_nn)


def test_evaluate_reports_every_target(dataset, small_config):
    metrics = evaluate(setup_training(dataset), setup_training(dataset).val_loader)
    assert set(metrics) == {"loss"} | {f"mae_{t}" for t in TARGETS}
    assert all(math.isfinite(v) and v >= 0 for v in metrics.values())


def test_can_overfit_a_single_batch(dataset, small_config):
    """The PoS sanity check: loss on one fixed batch must fall substantially."""
    s = setup_training(dataset)
    batch = next(iter(s.train_loader))
    first = train_step(s, batch)
    for _ in range(150):
        last = train_step(s, batch)
    assert last < 0.2 * first, (first, last)


def test_fit_writes_checkpoint_and_it_reloads(dataset, small_config, tmp_path):
    s = setup_training(dataset)
    metrics = fit(s, epochs=2, checkpoint_dir=tmp_path)
    assert (tmp_path / "best.pt").exists()
    assert math.isfinite(metrics["loss"])

    model, normalizer, ckpt = load_checkpoint(tmp_path / "best.pt")
    assert ckpt["epoch"] in (1, 2)
    model.eval()
    s.model.eval()
    batch = next(iter(s.test_loader))
    # best.pt may be from epoch 1, so compare against a fresh save of the current state instead
    save_checkpoint(s, tmp_path / "now.pt", 2, metrics)
    model, normalizer, _ = load_checkpoint(tmp_path / "now.pt")
    model.eval()
    assert torch.allclose(model(batch), s.model(batch), atol=1e-6)
    assert torch.equal(normalizer.e0, s.normalizer.e0)


def test_cli_dry_run_does_not_train(fake_qm9_dir, no_kaggle, small_config, monkeypatch, capsys, tmp_path):
    Configuration.set("split", [0.67, 0.0, 0.33])
    Configuration.set("checkpoint_dir", str(tmp_path / "ckpt"))
    import molgnn.train as train

    monkeypatch.setattr(train, "fit", lambda *a, **k: pytest.fail("dry run must not train"))
    assert main(["--dry-run", "--no-cache", "--limit", "3", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "3 molecules" in out and "dry run OK" in out
    assert not (tmp_path / "ckpt").exists()


def test_cli_trains_end_to_end_on_tiny_data(fake_qm9_dir, no_kaggle, small_config, tmp_path, capsys):
    Configuration.set("split", [0.67, 0.0, 0.33])
    Configuration.set("checkpoint_dir", str(tmp_path / "ckpt"))
    Configuration.set("cache_path", str(tmp_path / "graphs.pt"))
    assert main(["--epochs", "1", "--limit", "3", "--device", "cpu"]) == 0
    assert "test:" in capsys.readouterr().out
    assert (tmp_path / "graphs.limit3.pt").exists()
