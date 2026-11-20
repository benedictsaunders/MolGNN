"""Training setup for MiniMACE on QM9."""

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from molgnn.configuration import Configuration
from molgnn.loader import TARGET_EXTENSIVE, TARGETS, load_qm9_graphs
from molgnn.men import MiniMACE
from molgnn.normalization import TargetNormalizer

logger = logging.getLogger(__name__)

# Every setting can be overridden through configuration.json (see Configuration).
DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "r_cut": 5.0,
    "channels": 32,
    "lmax_hidden": 1,
    "lmax_sh": 3,
    "num_layers": 2,
    "num_basis": 8,
    "radial_neurons": [64, 64],
    "readout_hidden": 16,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 5e-7,
    "epochs": 200,
    "grad_clip": 10.0,
    "split": [0.8, 0.1, 0.1],
    "cache_path": "data/qm9_graphs.pt",
    "checkpoint_dir": "checkpoints",
}


def setting(key: str) -> Any:
    return Configuration.get(key, DEFAULTS[key])


def split_indices(n: int, fractions: Sequence[float], seed: int) -> tuple[list[int], list[int], list[int]]:
    """Deterministic random train/val/test split. Any rounding remainder goes to the test set."""
    if len(fractions) != 3 or any(f < 0 for f in fractions) or sum(fractions) > 1.0 + 1e-9:
        raise ValueError(f"Invalid split fractions {fractions}")
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    n_train = int(fractions[0] * n)
    n_val = int(fractions[1] * n)
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]


def average_num_neighbors(graphs: Sequence[Data]) -> float:
    edges = sum(g.edge_index.shape[1] for g in graphs)
    nodes = sum(g.num_nodes for g in graphs)
    return edges / max(nodes, 1)


def build_model(avg_num_neighbors: float = 1.0) -> MiniMACE:
    return MiniMACE(
        num_targets=len(TARGETS),
        extensive=TARGET_EXTENSIVE,
        channels=setting("channels"),
        lmax_hidden=setting("lmax_hidden"),
        lmax_sh=setting("lmax_sh"),
        num_layers=setting("num_layers"),
        r_cut=setting("r_cut"),
        num_basis=setting("num_basis"),
        radial_neurons=setting("radial_neurons"),
        readout_hidden=setting("readout_hidden"),
        avg_num_neighbors=avg_num_neighbors,
    )


@dataclass
class TrainingSetup:
    model: MiniMACE
    normalizer: TargetNormalizer
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    device: torch.device


def setup_training(graphs: Sequence[Data], device: str | torch.device = "cpu") -> TrainingSetup:
    """Split, fit the normaliser on the training split only, and build model/optimiser/loaders."""
    torch.manual_seed(setting("seed"))
    device = torch.device(device)
    train_idx, val_idx, test_idx = split_indices(len(graphs), setting("split"), setting("seed"))
    if not train_idx:
        raise ValueError("Training split is empty")
    train = [graphs[i] for i in train_idx]
    val = [graphs[i] for i in val_idx]
    test = [graphs[i] for i in test_idx]

    normalizer = TargetNormalizer(TARGET_EXTENSIVE).fit(train).to(device)
    model = build_model(average_num_neighbors(train)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=setting("lr"), weight_decay=setting("weight_decay")
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=10)
    bs = setting("batch_size")
    return TrainingSetup(
        model=model,
        normalizer=normalizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=DataLoader(train, batch_size=bs, shuffle=True),
        val_loader=DataLoader(val, batch_size=bs),
        test_loader=DataLoader(test, batch_size=bs),
        device=device,
    )


def train_step(s: TrainingSetup, batch: Data) -> float:
    s.model.train()
    batch = batch.to(s.device)
    target = s.normalizer.normalize(batch.y, batch)
    loss = nn.functional.mse_loss(s.model(batch), target)
    s.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(s.model.parameters(), setting("grad_clip"))
    s.optimizer.step()
    return loss.item()


def train_one_epoch(s: TrainingSetup) -> float:
    losses = [train_step(s, batch) for batch in s.train_loader]
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def evaluate(s: TrainingSetup, loader: DataLoader) -> dict[str, float]:
    """Normalised MSE loss plus per-target MAE in physical units (Hartree)."""
    s.model.eval()
    abs_err = torch.zeros(len(TARGETS), device=s.device, dtype=torch.float64)
    sq_err_norm = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(s.device)
        pred_norm = s.model(batch)
        sq_err_norm += ((pred_norm - s.normalizer.normalize(batch.y, batch)) ** 2).mean(1).sum().item()
        abs_err += (s.normalizer.denormalize(pred_norm, batch) - batch.y.to(torch.float64)).abs().sum(0)
        n += batch.num_graphs
    if n == 0:
        return {"loss": math.nan, **{f"mae_{t}": math.nan for t in TARGETS}}
    mae = (abs_err / n).tolist()
    return {"loss": sq_err_norm / n, **{f"mae_{t}": m for t, m in zip(TARGETS, mae)}}


def save_checkpoint(s: TrainingSetup, path: str | Path, epoch: int, metrics: dict[str, float]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "metrics": metrics,
            "avg_num_neighbors": s.model.interactions[0].avg_num_neighbors,
            "model": s.model.state_dict(),
            "normalizer": s.normalizer.state_dict(),
            "optimizer": s.optimizer.state_dict(),
            "scheduler": s.scheduler.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[MiniMACE, TargetNormalizer, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["avg_num_neighbors"]).to(device)
    model.load_state_dict(ckpt["model"])
    normalizer = TargetNormalizer(TARGET_EXTENSIVE).to(device)
    normalizer.load_state_dict(ckpt["normalizer"])
    return model, normalizer, ckpt


def fit(s: TrainingSetup, epochs: int, checkpoint_dir: Optional[str | Path] = None) -> dict[str, float]:
    best = math.inf
    val_metrics: dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(s)
        val_metrics = evaluate(s, s.val_loader)
        s.scheduler.step(val_metrics["loss"])
        logger.info("epoch %d train %.4f val %s", epoch, train_loss, val_metrics)
        if checkpoint_dir is not None and val_metrics["loss"] < best:
            best = val_metrics["loss"]
            save_checkpoint(s, Path(checkpoint_dir) / "best.pt", epoch, val_metrics)
    return val_metrics


def gpu_usable() -> bool:
    """
    True if a GPU is visible *and* this torch build can run kernels on it. A ROCm wheel can
    report `is_available()` for a card whose architecture it wasn't compiled for (e.g. gfx1201
    with rocm6.2 wheels), failing only at the first kernel launch with "invalid device function".
    """
    # ROCm builds of PyTorch expose AMD GPUs through the `cuda` device type.
    if not torch.cuda.is_available():
        return False
    try:
        (torch.ones(2, device="cuda") + 1).cpu()
        return True
    except RuntimeError:
        return False


def default_device() -> str:
    if gpu_usable():
        return "cuda"
    if torch.cuda.is_available():
        logger.warning(
            "GPU %s is visible but unusable with torch %s (compiled for %s); falling back to CPU",
            torch.cuda.get_device_name(0), torch.__version__, torch.cuda.get_arch_list(),
        )
    return "cpu"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="molgnn", description="Train a minimal MACE on QM9")
    parser.add_argument("--config", default=None, help="path to configuration.json")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="only use the first N molecules")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-cache", action="store_true", help="do not read/write the graph cache")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build data, model and optimiser, run one forward pass, then exit without training",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    if args.config is not None:
        Configuration.set_filepath(args.config)

    cache = None if args.no_cache else setting("cache_path")
    if cache is not None and args.limit is not None:
        cache = str(Path(cache).with_suffix(f".limit{args.limit}.pt"))
    graphs = load_qm9_graphs(setting("r_cut"), limit=args.limit, cache_path=cache)
    s = setup_training(graphs, device=args.device or default_device())

    n_params = sum(p.numel() for p in s.model.parameters())
    print(
        f"{len(graphs)} molecules | train/val/test {len(s.train_loader.dataset)}/"
        f"{len(s.val_loader.dataset)}/{len(s.test_loader.dataset)} | "
        f"{n_params} parameters | device {s.device}"
    )

    if args.dry_run:
        s.model.eval()
        with torch.no_grad():
            batch = next(iter(s.train_loader)).to(s.device)
            out = s.model(batch)
        print(f"dry run OK: forward pass on batch of {batch.num_graphs} -> {tuple(out.shape)}")
        return 0

    fit(s, args.epochs or setting("epochs"), checkpoint_dir=setting("checkpoint_dir"))
    print("test:", evaluate(s, s.test_loader))
    return 0
