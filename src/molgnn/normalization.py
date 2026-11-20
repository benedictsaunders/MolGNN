"""Target normalisation: per-element reference energies for extensive targets, mean/std otherwise."""

from typing import Iterable, Sequence

import torch
import torch.nn as nn
from torch_geometric.data import Data


def composition_matrix(z: torch.Tensor, batch: torch.Tensor, num_graphs: int, num_species: int) -> torch.Tensor:
    """(num_graphs, num_species) matrix counting atoms of each atomic number in each graph."""
    counts = torch.zeros((num_graphs, num_species), dtype=torch.get_default_dtype(), device=z.device)
    counts.index_put_((batch, z), torch.ones_like(z, dtype=counts.dtype), accumulate=True)
    return counts


class TargetNormalizer(nn.Module):
    """
    y_norm = (y - baseline(composition)) / std

    For extensive targets (energies) the baseline is sum_i E0[Z_i], where E0 is fitted by
    least squares on the training set (MACE's "atomic reference energies"). For intensive
    targets (e.g. the HOMO-LUMO gap) the baseline is the training mean.
    """

    def __init__(self, extensive: Sequence[bool], num_species: int = 10) -> None:
        super().__init__()
        num_targets = len(extensive)
        self.num_species = num_species
        self.register_buffer("extensive", torch.tensor(list(extensive), dtype=torch.bool))
        # Kept in float64 (and preserved by casting only via .to(device)): the baseline is
        # subtracted from totals of order 10^2 Ha to leave residuals of order 10^-2 Ha.
        self.register_buffer("e0", torch.zeros(num_species, num_targets, dtype=torch.float64))
        self.register_buffer("mean", torch.zeros(num_targets, dtype=torch.float64))
        self.register_buffer("std", torch.ones(num_targets, dtype=torch.float64))

    @torch.no_grad()
    def fit(self, dataset: Iterable[Data]) -> "TargetNormalizer":
        comps, ys = [], []
        for data in dataset:
            comps.append(torch.bincount(data.z, minlength=self.num_species).to(torch.float64))
            ys.append(data.y.reshape(-1).to(torch.float64))
        C = torch.stack(comps)  # (M, S)
        Y = torch.stack(ys)  # (M, T)

        # Ridge-free least squares; elements absent from the data get E0 = 0 (lstsq min-norm).
        e0 = torch.linalg.lstsq(C, Y, driver="gelsd").solution  # (S, T)
        ext = self.extensive.cpu()
        e0[:, ~ext] = 0.0
        mean = torch.where(ext, torch.zeros(Y.shape[1], dtype=Y.dtype), Y.mean(0))

        residual = Y - C @ e0 - mean
        std = residual.std(0, unbiased=False).clamp(min=1e-8)

        self.e0.copy_(e0.to(self.e0))
        self.mean.copy_(mean.to(self.mean))
        self.std.copy_(std.to(self.std))
        return self

    def baseline(self, data: Data) -> torch.Tensor:
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros_like(data.z)
        num_graphs = int(data.num_graphs) if hasattr(data, "num_graphs") else int(batch.max()) + 1
        comp = composition_matrix(data.z, batch, num_graphs, self.num_species).to(self.e0)
        return comp @ self.e0 + self.mean

    def normalize(self, y: torch.Tensor, data: Data, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Physical targets -> standardised residuals, computed in float64 and returned as `dtype`."""
        return ((y.to(torch.float64) - self.baseline(data)) / self.std).to(dtype)

    def denormalize(self, y_norm: torch.Tensor, data: Data) -> torch.Tensor:
        """Standardised predictions -> physical units, returned in float64."""
        return y_norm.to(torch.float64) * self.std + self.baseline(data)
