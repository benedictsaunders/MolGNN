"""Graph construction and radial featurisation (MACE-style Bessel basis + polynomial envelope)."""

import math

import torch
import torch.nn as nn
from torch_geometric.data import Data


def radius_graph(pos: torch.Tensor, r_cut: float) -> torch.Tensor:
    """
    Dense O(N^2) radius graph for a single molecule. QM9 molecules have <= 29 atoms, so
    this avoids a torch_cluster dependency (awkward on ROCm) at negligible cost.

    Args:
        pos (Tensor): (N, 3) Cartesian positions
        r_cut (float): cutoff radius, same units as pos

    Returns:
        Tensor: (2, E) edge_index with edge_index[0] = source j, edge_index[1] = receiver i,
        for every ordered pair i != j with |r_i - r_j| < r_cut.
    """
    dist = torch.cdist(pos, pos)
    mask = (dist < r_cut) & ~torch.eye(pos.shape[0], dtype=torch.bool, device=pos.device)
    dst, src = mask.nonzero(as_tuple=True)
    return torch.stack([src, dst], dim=0)


class RadiusGraph:
    """PyG-style transform attaching `edge_index` to a Data object that has `pos`."""

    def __init__(self, r_cut: float) -> None:
        self.r_cut = r_cut

    def __call__(self, data: Data) -> Data:
        data.edge_index = radius_graph(data.pos, self.r_cut)
        return data


def edge_vectors(pos: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (vec, length) with vec = r_src - r_dst (pointing from receiver i to neighbour j)."""
    src, dst = edge_index
    vec = pos[src] - pos[dst]
    return vec, vec.norm(dim=-1, keepdim=True)


class PolynomialCutoff(nn.Module):
    """Smooth envelope u(r) with u(0)=1 and u, u', u'' -> 0 at r_cut (DimeNet / MACE)."""

    def __init__(self, r_cut: float, p: int = 6) -> None:
        super().__init__()
        self.register_buffer("r_cut", torch.tensor(float(r_cut)))
        self.p = p

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        p = self.p
        x = r / self.r_cut
        env = (
            1.0
            - (p + 1) * (p + 2) / 2 * x**p
            + p * (p + 2) * x ** (p + 1)
            - p * (p + 1) / 2 * x ** (p + 2)
        )
        return env * (x < 1.0)


class BesselBasis(nn.Module):
    """sqrt(2/r_c) * sin(n*pi*r/r_c) / r, n = 1..num_basis, with trainable frequencies."""

    def __init__(self, r_cut: float, num_basis: int = 8, trainable: bool = False) -> None:
        super().__init__()
        freqs = math.pi / r_cut * torch.arange(1, num_basis + 1, dtype=torch.get_default_dtype())
        if trainable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)
        self.register_buffer("prefactor", torch.tensor(math.sqrt(2.0 / r_cut)))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # r: (E, 1) -> (E, num_basis)
        return self.prefactor * torch.sin(self.freqs * r) / r


class RadialEmbedding(nn.Module):
    """Bessel basis multiplied by the smooth cutoff envelope."""

    def __init__(self, r_cut: float, num_basis: int = 8, p: int = 6) -> None:
        super().__init__()
        self.num_basis = num_basis
        self.bessel = BesselBasis(r_cut, num_basis)
        self.cutoff = PolynomialCutoff(r_cut, p)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.bessel(r) * self.cutoff(r)
