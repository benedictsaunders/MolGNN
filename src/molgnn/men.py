# MEN is short for minimal equivariant network!
#
# A deliberately small MACE-style model:
#   1. species embedding           h_i^(0)  = W z_i                         (C x 0e)
#   2. equivariant interaction     A_i      = sum_j R(r_ij) * (W h_j ⊗ Y(r̂_ij))  (2-body)
#   3. product basis               B_i      = W1 A_i + W2 (A_i ⊗ A_i)        (body order 3)
#   4. residual update             h_i      = B_i + W_skip h_i
#   5. per-layer invariant readout, summed over layers, pooled over atoms
#
# Relative to full MACE, the main simplifications are: correlation order 2 instead of 3,
# product-basis weights shared across elements, and a plain (not species-dependent) skip.

from typing import Sequence

import torch
import torch.nn as nn
from e3nn import o3
from torch_geometric.data import Data

from molgnn.graph import RadialEmbedding, edge_vectors

""" e3nn nomenclature
    AxBC, e.g. 16x0e
    A: Multiplier
    B: Rank (l)
    C: Parity (Odd or Even)

    Dimension (2*Rank + 1)

    so 16x0e
    16 x scalar (l=0) with even parity
"""


def channelwise_tp_irreps(
    irreps_in: o3.Irreps, irreps_sh: o3.Irreps, target_irreps: o3.Irreps
) -> tuple[o3.Irreps, list[tuple[int, int, int, str, bool]]]:
    """
    Build the output irreps and instructions for a channel-wise ("uvu") tensor product,
    keeping only output irreps that appear in `target_irreps`. Each input channel is
    coupled to the spherical harmonics independently, so the number of per-edge weights
    is C * num_paths instead of C^2 * num_paths for a fully connected product.
    """
    targets = {ir for _, ir in target_irreps}
    irreps_mid = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps_in):
        for j, (_, ir_sh) in enumerate(irreps_sh):
            for ir_out in ir_in * ir_sh:
                if ir_out in targets:
                    instructions.append((i, j, len(irreps_mid), "uvu", True))
                    irreps_mid.append((mul, ir_out))
    irreps_mid = o3.Irreps(irreps_mid)
    irreps_mid, perm, _ = irreps_mid.sort()
    instructions = [(i, j, perm[k], mode, train) for i, j, k, mode, train in instructions]
    return irreps_mid, instructions


def radial_mlp(sizes: Sequence[int]) -> nn.Sequential:
    """Bias-free MLP, so a radial embedding that vanishes at r_cut gives weights that do too."""
    layers: list[nn.Module] = []
    for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b, bias=False))
        if k < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class EquivariantInteraction(nn.Module):
    """Equivariant 2-body message passing: A_i = 1/avg_nn * sum_j R(r_ij) * (W h_j ⊗ Y(r̂_ij))."""

    def __init__(
        self,
        irreps_in: str | o3.Irreps,
        irreps_sh: str | o3.Irreps,
        irreps_out: str | o3.Irreps,
        num_basis: int,
        radial_neurons: Sequence[int] = (64, 64),
        avg_num_neighbors: float = 1.0,
    ) -> None:
        """
        Args:
            irreps_in (str): e3nn irreps of the incoming node features
            irreps_sh (str): e3nn irreps of the edge spherical harmonics
            irreps_out (str): e3nn irreps of the aggregated messages A_i
            num_basis (int): size of the radial embedding fed to the radial MLP
            radial_neurons (list[int]): hidden widths of the radial MLP
            avg_num_neighbors (float): normaliser for the neighbour sum
        """
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_out = o3.Irreps(irreps_out)
        self.avg_num_neighbors = avg_num_neighbors

        self.linear_up = o3.Linear(self.irreps_in, self.irreps_in)
        irreps_mid, instructions = channelwise_tp_irreps(self.irreps_in, self.irreps_sh, self.irreps_out)
        self.tp = o3.TensorProduct(
            self.irreps_in,
            self.irreps_sh,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.radial_mlp = radial_mlp([num_basis, *radial_neurons, self.tp.weight_numel])
        self.linear = o3.Linear(irreps_mid, self.irreps_out)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_radial: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        h = self.linear_up(node_features)
        weight = self.radial_mlp(edge_radial)
        messages = self.tp(h[src], edge_sh, weight)
        agg = torch.zeros(
            (node_features.shape[0], messages.shape[1]), device=messages.device, dtype=messages.dtype
        )
        agg.index_add_(0, dst, messages)
        return self.linear(agg / self.avg_num_neighbors)


class ProductBasis(nn.Module):
    """B = W1 A + W2 (A ⊗ A): a correlation-order-2 truncation of MACE's symmetric contraction."""

    def __init__(self, irreps_in: str | o3.Irreps, irreps_out: str | o3.Irreps) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.square = o3.ElementwiseTensorProduct(
            self.irreps_in, self.irreps_in, filter_ir_out=[ir for _, ir in self.irreps_out]
        )
        self.linear1 = o3.Linear(self.irreps_in, self.irreps_out)
        self.linear2 = o3.Linear(self.square.irreps_out, self.irreps_out)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        return self.linear1(A) + self.linear2(self.square(A, A))


class MiniMACE(nn.Module):
    """Minimal MACE-style invariant property predictor for small molecules."""

    def __init__(
        self,
        num_targets: int,
        extensive: Sequence[bool],
        num_species: int = 10,
        channels: int = 32,
        lmax_hidden: int = 1,
        lmax_sh: int = 3,
        num_layers: int = 2,
        r_cut: float = 5.0,
        num_basis: int = 8,
        radial_neurons: Sequence[int] = (64, 64),
        readout_hidden: int = 16,
        avg_num_neighbors: float = 1.0,
    ) -> None:
        """
        Args:
            num_targets (int): number of graph-level outputs
            extensive (list[bool]): per target, sum-pool (True, e.g. energies) or mean-pool (False)
            num_species (int): embedding table size, indexed directly by atomic number
            channels (int): multiplicity C of every irrep in the hidden features
            lmax_hidden (int): max l carried in the hidden node features
            lmax_sh (int): max l of the edge spherical harmonics
            num_layers (int): number of interaction + product blocks
            r_cut (float): cutoff radius in Å, must match the graph construction
            num_basis (int): number of Bessel radial basis functions
            radial_neurons (list[int]): hidden widths of the radial MLPs
            readout_hidden (int): hidden width of the final non-linear readout
            avg_num_neighbors (float): neighbour-sum normaliser, typically estimated from data
        """
        super().__init__()
        if len(extensive) != num_targets:
            raise ValueError("`extensive` must have one entry per target")
        self.register_buffer("extensive", torch.tensor(list(extensive), dtype=torch.bool))
        self.r_cut = r_cut

        irreps_scalar = o3.Irreps(f"{channels}x0e")
        # "natural" parity: 0e + 1o + 2e + ..., the same parity the spherical harmonics carry
        self.irreps_hidden = o3.Irreps([(channels, (l, (-1) ** l)) for l in range(lmax_hidden + 1)])
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax_sh)
        irreps_readout = o3.Irreps(f"{num_targets}x0e")

        self.species_embedding = nn.Embedding(num_species, channels)
        self.radial = RadialEmbedding(r_cut, num_basis)
        self.spherical_harmonics = o3.SphericalHarmonics(
            self.irreps_sh, normalize=True, normalization="component"
        )

        self.interactions = nn.ModuleList()
        self.products = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.readouts = nn.ModuleList()
        irreps_in = irreps_scalar
        for layer in range(num_layers):
            last = layer == num_layers - 1
            # Only scalars are needed after the final layer.
            irreps_out = irreps_scalar if last else self.irreps_hidden
            self.interactions.append(
                EquivariantInteraction(
                    irreps_in,
                    self.irreps_sh,
                    self.irreps_hidden,
                    num_basis=num_basis,
                    radial_neurons=radial_neurons,
                    avg_num_neighbors=avg_num_neighbors,
                )
            )
            self.products.append(ProductBasis(self.irreps_hidden, irreps_out))
            self.skips.append(o3.Linear(irreps_in, irreps_out))
            if last:
                self.readouts.append(
                    nn.Sequential(
                        nn.Linear(channels, readout_hidden),
                        nn.SiLU(),
                        nn.Linear(readout_hidden, num_targets),
                    )
                )
            else:
                self.readouts.append(o3.Linear(irreps_out, irreps_readout))
            irreps_in = irreps_out

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: (batched) graph with `z` (N,), `pos` (N, 3), `edge_index` (2, E) and,
                if batched, `batch` (N,)

        Returns:
            Tensor: (num_graphs, num_targets) predictions
        """
        z, pos, edge_index = data.z, data.pos, data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros_like(z)
        num_graphs = int(data.num_graphs) if hasattr(data, "num_graphs") else int(batch.max()) + 1

        vec, length = edge_vectors(pos, edge_index)
        edge_sh = self.spherical_harmonics(vec)
        edge_radial = self.radial(length)

        h = self.species_embedding(z)
        atom_out = torch.zeros((z.shape[0], self.extensive.shape[0]), device=pos.device, dtype=pos.dtype)
        for interaction, product, skip, readout in zip(
            self.interactions, self.products, self.skips, self.readouts
        ):
            A = interaction(h, edge_index, edge_sh, edge_radial)
            h = product(A) + skip(h)
            atom_out = atom_out + readout(h)

        summed = torch.zeros((num_graphs, atom_out.shape[1]), device=pos.device, dtype=pos.dtype)
        summed.index_add_(0, batch, atom_out)
        counts = torch.bincount(batch, minlength=num_graphs).clamp(min=1).unsqueeze(-1).to(pos.dtype)
        return torch.where(self.extensive, summed, summed / counts)
