# MEN is short for minimal equivanriant network!

import torch
import torch.nn as nn
from e3nn import o3
from e3nn import FullyConnectedTensorProduct
from collections.abc import Iterable
from typing import Tuple, List, Dict, Any, Union, Optional

""" e3nn nomenclature
    AxBC, e.g. 16x0e
    A: Multiplier
    B: Rank
    C: Parity (Odd or Even)

    Dimension (2*Rank + 1)

    so 16x0e
    16 x scalar (R1) with even parity
"""

class MinialEquivarientLayer(nn.Module):
    def __init__(
            self,
            irreps_in: str,
            irreps_out: str,
            lmax: int = 3,
            radial_neurons: List[int] = [12, 12],
    ) -> None:
        """
        Args:
            irreps_in (str): e3nn irreps representations
            irreps_out (str): e3nn irreps representations
            lmax (int): maximum radial quantum number
        """  
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.spherical_harmonics = o3.SphericalHarmonics(
            self.irreps_sh, normalize=True, normalization='component'
        )
        
        self.tp = FullyConnectedTensorProduct(
            self.irreps_in, 
            self.irreps_sh, 
            self.irreps_out, 
            shared_weights=False
        )
        
        num_weights = self.tp.weight_numel
        
        self.radial_mlp = nn.Sequential(
            nn.Linear(1, radial_neurons[0]), # Input is distance d_ij
            nn.SiLU(),
            nn.Linear(radial_neurons[0], radial_neurons[1]),
            nn.SiLU(),
            nn.Linear(radial_neurons[1], num_weights)
        )

    def forward(self, node_features, edge_index, edge_vec, edge_dist):
            src, dst = edge_index
            sh = self.spherical_harmonics(edge_vec)
            weight = self.radial_mlp(edge_dist)
            features_src = node_features[src]
            messages = self.tp(features_src, sh, weight)
            num_nodes = node_features.shape[0]
            feature_dim = messages.shape[1]
            
            node_features_out = torch.zeros(
                (num_nodes, feature_dim), 
                device=messages.device, 
                dtype=messages.dtype
            )
            node_features_out.index_add_(dim=0, index=dst, source=messages)
            
            return node_features_out