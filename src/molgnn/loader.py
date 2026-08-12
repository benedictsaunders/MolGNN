from pathlib import Path
from typing import List, Tuple
import os
import numpy as np
import logging
from glob import glob
from ase import Atoms
from molgnn.configuration import Configuration
from molgnn.data import PERIODIC_TABLE_SYMS
from molgnn.utils import str_to_float

from tqdm import tqdm
import torch
from torch_geometric.data import Data

logging.basicConfig()

def parse_kaggle_qm9(filepath: str | Path) -> Tuple[Atoms, Data]:
    """
    Parse a non-standard XYZ file to an ASE atoms object and torchgeometric data object

    Args:
        filepath (str or Path): Path to the XYZ file

    Returns:
        Tuple[ase.Atoms, torch_geometric.data.Data]
    """
    with open(filepath) as f:
        lines = [line.strip() for line in f.readlines()]
        natoms = int(lines[0])
        props = lines[1].strip().split()
        homolumo = str_to_float(props[9]) # HOMO-LUMO gap
        U0 = str_to_float(props[12]) # Internal energy at 0K
        U  = str_to_float(props[13]) # Internal energy at RT
        H  = str_to_float(props[14]) # Enthalpy
        G  = str_to_float(props[15]) # Gibbs free
        Cv = str_to_float(props[16]) # Heat capacity

        y_tensor = torch.tensor([[U0, U, H, G]], dtype=torch.float)
        Zs = []
        Qs = []
        coordinates = []

        for line in lines[2 : 2 + natoms]:
                    parts = line.strip().split()
                    Zs.append(PERIODIC_TABLE_SYMS[parts[0]])
                    Qs.append(str_to_float(parts[4]))
                    coordinates.append([str_to_float(parts[1]), str_to_float(parts[2]), str_to_float(parts[3])])
        atoms = Atoms(symbols=Zs, positions=coordinates)

    data = Data(
         z = torch.tensor(Zs, dtype=torch.long),
         coords = torch.tensor(coordinates, dtype=torch.float),
         q = torch.tensor(Qs, dtype=torch.float),
         y = y_tensor
    )
    return (atoms, data)


def get_qm9() -> List[Tuple[Atoms, Data]]:
    """
    Load the QM9 dataset. If it does not exist at the path specified in the MolGNN
    configuration file or such a key doesn't exist, then it will be downloaded from
    Kaggle accordingly.

    Returns:
        List: The QM9 dataset as ASE geometries.
    """

    dataset_path = Configuration.get("dataset_path", "_")
    if dataset_path == "_" or not Path(os.getcwd() + "/" + dataset_path).exists():
        import kagglehub
        dataset_path = kagglehub.dataset_download("zaharch/quantum-machine-9-aka-qm9")
        Configuration.set("dataset_path", dataset_path, auto_save=True)
    files = list(glob(str(dataset_path) + "/dsgdb9nsd_*.xyz"))
    assert len(files) == 133885
    return [parse_kaggle_qm9(f) for f in tqdm(files)]
    
if __name__ == "__main__":
    get_qm9()
        