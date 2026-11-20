from pathlib import Path
from typing import List, Optional, Tuple
import logging
from glob import glob
from ase import Atoms
from molgnn.configuration import Configuration
from molgnn.data import PERIODIC_TABLE_SYMS
from molgnn.graph import RadiusGraph
from molgnn.utils import str_to_float

from tqdm import tqdm
import torch
from torch_geometric.data import Data

logging.basicConfig()
logger = logging.getLogger(__name__)

QM9_NUM_MOLECULES = 133885

# Columns of the `y` tensor built by `parse_kaggle_qm9`, all in Hartree (stored as float64).
# `extensive` targets scale with molecule size and are sum-pooled / given atomic reference energies.
TARGETS = ("gap", "U0", "U", "H", "G")
TARGET_EXTENSIVE = (False, True, True, True, True)


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

        # float64: U0 etc. are ~ -400 Ha, where float32 resolution (~3e-5 Ha) is too coarse
        y_tensor = torch.tensor([[homolumo, U0, U, H, G]], dtype=torch.float64)
        Zs = []
        Qs = []
        coordinates = []

        for line in lines[2 : 2 + natoms]:
                    parts = line.strip().split()
                    Zs.append(PERIODIC_TABLE_SYMS[parts[0]])
                    Qs.append(str_to_float(parts[4]))
                    coordinates.append([str_to_float(parts[1]), str_to_float(parts[2]), str_to_float(parts[3])])
        atoms = Atoms(numbers=Zs, positions=coordinates)

    data = Data(
         z = torch.tensor(Zs, dtype=torch.long),
         pos = torch.tensor(coordinates, dtype=torch.float),
         q = torch.tensor(Qs, dtype=torch.float),
         y = y_tensor,
         cv = torch.tensor([Cv], dtype=torch.float),
         idx = int(Path(filepath).stem.split("_")[-1]),
    )
    return (atoms, data)


def resolve_dataset_path() -> Path:
    """
    Return the QM9 directory from the configuration, downloading from Kaggle (and recording
    the location) if the key is missing or the directory does not exist.
    """
    dataset_path = Configuration.get("dataset_path")
    if dataset_path is None or not Path(dataset_path).exists():
        import kagglehub
        dataset_path = kagglehub.dataset_download("zaharch/quantum-machine-9-aka-qm9")
        Configuration.set("dataset_path", dataset_path, auto_save=True)
    return Path(dataset_path)


def get_qm9(limit: Optional[int] = None) -> List[Tuple[Atoms, Data]]:
    """
    Load the QM9 dataset. If it does not exist at the path specified in the MolGNN
    configuration file or such a key doesn't exist, then it will be downloaded from
    Kaggle accordingly.

    Args:
        limit (int, optional): only parse the first `limit` molecules (by GDB index)

    Returns:
        List: The QM9 dataset as (ASE geometry, PyG Data) pairs.
    """
    dataset_path = resolve_dataset_path()
    files = sorted(glob(str(dataset_path / "dsgdb9nsd_*.xyz")))
    if limit is None:
        if len(files) != QM9_NUM_MOLECULES:
            raise RuntimeError(
                f"Expected {QM9_NUM_MOLECULES} QM9 files in {dataset_path}, found {len(files)}"
            )
    else:
        files = files[:limit]
    return [parse_kaggle_qm9(f) for f in tqdm(files, desc="Parsing QM9")]


def load_qm9_graphs(
    r_cut: float,
    limit: Optional[int] = None,
    cache_path: Optional[str | Path] = None,
) -> List[Data]:
    """
    QM9 as a list of PyG graphs with `edge_index` built at cutoff `r_cut`.

    Parsing 134k XYZ files is slow, so the result can be cached to `cache_path` with
    torch.save. The cache records the cutoff and limit and is rebuilt if they differ.
    """
    if cache_path is not None and Path(cache_path).exists():
        cached = torch.load(cache_path, weights_only=False)
        if cached.get("r_cut") == r_cut and cached.get("limit") == limit:
            return cached["graphs"]
        logger.info("Cache %s was built with different settings, rebuilding", cache_path)

    transform = RadiusGraph(r_cut)
    graphs = [transform(data) for _, data in get_qm9(limit=limit)]

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"r_cut": r_cut, "limit": limit, "graphs": graphs}, cache_path)
    return graphs


if __name__ == "__main__":
    get_qm9()
