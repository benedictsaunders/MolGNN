import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data

from molgnn.configuration import Configuration
from molgnn.graph import radius_graph

# Verbatim copies of two real Kaggle QM9 files (tab separated, trailing tab on the property line).
METHANE = (
    "5\n"
    "gdb 1\t157.7118\t157.70997\t157.70699\t0.\t13.21\t-0.3877\t0.1171\t0.5048\t35.3641\t0.044749"
    "\t-40.47893\t-40.476062\t-40.475117\t-40.498597\t6.469\t\n"
    "C\t-0.0126981359\t 1.0858041578\t 0.0080009958\t-0.535689\n"
    "H\t 0.002150416\t-0.0060313176\t 0.0019761204\t 0.133921\n"
    "H\t 1.0117308433\t 1.4637511618\t 0.0002765748\t 0.133922\n"
    "H\t-0.540815069\t 1.4475266138\t-0.8766437152\t 0.133923\n"
    "H\t-0.5238136345\t 1.4379326443\t 0.9063972942\t 0.133923\n"
    "1341.307\t1341.3284\t1341.365\t1562.6731\t1562.7453\t3038.3205\t3151.6034\t3151.6788\t3151.7078\n"
    "C\tC\t\n"
    "InChI=1S/CH4/h1H4\tInChI=1S/CH4/h1H4\n"
)
ACETYLENE = (
    "4\n"
    "gdb 4\t0.\t35.6100361\t35.6100361\t0.\t16.28\t-0.2845\t0.0506\t0.3351\t59.5248\t0.026841"
    "\t-77.308427\t-77.305527\t-77.304583\t-77.327429\t8.574\t\n"
    "C\t 0.5995394918\t 0.\t 1.\t-0.207019\n"
    "C\t-0.5995394918\t 0.\t 1.\t-0.207019\n"
    "H\t-1.6616385861\t 0.\t 1.\t 0.207019\n"
    "H\t 1.6616385861\t 0.\t 1.\t 0.207019\n"
    "549.7648\t549.7648\t795.2713\t795.2713\t2078.1131\t3455.8096\t3557.8599\n"
    "C#C\tC#C\t\n"
    "InChI=1S/C2H2/c1-2/h1-2H\tInChI=1S/C2H2/c1-2/h1-2H\n"
)
# Water, with one coordinate rewritten in the Mathematica-style exponent ("*^") that some QM9 files use.
WATER_STAR_EXP = (
    "3\n"
    "gdb 3\t799.58812\t437.90386\t282.94545\t1.9255\t6.31\t-0.2928\t0.0687\t0.3615\t19.0002\t0.021375"
    "\t-76.404702\t-76.401867\t-76.400922\t-76.422349\t6.002\t\n"
    "O\t-0.0343604951\t 0.9775395708\t 0.0076015923\t-0.589706\n"
    "H\t 0.0647664923\t 0.0205721668\t 0.0015346341\t 0.294853\n"
    "H\t 0.8717903737\t 1.3008770188\t 1.5*^-5\t 0.294853\n"
    "1658.11\t3664.32\t3762.86\n"
    "O\tO\t\n"
    "InChI=1S/H2O/h1H2\tInChI=1S/H2O/h1H2\n"
)


@pytest.fixture(autouse=True)
def isolated_configuration(tmp_path, monkeypatch):
    """Point the global Configuration at a throwaway file so tests never touch configuration.json."""
    monkeypatch.setattr(Configuration, "_filepath", str(tmp_path / "configuration.json"))
    monkeypatch.setattr(Configuration, "_config", {})
    monkeypatch.setattr(Configuration, "_loaded", False)
    yield


@pytest.fixture
def fake_qm9_dir(tmp_path) -> Path:
    d = tmp_path / "qm9"
    d.mkdir()
    (d / "dsgdb9nsd_000001.xyz").write_text(METHANE)
    (d / "dsgdb9nsd_000003.xyz").write_text(WATER_STAR_EXP)
    (d / "dsgdb9nsd_000004.xyz").write_text(ACETYLENE)
    # The Kaggle archive also contains a combined file that must not be globbed.
    (d / "dsgdb9nsd.xyz").write_text("not a molecule")
    Configuration.set("dataset_path", str(d))
    return d


@pytest.fixture
def no_kaggle(monkeypatch):
    """Fail loudly if any code path tries to download from Kaggle."""
    import kagglehub

    def boom(*args, **kwargs):
        raise AssertionError("kagglehub.dataset_download should not be called")

    monkeypatch.setattr(kagglehub, "dataset_download", boom)


def random_molecule(n_atoms: int, generator: torch.Generator, r_cut: float = 5.0, n_targets: int = 5) -> Data:
    species = torch.tensor([1, 6, 7, 8, 9])
    z = species[torch.randint(0, len(species), (n_atoms,), generator=generator)]
    pos = 1.5 * torch.randn(n_atoms, 3, generator=generator)
    y = torch.randn(1, n_targets, generator=generator)
    return Data(z=z, pos=pos, y=y, edge_index=radius_graph(pos, r_cut))


@pytest.fixture
def molecules() -> list[Data]:
    g = torch.Generator().manual_seed(1234)
    return [random_molecule(n, g) for n in (3, 5, 8, 2, 6)]


@pytest.fixture
def batch(molecules) -> Batch:
    return Batch.from_data_list(molecules)
