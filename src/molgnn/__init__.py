import molgnn._compat  # noqa: F401  (must precede any e3nn import)
from molgnn.configuration import Configuration
from molgnn.men import MiniMACE
from molgnn.train import main

__all__ = ["Configuration", "MiniMACE", "main"]
