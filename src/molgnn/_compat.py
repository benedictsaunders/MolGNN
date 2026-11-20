"""Compatibility shims that must run before e3nn is imported."""

import warnings

import torch

# Harmless noise from e3nn compiling its modules with TorchScript at import time.
warnings.filterwarnings("ignore", message="The TorchScript type system")
warnings.filterwarnings("ignore", message="`torch.jit.script` is deprecated")

# e3nn 0.4.4 (pinned by mace-torch) loads its bundled constants.pt with a bare torch.load.
# Since torch 2.6 that defaults to weights_only=True, which rejects the pickled `slice`
# objects in that file. `slice` is an inert builtin, so allowlisting it is safe.
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])
