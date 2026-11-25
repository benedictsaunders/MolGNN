# MolGNN

This repository hosts my implementation of a GNN predictive model to predict various chemical properties of small organic molecules, using data from the QM9 dataset.

## Installation

Call me unconventional, but I went with team red, rather than team green, for my GPU. As such, the project configuration within the `pyproject.toml` file is geared towards the ROCm stack, instead of CUDA. It uses the PyTorch ROCm 7.2 wheels (torch 2.14), the first line whose builds include RDNA4 (gfx1200/gfx1201) kernels alongside older AMD architectures.

⚠️ **At this point (November 2026), it is mostly created in a few-shot approach with Claude Code, as an experiment. As such, it has not been tested as a formal research tool.** Building the configuration for PyTorch was a lot more fiddly, so that was done by hand.

## Usage

```bash
uv sync                                  # install (incl. the pytest dev group)
uv run pytest                            # test suite (~10 s, CPU only)
uv run molgnn --dry-run --limit 1000     # build data/model/optimiser, one forward pass, no training
uv run molgnn                            # train on the full dataset
```

Hyperparameters live in `src/molgnn/train.py` (`DEFAULTS`) and can be overridden in `configuration.json`.
Parsed graphs are cached to `data/qm9_graphs.pt`; the best checkpoint goes to `checkpoints/best.pt`.

## Model

`src/molgnn/men.py` is a minimal MACE-style network: species embedding, Bessel radial basis with a
smooth polynomial cutoff, equivariant channel-wise tensor-product message passing, a correlation-order-2
product basis (`A + A⊗A`), residual updates and per-layer invariant readouts. It predicts the HOMO-LUMO gap
(mean-pooled) and U0, U, H, G (sum-pooled, with fitted per-element reference energies), all in Hartree.

## Philosophy

Although QM9 exists in the `torchgeometric` library, this project uses the data on Kaggle, such that the XYZ geometries are preserved to create custom representations and apply existing, supplementary molecular descriptors.
