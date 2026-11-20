# MolGNN

This repository hosts my implementation of a GNN predictive model to predict various chemical properties of small organic molecules, trained on the QM9 dataset.

## Installation

Call me unconventional, but I went with team red, rather than team green, for my GPU. As such, the project configuration within the `pyproject.toml` file is geared towards the ROCm stack, instead of CUDA.

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

Although the QM9 exists in the `torchgeometric` library, this project uses the data on Kaggle, such that the XYZ geometries are preserved to create custom representations, and apply existing, supplementary molecular descriptors .
