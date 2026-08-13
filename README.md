# MolGNN

This repository hosts my implementation of a GNN predictive model to predict various chemical properties of small organic molecules, trained on the QM9 dataset.

## Installation

Call me unconventional, but I went with team red, rather than team green, for my GPU. As such, the project configuration within the `pyproject.toml` file is geared towards the ROCm stack, instead of CUDA.

## Philosophy

Although the QM9 exists in the `torchgeometric` library, this project uses the data on Kaggle, such that the XYZ geometries are preserved to create custom representations, and apply existing, supplementary molecular descriptors .
