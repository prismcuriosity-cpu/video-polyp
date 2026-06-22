"""Modular dataset layer.

Datasets are registered by name so placeholder sources can be swapped for real
ones by editing a single path in ``configs/data`` — no code change.  The sample
*discovery* functions are pure-Python (importable without torch); the torch
``Dataset`` wrapper and dataloaders live in :mod:`polypai.data.base` /
:mod:`polypai.data.datamodule`.
"""

from __future__ import annotations

from polypai.data.registry import (
    DATASET_REGISTRY,
    DatasetSpec,
    list_datasets,
    register_dataset,
    resolve_samples,
)

# import side-effect: populate the registry
from polypai.data import sources as _sources  # noqa: E402,F401

__all__ = [
    "DATASET_REGISTRY",
    "DatasetSpec",
    "register_dataset",
    "resolve_samples",
    "list_datasets",
]
