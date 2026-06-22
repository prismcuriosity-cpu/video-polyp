"""Dataset registry and the common ``Sample`` schema.

A registered source is a function ``(root: Path, split: str) -> list[Sample]``.
The :class:`Sample` carries everything the pipeline can supervise; any field may
be ``None`` when a dataset lacks that annotation (e.g. Kvasir-SEG has masks but
no Paris/NICE labels), and the training code masks missing targets per task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# task label keys a dataset may provide
LABEL_KEYS = ("paris", "nice", "kudo", "malignancy")


@dataclass
class Sample:
    image_path: str
    mask_path: Optional[str] = None
    boxes: Optional[list] = None                    # [[x1,y1,x2,y2], ...] if provided
    labels: dict = field(default_factory=dict)      # subset of LABEL_KEYS -> int
    dataset: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class DatasetSpec:
    name: str
    loader: Callable[[Path, str], list]
    has_masks: bool = True
    has_boxes: bool = False
    label_keys: tuple = ()
    note: str = ""


DATASET_REGISTRY: dict[str, DatasetSpec] = {}


def register_dataset(name: str, **kwargs):
    def deco(fn: Callable):
        DATASET_REGISTRY[name] = DatasetSpec(name=name, loader=fn, **kwargs)
        return fn

    return deco


def list_datasets() -> list[str]:
    return sorted(DATASET_REGISTRY)


def resolve_samples(name: str, root: str | Path, split: str = "train") -> list[Sample]:
    """Return samples for a registered dataset, or [] (with a warning) if the
    data root is absent — so configs can reference not-yet-downloaded datasets."""
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Registered: {list_datasets()}")
    root = Path(root)
    spec = DATASET_REGISTRY[name]
    if not root.exists():
        import warnings

        warnings.warn(f"[data] root for '{name}' not found: {root} — returning 0 samples.")
        return []
    samples = spec.loader(root, split)
    for s in samples:
        s.dataset = name
    return samples
