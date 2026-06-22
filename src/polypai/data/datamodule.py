"""Builds memory-efficient dataloaders from one or more registered datasets."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch.utils.data import ConcatDataset, DataLoader

from polypai.data.base import PolypDataset, collate_fn
from polypai.data.registry import resolve_samples
from polypai.data.transforms import default_train_transforms, default_val_transforms


@dataclass
class DataConfig:
    # {dataset_name: root_path}; placeholder roots that don't exist yield 0 samples.
    datasets: dict[str, str] = field(default_factory=lambda: {"kvasir_seg": "data/Kvasir-SEG"})
    image_size: int = 512
    batch_size: int = 8
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4


def _build_split(cfg: DataConfig, split: str, train: bool) -> PolypDataset:
    tfm = default_train_transforms(cfg.image_size) if train else default_val_transforms(cfg.image_size)
    parts = []
    for name, root in cfg.datasets.items():
        samples = resolve_samples(name, root, split)
        if samples:
            parts.append(PolypDataset(samples, tfm, train=train))
    if not parts:
        return PolypDataset([], tfm, train=train)
    merged = parts[0] if len(parts) == 1 else _ConcatPolyp(parts, tfm)
    return merged


class _ConcatPolyp(ConcatDataset):
    """ConcatDataset that still exposes the transforms attribute for clarity."""

    def __init__(self, datasets, transforms):
        super().__init__(datasets)
        self.transforms = transforms


def build_dataloaders(cfg: DataConfig) -> dict[str, DataLoader]:
    train_ds = _build_split(cfg, "train", train=True)
    val_ds = _build_split(cfg, "val", train=False)

    common = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
    )
    if cfg.num_workers > 0:
        common.update(persistent_workers=cfg.persistent_workers, prefetch_factor=cfg.prefetch_factor)

    loaders = {}
    if len(train_ds):
        loaders["train"] = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    if len(val_ds):
        loaders["val"] = DataLoader(val_ds, shuffle=False, **common)
    return loaders
