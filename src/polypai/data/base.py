"""Torch ``Dataset`` over the registry ``Sample`` schema.

Handles heterogeneous supervision: masks are loaded when present (else an empty
mask), detection boxes are derived from mask connected-components when a dataset
ships only masks, and missing classification labels are encoded as ``-100`` so
the loss can ignore them per task.  This is what lets a single training run mix
Kvasir-SEG (masks only) with PICCOLO (masks + Paris/NICE).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from polypai.data.registry import LABEL_KEYS, Sample
from polypai.data.transforms import Compose, default_train_transforms, default_val_transforms

IGNORE_INDEX = -100


def _imread_rgb(path: str) -> np.ndarray:
    try:
        import cv2

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except ImportError:  # pragma: no cover
        from PIL import Image

        return np.array(Image.open(path).convert("RGB"))


def _maskread(path: str | None, shape) -> np.ndarray:
    if not path:
        return np.zeros(shape[:2], np.uint8)
    try:
        import cv2

        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return np.zeros(shape[:2], np.uint8)
        return (m > 127).astype(np.uint8)
    except ImportError:  # pragma: no cover
        from PIL import Image

        return (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)


def mask_to_boxes(mask: np.ndarray, min_area: int = 8) -> list[list[float]]:
    """Axis-aligned boxes for each connected component of a binary mask."""
    boxes = []
    try:
        from scipy import ndimage

        lbl, n = ndimage.label(mask)
        for i in range(1, n + 1):
            ys, xs = np.where(lbl == i)
            if len(xs) < min_area:
                continue
            boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
    except Exception:  # pragma: no cover
        if mask.any():
            ys, xs = np.where(mask)
            boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
    return boxes


class PolypDataset(Dataset):
    def __init__(self, samples: list[Sample], transforms: Compose | None = None, train: bool = True):
        self.samples = samples
        self.transforms = transforms or (default_train_transforms() if train else default_val_transforms())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        image = _imread_rgb(s.image_path)
        mask = _maskread(s.mask_path, image.shape)

        out = self.transforms(image=image, mask=mask)
        image_t, mask_t = out["image"], out["mask"]

        mask_np = mask_t.numpy() if isinstance(mask_t, torch.Tensor) else np.asarray(mask_t)
        boxes = s.boxes if s.boxes else mask_to_boxes(mask_np.astype(np.uint8))
        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))

        labels = {k: int(s.labels.get(k, IGNORE_INDEX)) for k in LABEL_KEYS}
        return {
            "image": image_t,
            "mask": torch.as_tensor(mask_np, dtype=torch.long),
            "boxes": boxes_t,
            "box_labels": torch.zeros((len(boxes),), dtype=torch.long),
            "labels": labels,
            "dataset": s.dataset,
            "image_path": s.image_path,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate with variable-length boxes kept as lists (FCOS assigns per-image)."""
    images = torch.stack([b["image"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    labels = {k: torch.tensor([b["labels"][k] for b in batch], dtype=torch.long) for k in LABEL_KEYS}
    return {
        "image": images,
        "mask": masks,
        "boxes": [b["boxes"] for b in batch],
        "box_labels": [b["box_labels"] for b in batch],
        "labels": labels,
        "dataset": [b["dataset"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
    }
