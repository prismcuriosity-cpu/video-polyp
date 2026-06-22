"""Concrete dataset source loaders (modular & swappable).

Each loader discovers image/mask (and, where available, box/label) pairs on disk.
Real layouts are implemented for Kvasir-SEG, PICCOLO, CVC-ClinicDB/ColonDB and
HyperKvasir; GastroVision is a classification placeholder.  To add or relocate a
dataset, point its ``root`` in ``configs/data/*.yaml`` — no code change needed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from polypai.data.registry import Sample, register_dataset

_IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _find_dir(root: Path, names) -> Path | None:
    for n in names:
        for cand in (root / n, root / n.lower(), root / n.upper()):
            if cand.is_dir():
                return cand
    # case-insensitive scan
    lowered = {p.name.lower(): p for p in root.iterdir() if p.is_dir()} if root.is_dir() else {}
    for n in names:
        if n.lower() in lowered:
            return lowered[n.lower()]
    return None


def _list_images(d: Path) -> list[Path]:
    return sorted([p for p in d.iterdir() if p.suffix.lower() in _IMG_EXT]) if d and d.is_dir() else []


def _match_mask(stem: str, mask_dir: Path) -> Path | None:
    if mask_dir is None:
        return None
    for ext in _IMG_EXT:
        for cand in (mask_dir / f"{stem}{ext}", mask_dir / f"{stem}_mask{ext}"):
            if cand.exists():
                return cand
    return None


def _hash_split(stem: str, split: str, ratios=(0.8, 0.1)) -> bool:
    """Deterministic per-filename split for datasets without an official one."""
    h = int(hashlib.md5(stem.encode()).hexdigest(), 16) % 1000 / 1000.0
    train_max, val_max = ratios[0], ratios[0] + ratios[1]
    if split in ("train",):
        return h < train_max
    if split in ("val", "validation"):
        return train_max <= h < val_max
    if split in ("test",):
        return h >= val_max
    return True  # split == "all"


def _pair_dataset(root: Path, split: str, img_names, mask_names, official_split: bool):
    img_dir = _find_dir(root, img_names)
    mask_dir = _find_dir(root, mask_names)
    samples: list[Sample] = []
    for img in _list_images(img_dir):
        if not official_split and not _hash_split(img.stem, split):
            continue
        samples.append(
            Sample(image_path=str(img), mask_path=str(_match_mask(img.stem, mask_dir) or "") or None)
        )
    return samples


@register_dataset("kvasir_seg", has_masks=True, note="1000 polyp images + masks")
def load_kvasir_seg(root: Path, split: str) -> list[Sample]:
    # layout: <root>/images/*.jpg , <root>/masks/*.jpg
    return _pair_dataset(root, split, ["images", "image"], ["masks", "mask"], official_split=False)


@register_dataset("piccolo", has_masks=True, has_boxes=False,
                  label_keys=("paris", "nice"), note="white-light + NBI, Paris/NICE labels")
def load_piccolo(root: Path, split: str) -> list[Sample]:
    # layout: <root>/<split>/polyps/*.png , <root>/<split>/masks/*.png  (+ metadata)
    split_dir = _find_dir(root, [split, {"val": "validation"}.get(split, split)])
    base = split_dir or root
    img_dir = _find_dir(base, ["polyps", "images", "frames"])
    mask_dir = _find_dir(base, ["masks", "ground_truth"])
    samples: list[Sample] = []
    for img in _list_images(img_dir):
        samples.append(Sample(image_path=str(img),
                              mask_path=str(_match_mask(img.stem, mask_dir) or "") or None,
                              meta={"modality": "nbi" if "nbi" in img.stem.lower() else "wl"}))
    return samples


@register_dataset("cvc_clinicdb", has_masks=True, note="612 frames from 31 sequences")
def load_cvc_clinicdb(root: Path, split: str) -> list[Sample]:
    return _pair_dataset(root, split, ["Original", "images", "PNG/Original"],
                         ["Ground Truth", "masks", "PNG/Ground Truth"], official_split=False)


@register_dataset("cvc_colondb", has_masks=True, note="380 frames")
def load_cvc_colondb(root: Path, split: str) -> list[Sample]:
    return _pair_dataset(root, split, ["images", "Original"], ["masks", "Ground Truth"],
                         official_split=False)


@register_dataset("hyperkvasir_seg", has_masks=True, note="1000 segmentation frames")
def load_hyperkvasir(root: Path, split: str) -> list[Sample]:
    return _pair_dataset(root, split, ["images", "segmented-images/images"],
                         ["masks", "segmented-images/masks"], official_split=False)


@register_dataset("gastrovision", has_masks=False, label_keys=("malignancy",),
                  note="PLACEHOLDER: multi-class GI classification; insert link in config")
def load_gastrovision(root: Path, split: str) -> list[Sample]:
    # classification-only placeholder: <root>/<class_name>/*.jpg
    samples: list[Sample] = []
    if not root.is_dir():
        return samples
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in _list_images(cls_dir):
            if _hash_split(img.stem, split):
                samples.append(Sample(image_path=str(img), labels={}, meta={"class": cls_dir.name}))
    return samples
