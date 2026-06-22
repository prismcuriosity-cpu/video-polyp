"""Tests for the modular dataset layer using a synthetic on-disk layout."""

import numpy as np
import pytest

from polypai.data import list_datasets, resolve_samples

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from polypai.data.base import PolypDataset, collate_fn, mask_to_boxes  # noqa: E402


def _make_kvasir(root, n=6):
    img_dir = root / "images"
    mask_dir = root / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for i in range(n):
        img = (np.random.rand(64, 80, 3) * 255).astype(np.uint8)
        mask = np.zeros((64, 80), np.uint8)
        mask[20:40, 25:55] = 255  # one polyp blob
        cv2.imwrite(str(img_dir / f"f{i}.png"), img)
        cv2.imwrite(str(mask_dir / f"f{i}.png"), mask)


def test_registry_has_expected_sources():
    names = list_datasets()
    for expected in ["kvasir_seg", "piccolo", "cvc_clinicdb", "cvc_colondb", "gastrovision"]:
        assert expected in names


def test_missing_root_returns_empty(tmp_path):
    with pytest.warns(UserWarning):
        assert resolve_samples("kvasir_seg", tmp_path / "nope", "train") == []


def test_kvasir_loader_and_dataset(tmp_path):
    root = tmp_path / "Kvasir-SEG"
    _make_kvasir(root, n=8)
    samples = resolve_samples("kvasir_seg", root, "all")
    assert len(samples) == 8
    assert all(s.dataset == "kvasir_seg" for s in samples)

    ds = PolypDataset(samples, train=False)
    item = ds[0]
    assert item["image"].ndim == 3 and item["image"].shape[0] == 3
    assert item["mask"].ndim == 2
    assert item["boxes"].shape[1] == 4 and item["boxes"].shape[0] >= 1  # box from mask


def test_mask_to_boxes_counts_components():
    mask = np.zeros((50, 50), np.uint8)
    mask[5:15, 5:15] = 1
    mask[30:45, 30:45] = 1
    boxes = mask_to_boxes(mask)
    assert len(boxes) == 2


def test_collate_batches(tmp_path):
    root = tmp_path / "Kvasir-SEG"
    _make_kvasir(root, n=4)
    ds = PolypDataset(resolve_samples("kvasir_seg", root, "all"), train=False)
    batch = collate_fn([ds[0], ds[1]])
    assert batch["image"].shape[0] == 2
    assert isinstance(batch["boxes"], list) and len(batch["boxes"]) == 2
    assert batch["labels"]["paris"].shape[0] == 2


def test_split_partitioning_is_disjoint(tmp_path):
    root = tmp_path / "Kvasir-SEG"
    _make_kvasir(root, n=40)
    tr = {s.image_path for s in resolve_samples("kvasir_seg", root, "train")}
    va = {s.image_path for s in resolve_samples("kvasir_seg", root, "val")}
    te = {s.image_path for s in resolve_samples("kvasir_seg", root, "test")}
    assert tr and not (tr & va) and not (tr & te) and not (va & te)
