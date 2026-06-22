"""Segmentation metrics: Dice, IoU, 95th-percentile Hausdorff, ASSD."""

from __future__ import annotations

import numpy as np


def _bin(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0.5


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    p, g = _bin(pred), _bin(gt)
    inter = np.logical_and(p, g).sum()
    return float((2 * inter + eps) / (p.sum() + g.sum() + eps))


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    p, g = _bin(pred), _bin(gt)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float((inter + eps) / (union + eps))


def _surface_points(mask: np.ndarray) -> np.ndarray:
    """Coordinates of boundary voxels of a binary mask."""
    m = _bin(mask)
    if not m.any():
        return np.empty((0, m.ndim))
    try:
        from scipy import ndimage

        eroded = ndimage.binary_erosion(m, iterations=1, border_value=0)
        border = m & ~eroded
    except Exception:  # pragma: no cover
        border = m & ~_np_erode(m)
    return np.argwhere(border)


def _np_erode(m: np.ndarray) -> np.ndarray:
    e = m.copy()
    e[1:, :] &= m[:-1, :]
    e[:-1, :] &= m[1:, :]
    e[:, 1:] &= m[:, :-1]
    e[:, :-1] &= m[:, 1:]
    return e


def _symmetric_surface_distances(pred: np.ndarray, gt: np.ndarray):
    """Nearest-neighbour surface distances pred->gt and gt->pred."""
    sp = _surface_points(pred)
    sg = _surface_points(gt)
    if len(sp) == 0 or len(sg) == 0:
        return None, None
    try:
        from scipy.spatial import cKDTree

        d_pg, _ = cKDTree(sg).query(sp)
        d_gp, _ = cKDTree(sp).query(sg)
    except Exception:  # pragma: no cover
        d_pg = np.sqrt(((sp[:, None, :] - sg[None, :, :]) ** 2).sum(-1)).min(1)
        d_gp = np.sqrt(((sg[:, None, :] - sp[None, :, :]) ** 2).sum(-1)).min(1)
    return d_pg, d_gp


def hd95(pred: np.ndarray, gt: np.ndarray, spacing: float = 1.0) -> float:
    """95th-percentile symmetric Hausdorff distance (pixels * spacing)."""
    d_pg, d_gp = _symmetric_surface_distances(pred, gt)
    if d_pg is None:
        p, g = _bin(pred), _bin(gt)
        return 0.0 if (p == g).all() else float("inf")
    both = np.concatenate([d_pg, d_gp])
    return float(np.percentile(both, 95) * spacing)


def assd(pred: np.ndarray, gt: np.ndarray, spacing: float = 1.0) -> float:
    """Average Symmetric Surface Distance."""
    d_pg, d_gp = _symmetric_surface_distances(pred, gt)
    if d_pg is None:
        p, g = _bin(pred), _bin(gt)
        return 0.0 if (p == g).all() else float("inf")
    return float((d_pg.sum() + d_gp.sum()) / (len(d_pg) + len(d_gp)) * spacing)
