"""Polyp visibility (brief §2) and boundary completeness scoring.

These scores answer "is the whole lesion well-presented in this frame?" from the
candidate box/mask geometry plus local edge evidence:

* **Visibility (V)** rewards a centred lesion of sufficient pixel area that is
  not cut off by the image border and not occluded.
* **Boundary completeness (B)** rewards a lesion whose full perimeter is inside
  the frame and whose edge is crisp (high gradient across the contour).
"""

from __future__ import annotations

import numpy as np

from polypai.frame_selection.imageops import sobel_grad, to_gray
from polypai.structures import BBox


def _binary_erode(mask: np.ndarray) -> np.ndarray:
    """4-neighbour binary erosion (no SciPy dependency)."""
    m = mask.astype(bool)
    e = m.copy()
    e[1:, :] &= m[:-1, :]
    e[:-1, :] &= m[1:, :]
    e[:, 1:] &= m[:, :-1]
    e[:, :-1] &= m[:, 1:]
    return e


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """Boolean mask of contour (morphological gradient) pixels."""
    m = mask.astype(bool)
    return m & ~_binary_erode(m)


def _border_touch_fraction(mask: np.ndarray, margin: int = 2) -> float:
    """Fraction of lesion contour pixels lying within ``margin`` px of any edge."""
    contour = boundary_pixels(mask)
    if contour.sum() == 0:
        return 1.0
    h, w = mask.shape
    near = np.zeros_like(mask, dtype=bool)
    near[:margin, :] = near[-margin:, :] = True
    near[:, :margin] = near[:, -margin:] = True
    return float((contour & near).sum() / max(contour.sum(), 1))


def visibility_score(
    frame: np.ndarray,
    box: BBox,
    mask: np.ndarray | None = None,
    target_area_frac: float = 0.18,
) -> float:
    """Polyp visibility V in [0, 1]."""
    h, w = frame.shape[:2]
    frame_area = float(h * w)

    # 1) Area adequacy: too-small (far away) is bad; reward up to target then keep.
    if mask is not None and mask.any():
        area_frac = float(mask.sum()) / frame_area
    else:
        area_frac = box.area / frame_area
    area_term = float(np.clip(area_frac / target_area_frac, 0.0, 1.0))
    # gentle penalty if the lesion fills almost the entire frame (likely cut off)
    if area_frac > 0.7:
        area_term *= float(np.clip((1.0 - area_frac) / 0.3, 0.3, 1.0))

    # 2) Centeredness: distance of box centre from frame centre.
    dx = (box.cx - w / 2) / (w / 2)
    dy = (box.cy - h / 2) / (h / 2)
    centered = float(1.0 - np.clip(np.hypot(dx, dy) / np.sqrt(2), 0.0, 1.0))

    # 3) Border cut-off penalty from the bounding box.
    m = 0.01 * max(h, w)
    touches = (box.x1 <= m) or (box.y1 <= m) or (box.x2 >= w - m) or (box.y2 >= h - m)
    border_term = 0.55 if touches else 1.0

    # 4) Occlusion / completeness from mask solidity (mask area vs bbox area).
    if mask is not None and mask.any():
        solidity = float(mask.sum()) / max(box.area, 1.0)
        occ_term = float(np.clip(solidity, 0.3, 1.0))
    else:
        occ_term = 1.0

    score = 0.40 * area_term + 0.25 * centered + 0.20 * border_term + 0.15 * occ_term
    return float(np.clip(score, 0.0, 1.0))


def boundary_completeness_score(
    frame: np.ndarray,
    box: BBox,
    mask: np.ndarray | None = None,
) -> float:
    """Boundary completeness B in [0, 1]."""
    h, w = frame.shape[:2]
    gray = to_gray(frame)
    gx, gy = sobel_grad(gray)
    grad_mag = np.sqrt(gx**2 + gy**2)

    if mask is not None and mask.any():
        # (a) perimeter inside the frame
        inside = 1.0 - _border_touch_fraction(mask)
        # (b) edge crispness across the contour
        contour = boundary_pixels(mask)
        if contour.sum() > 0:
            edge = float(grad_mag[contour].mean())
            crisp = edge / (edge + 0.05)
        else:
            crisp = 0.0
        return float(np.clip(0.6 * inside + 0.4 * crisp, 0.0, 1.0))

    # Box-only fallback: distance to border + edge energy on the box ring.
    m = 0.01 * max(h, w)
    margins = [box.x1 - 0, box.y1 - 0, (w - box.x2), (h - box.y2)]
    inside = float(np.clip(min(margins) / (0.1 * max(h, w) + 1e-6), 0.0, 1.0))
    x1, y1 = int(np.clip(box.x1, 0, w - 1)), int(np.clip(box.y1, 0, h - 1))
    x2, y2 = int(np.clip(box.x2, x1 + 1, w)), int(np.clip(box.y2, y1 + 1, h))
    ring = grad_mag[y1:y2, x1:x2]
    edge = float(ring.mean()) if ring.size else 0.0
    crisp = edge / (edge + 0.05)
    return float(np.clip(0.6 * inside + 0.4 * crisp, 0.0, 1.0))
