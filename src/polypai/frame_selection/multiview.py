"""Multi-angle view selection (brief §8).

Rather than returning the K globally highest-scoring frames (which tend to be
near-identical "best" views), we keep a *diverse* set covering frontal, side,
close-up and boundary views.  Diversity is measured in a view-descriptor space
(geometry + appearance) and selected by farthest-point sampling seeded at the
highest-scoring frame, so each kept frame is both high quality and complementary.
"""

from __future__ import annotations

import numpy as np

from polypai.structures import FrameScore


def build_view_descriptor(fs: FrameScore, frame_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Compact view descriptor capturing pose/appearance differences."""
    cand = fs.candidate
    box = cand.bbox if cand is not None else None
    if box is not None and frame_shape is not None:
        h, w = frame_shape
        aspect = box.width / max(box.height, 1.0)
        area_frac = box.area / max(h * w, 1.0)
        cx = box.cx / max(w, 1.0)
        cy = box.cy / max(h, 1.0)
    else:
        aspect, area_frac, cx, cy = 1.0, 0.1, 0.5, 0.5
    geo = np.array([
        np.tanh(aspect - 1.0),          # elongation / orientation proxy
        np.sqrt(min(area_frac, 1.0)),   # distance / close-up proxy
        cx, cy,                          # in-frame position
        fs.texture, fs.vascular, fs.boundary,
    ], dtype=np.float64)
    if cand is not None and cand.embedding is not None:
        emb = np.asarray(cand.embedding, dtype=np.float64)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        geo = np.concatenate([geo, emb])
    return geo


def select_multiview(
    frame_scores: list[FrameScore],
    n_views: int,
    frame_shape: tuple[int, int] | None = None,
) -> tuple[list[FrameScore], dict[int, list[int]]]:
    """Farthest-point select up to ``n_views`` diverse, high-quality frames.

    Returns (selected, view_groups) where view_groups maps each selected frame's
    index to the list of frame indices assigned to its view cluster.
    """
    if not frame_scores:
        return [], {}
    if n_views >= len(frame_scores):
        groups = {fs.frame_index: [fs.frame_index] for fs in frame_scores}
        return list(frame_scores), groups

    descs = np.stack([build_view_descriptor(fs, frame_shape) for fs in frame_scores])
    # standardise dims so no single axis dominates the distance
    std = descs.std(axis=0)
    std[std < 1e-6] = 1.0
    descs = (descs - descs.mean(axis=0)) / std

    seed = int(np.argmax([fs.total for fs in frame_scores]))
    chosen = [seed]
    min_d = np.linalg.norm(descs - descs[seed], axis=1)
    while len(chosen) < n_views:
        nxt = int(np.argmax(min_d))
        if min_d[nxt] <= 1e-9:
            break
        chosen.append(nxt)
        min_d = np.minimum(min_d, np.linalg.norm(descs - descs[nxt], axis=1))

    # assign every frame to nearest chosen view
    chosen_descs = descs[chosen]
    groups: dict[int, list[int]] = {frame_scores[c].frame_index: [] for c in chosen}
    for i, fs in enumerate(frame_scores):
        nearest = int(np.argmin(np.linalg.norm(chosen_descs - descs[i], axis=1)))
        groups[frame_scores[chosen[nearest]].frame_index].append(fs.frame_index)

    selected = sorted((frame_scores[c] for c in chosen), key=lambda fs: -fs.total)
    return selected, groups
