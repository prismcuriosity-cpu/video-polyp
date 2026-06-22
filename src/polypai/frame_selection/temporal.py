"""Temporal stability analysis and peak-moment selection (brief §6).

Given the per-frame diagnostic scores along the video, we (a) smooth them to
suppress single-frame jitter, (b) locate the *peak diagnostic moments* (local
maxima of the smoothed score), and (c) run temporal non-maximum suppression so
adjacent near-identical frames are not all selected.
"""

from __future__ import annotations

import numpy as np

from polypai.structures import FrameScore


def smooth_scores(scores: np.ndarray, window: int = 5) -> np.ndarray:
    """Edge-aware moving-average smoothing of a 1-D score sequence."""
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0 or window <= 1:
        return scores
    window = min(window, n if n % 2 == 1 else n - 1)
    window = max(1, window | 1)  # force odd
    pad = window // 2
    padded = np.pad(scores, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def find_peak_indices(scores: np.ndarray, min_distance: int = 3, rel_height: float = 0.0) -> list[int]:
    """Indices of local maxima separated by at least ``min_distance`` frames."""
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return []
    if n <= 2:
        return [int(np.argmax(scores))]
    thr = scores.min() + rel_height * (scores.max() - scores.min())
    cand = [i for i in range(1, n - 1)
            if scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1] and scores[i] >= thr]
    if scores[0] >= scores[1] and scores[0] >= thr:
        cand.insert(0, 0)
    if scores[-1] >= scores[-2] and scores[-1] >= thr:
        cand.append(n - 1)
    # enforce min spacing, keeping higher peaks
    cand.sort(key=lambda i: -scores[i])
    chosen: list[int] = []
    for i in cand:
        if all(abs(i - j) >= min_distance for j in chosen):
            chosen.append(i)
    return sorted(chosen)


def temporal_nms(
    frame_scores: list[FrameScore],
    min_distance: int = 6,
    max_keep: int | None = None,
) -> list[FrameScore]:
    """Greedy temporal NMS over frame indices, keeping highest-scoring frames.

    Returns frames sorted by descending total score.  A selected frame suppresses
    any other frame whose ``frame_index`` is within ``min_distance``.
    """
    ordered = sorted(frame_scores, key=lambda fs: -fs.total)
    kept: list[FrameScore] = []
    for fs in ordered:
        if all(abs(fs.frame_index - k.frame_index) >= min_distance for k in kept):
            kept.append(fs)
            if max_keep is not None and len(kept) >= max_keep:
                break
    return kept
