"""End-to-end Polyp-centric Informative Frame Selector (PC-IFS).

Orchestrates the full brief pipeline::

    frames -> candidate detection (§1) -> reject non-polyp
           -> ROI diagnostic scoring (§2-5,7) -> temporal NMS / peaks (§6)
           -> near-duplicate removal (§10) -> multi-view selection (§8)
           -> top-K informative frames -> downstream pipeline

The selector is detector-agnostic: pass any :class:`CandidateDetector`
(the classical :class:`SaliencyCandidateDetector` by default, or a trained
Module-2 detector via :func:`from_torch_detector`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np

from polypai.frame_selection.candidate import CandidateDetector, SaliencyCandidateDetector
from polypai.frame_selection.dedup import dedup_by_embedding, dedup_by_phash, phash
from polypai.frame_selection.multiview import select_multiview
from polypai.frame_selection.scoring import DiagnosticScoreWeights, score_frame
from polypai.frame_selection.temporal import find_peak_indices, smooth_scores, temporal_nms
from polypai.structures import FrameScore, SelectionResult


@dataclass
class SelectorConfig:
    """Tunable knobs for the informative-frame selector."""

    weights: DiagnosticScoreWeights = field(default_factory=DiagnosticScoreWeights)
    roi_pad: float = 0.10

    # rejection of non-polyp / poor frames (brief §1)
    min_presence: float = 0.12
    min_total: float = 0.0

    # temporal stability (brief §6)
    smooth_window: int = 5
    restrict_to_peaks: bool = True
    peak_min_distance: int = 4
    temporal_min_distance: int = 6

    # near-duplicate removal (brief §10)
    compute_phash: bool = True
    max_hamming: int = 8
    embedding_sim_thresh: float = 0.92

    # output budget
    top_k: int = 12
    n_views: int = 5
    enforce_multiview: bool = True


class InformativeFrameSelector:
    def __init__(self, detector: CandidateDetector | None = None, config: SelectorConfig | None = None):
        self.detector = detector or SaliencyCandidateDetector()
        self.cfg = config or SelectorConfig()

    # ---- per-frame scoring -------------------------------------------------
    def score_frame_image(self, frame: np.ndarray, frame_index: int) -> FrameScore | None:
        candidates = self.detector.detect(frame)
        if not candidates:
            return None
        best = max(candidates, key=lambda c: c.presence)
        fs = score_frame(frame, best, frame_index, self.cfg.weights, self.cfg.roi_pad)
        fs.view_descriptor = None
        if self.cfg.compute_phash:
            fs.phash = phash(frame)
        return fs

    def score_video(self, frames: Iterable[np.ndarray]) -> tuple[list[FrameScore], list[int], tuple[int, int]]:
        """Score every frame; return (kept_scores, rejected_indices, frame_shape)."""
        kept: list[FrameScore] = []
        rejected: list[int] = []
        frame_shape = (0, 0)
        for idx, frame in enumerate(frames):
            if frame_shape == (0, 0):
                frame_shape = frame.shape[:2]
            fs = self.score_frame_image(frame, idx)
            if fs is None or fs.presence < self.cfg.min_presence or fs.total < self.cfg.min_total:
                rejected.append(idx)
                continue
            kept.append(fs)
        return kept, rejected, frame_shape

    # ---- full selection ----------------------------------------------------
    def select(self, frames: Iterable[np.ndarray]) -> SelectionResult:
        scored, rejected, frame_shape = self.score_video(frames)
        if not scored:
            return SelectionResult(selected=[], all_scores=[], rejected_indices=rejected)

        candidates = scored
        # 1) temporal peak restriction (peak diagnostic moments, brief §6)
        if self.cfg.restrict_to_peaks and len(scored) > 3:
            order = sorted(scored, key=lambda fs: fs.frame_index)
            totals = smooth_scores(np.array([fs.total for fs in order]), self.cfg.smooth_window)
            peak_pos = set(find_peak_indices(totals, self.cfg.peak_min_distance))
            peaks = [order[i] for i in peak_pos]
            if peaks:
                candidates = peaks

        # 2) temporal NMS (drop redundant adjacent frames)
        candidates = temporal_nms(candidates, self.cfg.temporal_min_distance)

        # 3) near-duplicate removal (perceptual hash + embedding)
        candidates = dedup_by_phash(candidates, max_hamming=self.cfg.max_hamming)
        candidates = dedup_by_embedding(candidates, sim_thresh=self.cfg.embedding_sim_thresh)

        # 4) multi-view diversity / top-K budget
        view_groups: dict[int, list[int]] = {}
        if self.cfg.enforce_multiview and len(candidates) > self.cfg.n_views:
            mv, view_groups = select_multiview(candidates, self.cfg.n_views, frame_shape)
            remaining = [fs for fs in candidates if fs.frame_index not in {m.frame_index for m in mv}]
            remaining.sort(key=lambda fs: -fs.total)
            selected = sorted(mv, key=lambda fs: -fs.total)
            for fs in remaining:
                if len(selected) >= self.cfg.top_k:
                    break
                selected.append(fs)
        else:
            selected = sorted(candidates, key=lambda fs: -fs.total)[: self.cfg.top_k]

        selected = sorted(selected, key=lambda fs: -fs.total)[: self.cfg.top_k]
        return SelectionResult(
            selected=selected,
            all_scores=sorted(scored, key=lambda fs: fs.frame_index),
            rejected_indices=rejected,
            view_groups=view_groups,
        )

    # ---- convenience -------------------------------------------------------
    @staticmethod
    def frames_from_array(video: np.ndarray) -> Iterator[np.ndarray]:
        """Yield frames from a THWC numpy array."""
        for t in range(video.shape[0]):
            yield video[t]
