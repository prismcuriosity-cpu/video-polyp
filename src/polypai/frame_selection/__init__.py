"""Polyp-centric Informative Frame Selection (PC-IFS).

This is *not* generic key-frame extraction.  Every score is computed **inside
the polyp ROI**, so a globally blurry frame with a crisp, well-exposed lesion is
preferred over a globally sharp frame whose lesion is washed out by specular
glare.  The module implements the pipeline described in the project brief:

    video -> candidate polyp frames -> reject non-polyp -> ROI diagnostic
    scoring -> temporal peak selection -> near-duplicate removal ->
    multi-angle view selection -> top-K informative frames

The deterministic, classical-CV scorers (quality, texture, vascular, boundary,
perceptual hashing, temporal NMS) are pure NumPy and fully unit-tested.  A
learned candidate detector (Module 2) and a learned diagnostic-quality head can
be injected through the :class:`~polypai.frame_selection.candidate.CandidateDetector`
protocol without changing the orchestration logic.
"""

from __future__ import annotations

from polypai.frame_selection.scoring import (
    DiagnosticScoreWeights,
    score_frame,
)
from polypai.frame_selection.selector import (
    InformativeFrameSelector,
    SelectorConfig,
)

__all__ = [
    "DiagnosticScoreWeights",
    "score_frame",
    "InformativeFrameSelector",
    "SelectorConfig",
]
