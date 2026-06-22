"""Diagnostic-information score (brief §7).

Implements the medical-relevance score

    S = alpha*P + beta*V + gamma*T + delta*Q + epsilon*B  (+ zeta*Vasc)

where every term is ROI-localised and in [0, 1].  The vascular term is an
explicit extension of the brief's formula because NICE classification depends on
microvascular visibility; it can be disabled by setting ``zeta = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polypai.frame_selection.imageops import crop_roi
from polypai.frame_selection.quality import assess_roi_quality
from polypai.frame_selection.texture import texture_clarity_score
from polypai.frame_selection.vascular import vascular_visibility_score
from polypai.frame_selection.visibility import (
    boundary_completeness_score,
    visibility_score,
)
from polypai.structures import FrameScore, PolypCandidate


@dataclass
class DiagnosticScoreWeights:
    """Weights (alpha..zeta) for the diagnostic-information score."""

    alpha: float = 0.30   # P  - polyp presence confidence
    beta: float = 0.22    # V  - visibility
    gamma: float = 0.18   # T  - texture / pit-pattern clarity
    delta: float = 0.15   # Q  - ROI image quality
    epsilon: float = 0.10  # B  - boundary completeness
    zeta: float = 0.05    # Vasc - vascular-pattern visibility (extension)

    def normalised(self) -> "DiagnosticScoreWeights":
        vals = np.array([self.alpha, self.beta, self.gamma, self.delta, self.epsilon, self.zeta])
        s = vals.sum()
        if s <= 0:
            return DiagnosticScoreWeights()
        vals = vals / s
        return DiagnosticScoreWeights(*vals.tolist())


def score_frame(
    frame: np.ndarray,
    candidate: PolypCandidate,
    frame_index: int = 0,
    weights: DiagnosticScoreWeights | None = None,
    roi_pad: float = 0.10,
) -> FrameScore:
    """Compute the full :class:`FrameScore` for one frame given its best candidate.

    All sub-scores are evaluated on the lesion ROI (box padded by ``roi_pad``),
    honouring the brief's rule that *global* quality is irrelevant when the
    lesion region is good.
    """
    w = (weights or DiagnosticScoreWeights()).normalised()
    roi = crop_roi(frame, candidate.bbox, pad=roi_pad)

    P = float(np.clip(candidate.presence, 0.0, 1.0))
    V = visibility_score(frame, candidate.bbox, candidate.mask)
    T = texture_clarity_score(roi)
    qb = assess_roi_quality(roi)
    Q = qb.aggregate()
    B = boundary_completeness_score(frame, candidate.bbox, candidate.mask)
    Vasc = vascular_visibility_score(roi)

    S = (
        w.alpha * P + w.beta * V + w.gamma * T
        + w.delta * Q + w.epsilon * B + w.zeta * Vasc
    )

    return FrameScore(
        frame_index=frame_index,
        presence=P,
        visibility=V,
        texture=T,
        quality=Q,
        boundary=B,
        vascular=Vasc,
        total=float(S),
        candidate=candidate,
        quality_breakdown=qb,
    )
