"""Candidate polyp detection front-end (brief §1).

The selector needs, per frame, zero or more :class:`PolypCandidate` objects
(box + presence + optional mask).  In deployment this comes from the trained
Module-2 detector; for that we expose the :class:`CandidateDetector` protocol
and a thin torch adapter (lazy, see :func:`from_torch_detector`).

So the whole pipeline is runnable and testable *without* a trained model, we
also ship :class:`SaliencyCandidateDetector` — a classical center-surround +
redness saliency proposer.  It is deliberately recall-oriented (it over-proposes
and lets the diagnostic scorer/temporal NMS prune), matching the brief's
"reject non-polyp frames downstream" philosophy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np

from polypai.frame_selection.imageops import gaussian_blur, to_gray
from polypai.structures import BBox, PolypCandidate


@runtime_checkable
class CandidateDetector(Protocol):
    """Anything that maps a frame (HxWx3 uint8/float) to candidate polyps."""

    def detect(self, frame: np.ndarray) -> list[PolypCandidate]:  # pragma: no cover - protocol
        ...


def _largest_component(mask: np.ndarray):
    """Return (labelled_mask, label_of_largest, count) using the best backend."""
    try:
        import cv2  # type: ignore

        n, labels = cv2.connectedComponents(mask.astype(np.uint8))
        if n <= 1:
            return labels, 0, 0
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        best = int(counts.argmax())
        return labels, best, int(counts[best])
    except Exception:
        pass
    try:
        from scipy import ndimage  # type: ignore

        labels, n = ndimage.label(mask)
        if n == 0:
            return labels, 0, 0
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        best = int(counts.argmax())
        return labels, best, int(counts[best])
    except Exception:
        # crude fallback: treat the whole thresholded region as one component
        labels = mask.astype(np.int32)
        return labels, 1, int(mask.sum())


class SaliencyCandidateDetector:
    """Classical, training-free polyp-candidate proposer.

    Saliency = Difference-of-Gaussians blob response (protrusions / focal
    structures) combined with a mucosal redness channel.  The largest salient
    region above an adaptive threshold becomes the candidate.
    """

    def __init__(
        self,
        sigma_center: float = 2.0,
        sigma_surround: float = 9.0,
        thr_k: float = 1.0,
        min_area_frac: float = 0.006,
        min_comp_sig: float = 2.0,
    ) -> None:
        self.sigma_center = sigma_center
        self.sigma_surround = sigma_surround
        self.thr_k = thr_k
        self.min_area_frac = min_area_frac
        # A genuine lesion forms a *spatially coherent* salient region, whereas
        # sensor noise produces scattered high pixels.  We therefore gate on the
        # largest connected component's area and its robust significance above
        # background (median/MAD), which separates lesions from noise far better
        # than a single-pixel peak z-score on realistic, textured frames.
        self.min_comp_sig = min_comp_sig

    def saliency_map(self, frame: np.ndarray) -> np.ndarray:
        gray = to_gray(frame)
        dog = gaussian_blur(gray, self.sigma_center) - gaussian_blur(gray, self.sigma_surround)
        sal = np.abs(dog)
        f = np.asarray(frame, dtype=np.float32)
        if f.ndim == 3:
            f = f / (f.max() + 1e-6) if f.max() > 1 else f
            redness = np.clip(f[..., 0] - 0.5 * (f[..., 1] + f[..., 2]), 0, None)
            redness = gaussian_blur(redness, self.sigma_center)
            if redness.max() > 1e-6:
                redness /= redness.max()
            sal = sal / (sal.max() + 1e-8)
            sal = 0.7 * sal + 0.3 * redness
        return sal

    def detect(self, frame: np.ndarray) -> list[PolypCandidate]:
        h, w = frame.shape[:2]
        sal = self.saliency_map(frame)
        med = float(np.median(sal))
        mad = float(np.median(np.abs(sal - med))) * 1.4826 + 1e-8
        if mad < 1e-6:  # essentially constant frame -> no structure
            return []

        thr = med + max(self.thr_k, 1.0) * mad
        binary = sal >= thr
        labels, best, count = _largest_component(binary)
        if count < self.min_area_frac * h * w:  # scattered noise -> reject
            return []
        comp = labels == best

        # robust significance of the coherent region above background
        comp_sig = (float(sal[comp].mean()) - med) / mad
        if comp_sig < self.min_comp_sig:
            return []

        ys, xs = np.where(comp)
        box = BBox(x1=float(xs.min()), y1=float(ys.min()),
                   x2=float(xs.max() + 1), y2=float(ys.max() + 1))

        # presence reflects region significance and lesion extent.
        sig_term = np.clip((comp_sig - self.min_comp_sig) / 6.0, 0.0, 1.0)
        area_term = np.clip((count / (h * w)) / 0.02, 0.0, 1.0)
        presence = float(np.clip(0.3 + 0.45 * sig_term + 0.25 * area_term, 0.0, 1.0))
        mask = comp.astype(np.uint8)
        return [PolypCandidate(bbox=box, presence=presence, mask=mask)]


def from_torch_detector(
    model,
    preprocess: Callable | None = None,
    score_thresh: float = 0.3,
    device: str = "cuda",
) -> CandidateDetector:
    """Adapt a trained Module-2 torch detector into a :class:`CandidateDetector`.

    Imported lazily so this module stays torch-free.  The wrapped ``model`` is
    expected to return an iterable of ``(box_xyxy, score, mask|None)``.
    """
    from polypai.frame_selection._torch_adapter import TorchDetectorAdapter

    return TorchDetectorAdapter(model, preprocess, score_thresh, device)
