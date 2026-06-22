"""ROI-localised image-quality assessment (brief §5).

The clinical requirement is explicit: *bad global frame quality is acceptable if
the polyp region quality is good*.  Every metric below therefore operates on the
lesion crop (optionally masked) rather than the whole frame.

All scores are returned in [0, 1] where **higher is better**.
"""

from __future__ import annotations

import numpy as np

from polypai.frame_selection.imageops import (
    laplacian,
    saturating_norm,
    to_gray,
    to_hsv,
)
from polypai.structures import QualityBreakdown


def sharpness_score(roi: np.ndarray, scale: float = 0.0015) -> float:
    """Inverse motion blur via variance-of-Laplacian (Pech-Pacheco focus measure).

    A sharp lesion has high high-frequency energy.  We normalise with a
    dataset-free saturating function so the score is comparable across frames of
    differing brightness.
    """
    gray = to_gray(roi)
    if gray.size < 9:
        return 0.0
    lap = laplacian(gray)
    return saturating_norm(float(lap.var()), scale)


def contrast_score(roi: np.ndarray, scale: float = 0.06) -> float:
    """RMS (standard-deviation) contrast of the lesion luminance."""
    gray = to_gray(roi)
    if gray.size == 0:
        return 0.0
    return saturating_norm(float(gray.std()), scale)


def specular_free_score(roi: np.ndarray, v_thr: float = 0.92, s_thr: float = 0.20) -> float:
    """1 - fraction of specular-highlight pixels over the lesion.

    Specular reflections are bright (high V) and desaturated (low S); they erase
    pit/vascular structure and should penalise a frame.
    """
    hsv = to_hsv(roi)
    if hsv.size == 0:
        return 1.0
    v, s = hsv[..., 2], hsv[..., 1]
    specular = (v >= v_thr) & (s <= s_thr)
    frac = float(specular.mean())
    return float(1.0 - frac)


def exposure_score(roi: np.ndarray, low: float = 0.04, high: float = 0.96) -> float:
    """1 - fraction of over/under-exposed pixels over the lesion."""
    gray = to_gray(roi)
    if gray.size == 0:
        return 0.0
    clipped = (gray <= low) | (gray >= high)
    return float(1.0 - clipped.mean())


def noise_free_score(roi: np.ndarray, scale: float = 0.02) -> float:
    """Inverse of the Immerkaer fast noise-variance estimate.

    Uses the 3x3 mask N = [[1,-2,1],[-2,4,-2],[1,-2,1]] which is blind to
    smoothly varying signal, so it isolates additive sensor noise.
    """
    gray = to_gray(roi)
    h, w = gray.shape[:2]
    if h < 3 or w < 3:
        return 1.0
    g = np.pad(gray, 1, mode="edge")
    response = (
        1 * g[:-2, :-2] - 2 * g[:-2, 1:-1] + 1 * g[:-2, 2:]
        - 2 * g[1:-1, :-2] + 4 * g[1:-1, 1:-1] - 2 * g[1:-1, 2:]
        + 1 * g[2:, :-2] - 2 * g[2:, 1:-1] + 1 * g[2:, 2:]
    )
    sigma = np.sqrt(np.pi / 2.0) / (6.0 * (w - 2) * (h - 2)) * float(np.abs(response).sum())
    return float(1.0 - saturating_norm(sigma, scale))


def assess_roi_quality(roi: np.ndarray) -> QualityBreakdown:
    """Compute the full :class:`QualityBreakdown` for a lesion crop."""
    return QualityBreakdown(
        sharpness=sharpness_score(roi),
        contrast=contrast_score(roi),
        specular_free=specular_free_score(roi),
        exposure=exposure_score(roi),
        noise_free=noise_free_score(roi),
    )
