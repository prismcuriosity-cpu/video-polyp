"""Vascular-pattern visibility scoring (brief §4).

Vessel / capillary architecture drives NICE classification and histology
prediction, so a frame that hides the microvasculature (glare, blur, distance)
is diagnostically poor even if otherwise crisp.  We quantify visible tubular
structure with a multi-scale Frangi vesselness response restricted to the lesion
ROI.

``scikit-image``'s well-validated ``frangi`` is used when available; a compact,
dependency-free NumPy Hessian-eigenvalue implementation is the fallback so the
score is always computable.
"""

from __future__ import annotations

import numpy as np

from polypai.frame_selection.imageops import gaussian_blur, saturating_norm, to_gray

try:
    from skimage.filters import frangi as _sk_frangi  # type: ignore

    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover
    _HAS_SKIMAGE = False


def _hessian(gray: np.ndarray, sigma: float):
    """Scale-normalised Hessian (Lxx, Lxy, Lyy) at scale ``sigma``."""
    g = gaussian_blur(gray, sigma)
    gy, gx = np.gradient(g)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    norm = sigma**2  # gamma-normalisation (gamma=1) keeps response scale-comparable
    return norm * gxx, norm * 0.5 * (gxy + gyx), norm * gyy


def _frangi_numpy(gray: np.ndarray, scales, beta: float = 0.5, c: float = 0.15) -> np.ndarray:
    out = np.zeros_like(gray, dtype=np.float64)
    for s in scales:
        lxx, lxy, lyy = _hessian(gray, s)
        tmp = np.sqrt(np.maximum((lxx - lyy) ** 2 + 4 * lxy**2, 0.0))
        lam1 = 0.5 * (lxx + lyy - tmp)
        lam2 = 0.5 * (lxx + lyy + tmp)
        # order by magnitude: |a1| <= |a2|
        a1 = np.where(np.abs(lam1) <= np.abs(lam2), lam1, lam2)
        a2 = np.where(np.abs(lam1) <= np.abs(lam2), lam2, lam1)
        rb = a1 / (a2 + 1e-10)
        struct = np.sqrt(a1**2 + a2**2)
        vness = np.exp(-(rb**2) / (2 * beta**2)) * (1.0 - np.exp(-(struct**2) / (2 * c**2)))
        # vessels in endoscopy are dark tubular structures -> a2 > 0; but we keep
        # an absolute response so the score is robust to NBI/WL polarity.
        out = np.maximum(out, vness)
    return out


def vesselness_map(roi: np.ndarray, scales=(1.0, 2.0, 3.0)) -> np.ndarray:
    """Per-pixel vesselness response in [0, 1]-ish range for the lesion crop."""
    gray = to_gray(roi)
    if gray.size < 16:
        return np.zeros_like(gray, dtype=np.float64)
    if _HAS_SKIMAGE:
        # skimage frangi already integrates over the sigma range
        resp = _sk_frangi(gray, sigmas=scales, black_ridges=True)
        return np.nan_to_num(resp)
    return _frangi_numpy(gray, scales)


def vascular_visibility_score(roi: np.ndarray, scale: float = 0.02) -> float:
    """Aggregate vascular-pattern visibility in [0, 1].

    Combines the *strength* (mean response) and *extent* (fraction of the ROI
    occupied by tubular structure) of the vesselness map.
    """
    resp = vesselness_map(roi)
    if resp.size == 0:
        return 0.0
    strength = saturating_norm(float(resp.mean()), scale)
    thr = max(1e-4, 0.5 * float(resp.max()))
    extent = float((resp >= thr).mean())
    return float(0.6 * strength + 0.4 * extent)
