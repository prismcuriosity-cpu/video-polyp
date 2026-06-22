"""Pseudo-NBI (Narrow-Band Imaging) synthesis from white-light frames.

Real NBI illuminates tissue with narrow 415 nm (blue) and 540 nm (green) bands.
Haemoglobin absorbs strongly there, so superficial capillaries appear brown and
deeper vessels cyan against a green-tan mucosa — exactly the contrast NICE
classification relies on.  White-light colonoscopy lacks this, so we synthesise
a perceptually similar rendering:

1. Channel remap emphasising the green/blue absorption bands and suppressing red.
2. Vesselness-guided darkening so tubular structures are pushed toward brown.
3. CLAHE-style local contrast enhancement of the mucosal pattern.

This is an explicit, inspectable transform (no training), which makes it a
reproducible front-end for the learned NICE head and a useful data-augmentation
that bridges the white-light/NBI domain gap.
"""

from __future__ import annotations

import numpy as np

from polypai.frame_selection.imageops import as_float01
from polypai.frame_selection.vascular import vesselness_map

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


def _clahe(gray: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    g = np.clip(gray, 0, 1)
    if _HAS_CV2:
        c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        return c.apply((g * 255).astype(np.uint8)).astype(np.float32) / 255.0
    # global histogram equalisation fallback
    hist, bins = np.histogram(g.flatten(), 256, (0, 1))
    cdf = hist.cumsum().astype(np.float32)
    cdf = cdf / max(cdf[-1], 1e-8)
    return np.interp(g.flatten(), bins[:-1], cdf).reshape(g.shape).astype(np.float32)


def pseudo_nbi(rgb: np.ndarray, vessel_gain: float = 0.6, contrast: bool = True) -> np.ndarray:
    """Render a white-light RGB frame as a pseudo-NBI RGB image in [0, 1].

    The mapping keeps the green channel (mucosal pattern), promotes blue
    (superficial capillaries), suppresses red (haemoglobin absorption band), then
    darkens detected vessels to mimic NBI's brown/cyan vascular contrast.
    """
    img = as_float01(rgb)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    # spectral remap toward the 415/540 nm bands
    nbi_r = np.clip(0.30 * r + 0.55 * g + 0.15 * b, 0, 1)
    nbi_g = np.clip(0.10 * r + 0.70 * g + 0.20 * b, 0, 1)
    nbi_b = np.clip(0.05 * r + 0.35 * g + 0.60 * b, 0, 1)

    if contrast:
        nbi_g = _clahe(nbi_g)
        nbi_b = _clahe(nbi_b)

    out = np.stack([nbi_r, nbi_g, nbi_b], axis=-1)

    # vesselness-guided darkening -> capillaries trend brown/cyan
    vmap = vesselness_map(img)
    if vmap.max() > 1e-8:
        vmap = vmap / vmap.max()
    darken = 1.0 - vessel_gain * vmap[..., None]
    out = np.clip(out * darken, 0, 1)
    return out.astype(np.float32)


def vessel_density(rgb_or_nbi: np.ndarray, thresh_rel: float = 0.4) -> float:
    """Fraction of pixels occupied by vascular structure (NICE-relevant feature)."""
    vmap = vesselness_map(as_float01(rgb_or_nbi))
    if vmap.max() <= 1e-8:
        return 0.0
    return float((vmap >= thresh_rel * vmap.max()).mean())
