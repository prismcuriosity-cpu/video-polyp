"""Surface-structure / pit-pattern clarity scoring (brief §3).

These features quantify whether the fine mucosal structure needed for later
Kudo pit-pattern and NICE surface analysis is actually resolvable in the frame.
A frame can be sharp yet textureless (e.g. washed-out mucosa); we reward visible
crypt/gland structure, not raw edge energy.
"""

from __future__ import annotations

import numpy as np

from polypai.frame_selection.imageops import (
    gaussian_blur,
    saturating_norm,
    sobel_grad,
    to_gray,
)


def high_frequency_ratio(gray: np.ndarray, sigma: float = 1.6) -> float:
    """Fraction of signal energy living in the high-frequency band.

    Computed as ``||gray - blur(gray)||^2 / ||gray - mean||^2`` which is scale
    invariant to overall brightness and contrast.
    """
    if gray.size < 9:
        return 0.0
    low = gaussian_blur(gray, sigma)
    high = gray - low
    denom = float(((gray - gray.mean()) ** 2).sum()) + 1e-8
    return float((high**2).sum() / denom)


def gradient_density(gray: np.ndarray, scale: float = 0.05) -> float:
    """Mean Sobel gradient magnitude (structural density)."""
    if gray.size < 9:
        return 0.0
    gx, gy = sobel_grad(gray)
    mag = np.sqrt(gx**2 + gy**2)
    return saturating_norm(float(mag.mean()), scale)


def lbp_entropy(gray: np.ndarray) -> float:
    """Shannon entropy of the 8-neighbour LBP code distribution, normalised.

    High entropy => rich, varied micro-texture (pits/crypts).  Returned in
    [0, 1] by dividing by log2(256).
    """
    h, w = gray.shape[:2]
    if h < 3 or w < 3:
        return 0.0
    g = gray
    c = g[1:-1, 1:-1]
    neigh = [
        g[:-2, :-2], g[:-2, 1:-1], g[:-2, 2:],
        g[1:-1, 2:], g[2:, 2:], g[2:, 1:-1],
        g[2:, :-2], g[1:-1, :-2],
    ]
    code = np.zeros_like(c, dtype=np.int32)
    for i, nb in enumerate(neigh):
        code |= ((nb >= c).astype(np.int32) << i)
    hist = np.bincount(code.ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1e-8)
    nz = p[p > 0]
    ent = float(-(nz * np.log2(nz)).sum())
    return float(np.clip(ent / 8.0, 0.0, 1.0))


def texture_clarity_score(
    roi: np.ndarray,
    weights: tuple[float, float, float] = (0.45, 0.25, 0.30),
) -> float:
    """Composite pit/surface-structure clarity in [0, 1]."""
    gray = to_gray(roi)
    hf = saturating_norm(high_frequency_ratio(gray), scale=0.12)
    gd = gradient_density(gray)
    le = lbp_entropy(gray)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return float(w[0] * hf + w[1] * gd + w[2] * le)
