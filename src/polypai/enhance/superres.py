"""Super-resolution pre-pass interface for diminutive-lesion detection (Module 2).

A learned SR model (e.g. a lightweight ESRGAN/SwinIR distilled for endoscopy) can
be dropped in by implementing :class:`SuperResolver`.  Until trained weights are
available, :func:`lanczos_upsample` provides a deterministic high-quality
fallback so the rest of the pipeline (tiling, multi-scale detection) is testable.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


@runtime_checkable
class SuperResolver(Protocol):
    def upscale(self, img: np.ndarray, scale: int = 2) -> np.ndarray:  # pragma: no cover
        ...


def lanczos_upsample(img: np.ndarray, scale: int = 2) -> np.ndarray:
    """Deterministic Lanczos (or bilinear fallback) upsampling by an integer scale."""
    h, w = img.shape[:2]
    if _HAS_CV2:
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
    from polypai.frame_selection.imageops import resize

    return resize(img, (h * scale, w * scale))


class LanczosSuperResolver:
    """Trivial :class:`SuperResolver` used as the default until SR weights exist."""

    def upscale(self, img: np.ndarray, scale: int = 2) -> np.ndarray:
        return lanczos_upsample(img, scale)
