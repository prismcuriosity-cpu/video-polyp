"""Lightweight image operations used by the ROI scorers.

Pure-NumPy implementations are provided for everything so the scorers run in CI
without OpenCV.  When OpenCV is importable it is used for the few heavy ops
(colour conversion, separable blur) purely for speed; results are equivalent.
"""

from __future__ import annotations

import numpy as np

try:  # optional acceleration only
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:  # pragma: no cover - exercised only on machines without cv2
    _HAS_CV2 = False


def as_float01(img: np.ndarray) -> np.ndarray:
    """Return ``img`` as float32 in [0, 1]."""
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    img = img.astype(np.float32)
    mx = float(img.max()) if img.size else 1.0
    if mx > 1.0:
        img = img / max(mx, 1e-8)
    return np.clip(img, 0.0, 1.0)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Luminance (Rec.601) in [0, 1], shape HxW."""
    img = as_float01(img)
    if img.ndim == 2:
        return img
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def to_hsv(img: np.ndarray) -> np.ndarray:
    """RGB->HSV with H in [0,1], S in [0,1], V in [0,1]."""
    img = as_float01(img)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if _HAS_CV2:
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] /= 179.0
        hsv[..., 1] /= 255.0
        hsv[..., 2] /= 255.0
        return hsv
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx = np.max(img, axis=-1)
    mn = np.min(img, axis=-1)
    diff = mx - mn
    h = np.zeros_like(mx)
    mask = diff > 1e-8
    # red is max
    idx = (mx == r) & mask
    h[idx] = (60 * ((g[idx] - b[idx]) / diff[idx]) + 360) % 360
    idx = (mx == g) & mask
    h[idx] = (60 * ((b[idx] - r[idx]) / diff[idx]) + 120) % 360
    idx = (mx == b) & mask
    h[idx] = (60 * ((r[idx] - g[idx]) / diff[idx]) + 240) % 360
    s = np.where(mx > 1e-8, diff / np.maximum(mx, 1e-8), 0.0)
    return np.stack([h / 360.0, s, mx], axis=-1)


def laplacian(gray: np.ndarray) -> np.ndarray:
    """4-neighbour Laplacian (edge-replicated)."""
    if _HAS_CV2:
        return cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    g = np.pad(gray.astype(np.float32), 1, mode="edge")
    lap = (
        g[:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, :-2]
        + g[1:-1, 2:]
        - 4.0 * g[1:-1, 1:-1]
    )
    return lap


def sobel_grad(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (gx, gy) Sobel gradients."""
    if _HAS_CV2:
        gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        return gx, gy
    g = np.pad(gray.astype(np.float32), 1, mode="edge")
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    gx = (
        kx[0, 0] * g[:-2, :-2] + kx[0, 2] * g[:-2, 2:]
        + kx[1, 0] * g[1:-1, :-2] + kx[1, 2] * g[1:-1, 2:]
        + kx[2, 0] * g[2:, :-2] + kx[2, 2] * g[2:, 2:]
    )
    gy = (
        ky[0, 0] * g[:-2, :-2] + ky[2, 0] * g[2:, :-2]
        + ky[0, 1] * g[:-2, 1:-1] + ky[2, 1] * g[2:, 1:-1]
        + ky[0, 2] * g[:-2, 2:] + ky[2, 2] * g[2:, 2:]
    )
    return gx, gy


def gaussian_blur(gray: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return gray.astype(np.float32)
    if _HAS_CV2:
        k = int(max(3, 2 * round(3 * sigma) + 1))
        return cv2.GaussianBlur(gray.astype(np.float32), (k, k), sigma)
    # separable Gaussian in numpy
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1)
    kern = np.exp(-(x**2) / (2 * sigma**2))
    kern /= kern.sum()
    g = gray.astype(np.float32)
    g = _conv1d(g, kern, axis=0)
    g = _conv1d(g, kern, axis=1)
    return g


def _conv1d(arr: np.ndarray, kern: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kern) // 2
    arr_p = np.pad(arr, [(pad, pad) if a == axis else (0, 0) for a in range(arr.ndim)], mode="edge")
    out = np.zeros_like(arr)
    for i, w in enumerate(kern):
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(i, i + arr.shape[axis])
        out += w * arr_p[tuple(sl)]
    return out


def saturating_norm(value: float, scale: float, power: float = 1.0) -> float:
    """Map a non-negative magnitude to [0, 1) via ``v / (v + scale)``.

    A robust, monotone alternative to min-max scaling that needs no dataset
    statistics, which matters for a streaming video where the score range is not
    known a priori.
    """
    v = max(0.0, float(value)) ** power
    s = max(1e-8, float(scale)) ** power
    return v / (v + s)


def resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize ``img`` to ``(H, W)`` with bilinear interpolation (NumPy fallback)."""
    out_h, out_w = size
    if _HAS_CV2:
        return cv2.resize(img.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    in_h, in_w = img.shape[:2]
    ys = (np.arange(out_h) + 0.5) * in_h / out_h - 0.5
    xs = (np.arange(out_w) + 0.5) * in_w / out_w - 0.5
    ys = np.clip(ys, 0, in_h - 1)
    xs = np.clip(xs, 0, in_w - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    if img.ndim == 3:
        wy = wy[..., None]
        wx = wx[..., None]
    Ia = img[np.ix_(y0, x0)]
    Ib = img[np.ix_(y0, x1)]
    Ic = img[np.ix_(y1, x0)]
    Id = img[np.ix_(y1, x1)]
    top = Ia * (1 - wx) + Ib * wx
    bot = Ic * (1 - wx) + Id * wx
    return top * (1 - wy) + bot * wy


def crop_roi(img: np.ndarray, box, pad: float = 0.0) -> np.ndarray:
    """Crop ``img`` (HxW or HxWxC) to a padded bbox (xyxy)."""
    h, w = img.shape[:2]
    bw, bh = box.x2 - box.x1, box.y2 - box.y1
    x1 = int(np.clip(box.x1 - pad * bw, 0, w - 1))
    y1 = int(np.clip(box.y1 - pad * bh, 0, h - 1))
    x2 = int(np.clip(box.x2 + pad * bw, x1 + 1, w))
    y2 = int(np.clip(box.y2 + pad * bh, y1 + 1, h))
    return img[y1:y2, x1:x2]
