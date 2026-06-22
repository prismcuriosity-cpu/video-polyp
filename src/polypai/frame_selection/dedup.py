"""Near-duplicate frame removal (brief §10).

Two complementary signals:

* **Perceptual hashing (pHash)** — a 64-bit DCT hash gives a cheap, rotation/
  noise-robust fingerprint; frames within a small Hamming radius are duplicates.
* **Embedding similarity** — when an appearance embedding is available (from the
  shared encoder), cosine similarity catches duplicates that pHash misses
  (e.g. small viewpoint changes with identical structure).

Both are exposed as greedy "keep the highest-scoring representative" filters.
"""

from __future__ import annotations

import numpy as np

from polypai.frame_selection.imageops import resize, to_gray
from polypai.structures import FrameScore

_DCT_CACHE: dict[int, np.ndarray] = {}


def _dct_matrix(n: int) -> np.ndarray:
    if n not in _DCT_CACHE:
        k = np.arange(n)
        m = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
        m[0, :] = 1.0 / np.sqrt(2)
        _DCT_CACHE[n] = m * np.sqrt(2.0 / n)
    return _DCT_CACHE[n]


def phash(image: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """64-bit (for ``hash_size=8``) DCT perceptual hash."""
    n = hash_size * highfreq_factor
    gray = to_gray(image)
    small = resize(gray, (n, n))
    d = _dct_matrix(n)
    dct = d @ small @ d.T
    block = dct[:hash_size, :hash_size]
    med = np.median(block[1:].flatten() if block.size > 1 else block.flatten())
    bits = (block > med).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def dedup_by_phash(
    frame_scores: list[FrameScore],
    images: dict[int, np.ndarray] | None = None,
    max_hamming: int = 8,
) -> list[FrameScore]:
    """Keep highest-scoring representative per perceptual-hash cluster.

    Hashes already stored on ``FrameScore.phash`` are reused; otherwise computed
    from ``images[frame_index]``.
    """
    ordered = sorted(frame_scores, key=lambda fs: -fs.total)
    kept: list[FrameScore] = []
    kept_hashes: list[int] = []
    for fs in ordered:
        h = fs.phash
        if h is None and images is not None and fs.frame_index in images:
            h = phash(images[fs.frame_index])
            fs.phash = h
        if h is None:
            kept.append(fs)
            continue
        if all(hamming(h, kh) > max_hamming for kh in kept_hashes):
            kept.append(fs)
            kept_hashes.append(h)
    return kept


def dedup_by_embedding(
    frame_scores: list[FrameScore],
    sim_thresh: float = 0.92,
) -> list[FrameScore]:
    """Greedy cosine-similarity dedup using ``candidate.embedding`` vectors."""
    ordered = sorted(frame_scores, key=lambda fs: -fs.total)
    kept: list[FrameScore] = []
    kept_emb: list[np.ndarray] = []
    for fs in ordered:
        emb = fs.candidate.embedding if fs.candidate is not None else None
        if emb is None:
            kept.append(fs)
            continue
        emb = np.asarray(emb, dtype=np.float64)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        if all(float(emb @ ke) < sim_thresh for ke in kept_emb):
            kept.append(fs)
            kept_emb.append(emb)
    return kept
