"""Lazy torch adapter that turns a trained detector into a CandidateDetector.

Kept in a separate module so importing :mod:`polypai.frame_selection.candidate`
never imports torch.  The wrapped model is expected to return, per frame, a dict
with ``boxes`` (N,4 xyxy), ``scores`` (N,), and optionally ``masks`` (N,H,W).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from polypai.structures import BBox, PolypCandidate


class TorchDetectorAdapter:
    def __init__(self, model, preprocess: Callable | None, score_thresh: float, device: str):
        import torch  # noqa: F401  (validate availability eagerly, fail fast)

        self.model = model
        self.preprocess = preprocess
        self.score_thresh = score_thresh
        self.device = device
        self.model.eval()

    def detect(self, frame: np.ndarray) -> list[PolypCandidate]:
        import torch

        with torch.no_grad():
            if self.preprocess is not None:
                x = self.preprocess(frame)
            else:
                t = torch.from_numpy(np.ascontiguousarray(frame)).float()
                if t.ndim == 3:
                    t = t.permute(2, 0, 1)
                x = (t / 255.0).unsqueeze(0)
            x = x.to(self.device)
            out = self.model(x)

        out = out[0] if isinstance(out, (list, tuple)) else out
        boxes = _to_numpy(out["boxes"])
        scores = _to_numpy(out["scores"])
        masks = _to_numpy(out["masks"]) if "masks" in out else None
        embs = _to_numpy(out["embeddings"]) if "embeddings" in out else None

        cands: list[PolypCandidate] = []
        for i, (b, s) in enumerate(zip(boxes, scores)):
            if s < self.score_thresh:
                continue
            m = None
            if masks is not None:
                mi = masks[i]
                m = (mi[0] if mi.ndim == 3 else mi)
                m = (m > 0.5).astype(np.uint8)
            cands.append(
                PolypCandidate(
                    bbox=BBox(float(b[0]), float(b[1]), float(b[2]), float(b[3]), score=float(s)),
                    presence=float(s),
                    mask=m,
                    embedding=embs[i] if embs is not None else None,
                )
            )
        return cands


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)
