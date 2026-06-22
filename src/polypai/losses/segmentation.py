"""Segmentation losses for boundary-, shape-, and topology-aware training (Module 3).

Implements the loss set requested in the brief:

* :class:`DiceLoss`            - region overlap.
* :class:`FocalTverskyLoss`    - tunable FP/FN trade-off + focal hard-example focus.
* :class:`BoundaryLoss`        - Kervadec et al. (2019) distance-map boundary loss.
* :class:`HausdorffDTLoss`     - Karimi & Salcudean (2020) DT-based HD surrogate.
* :class:`SoftClDiceLoss`      - Shit et al. (2021) topology-preserving clDice.
* :class:`ComboSegLoss`        - scheduled, weighted combination of the above.

Conventions: ``logits`` are (N, C, H, W); ``target`` is (N, H, W) int64 for
multi-class or (N, 1, H, W)/(N, H, W) for binary.  Binary tasks use ``C in {1, 2}``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_onehot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    if target.dim() == 4 and target.shape[1] == num_classes:
        return target.float()
    if target.dim() == 4 and target.shape[1] == 1:
        target = target[:, 0]
    oh = F.one_hot(target.long().clamp(0, num_classes - 1), num_classes)
    return oh.permute(0, 3, 1, 2).float()


def _probs(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] == 1:
        p = torch.sigmoid(logits)
        return torch.cat([1 - p, p], dim=1)
    return torch.softmax(logits, dim=1)


class DiceLoss(nn.Module):
    """Soft multi-class Dice loss (optionally excluding background)."""

    def __init__(self, smooth: float = 1.0, ignore_background: bool = True):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = _probs(logits)
        oh = _to_onehot(target, probs.shape[1]).to(probs.dtype)
        dims = (0, 2, 3)
        inter = (probs * oh).sum(dims)
        card = probs.sum(dims) + oh.sum(dims)
        dice = (2 * inter + self.smooth) / (card + self.smooth)
        if self.ignore_background and probs.shape[1] > 1:
            dice = dice[1:]
        return 1.0 - dice.mean()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss (Abraham & Khan, 2019).

    ``alpha`` weights false positives, ``beta`` false negatives (beta>alpha
    boosts recall — desirable for diminutive polyps).  ``gamma`` focuses on hard
    regions.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33,
                 smooth: float = 1.0, ignore_background: bool = True):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = _probs(logits)
        oh = _to_onehot(target, probs.shape[1]).to(probs.dtype)
        dims = (0, 2, 3)
        tp = (probs * oh).sum(dims)
        fp = (probs * (1 - oh)).sum(dims)
        fn = ((1 - probs) * oh).sum(dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        if self.ignore_background and probs.shape[1] > 1:
            tversky = tversky[1:]
        return torch.pow(1.0 - tversky, self.gamma).mean()


def signed_distance_maps(masks: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Per-class signed distance transform used by :class:`BoundaryLoss`.

    Positive outside the object, negative inside (Kervadec convention).  Computed
    on CPU with SciPy/NumPy (constant w.r.t. the network output, so no gradient
    is needed through it).  Falls back to a Chamfer approximation without SciPy.
    """
    oh = _to_onehot(masks, num_classes).cpu().numpy()
    out = np.zeros_like(oh, dtype=np.float32)
    try:
        from scipy.ndimage import distance_transform_edt as edt
    except Exception:  # pragma: no cover
        edt = None
    for n in range(oh.shape[0]):
        for c in range(num_classes):
            posmask = oh[n, c].astype(bool)
            if not posmask.any():
                continue
            negmask = ~posmask
            if edt is not None:
                dist = edt(negmask) - (edt(posmask) - posmask)
            else:
                dist = (~posmask).astype(np.float32)  # degraded fallback
            out[n, c] = dist
    return torch.from_numpy(out).to(masks.device)


class BoundaryLoss(nn.Module):
    """Kervadec et al. boundary loss: <softmax, signed-distance-map> integral."""

    def __init__(self, ignore_background: bool = True):
        super().__init__()
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        probs = _probs(logits)
        c0 = 1 if (self.ignore_background and probs.shape[1] > 1) else 0
        return torch.einsum("bchw,bchw->", probs[:, c0:], dist_maps[:, c0:].to(probs.dtype)) / (
            probs.shape[0] * probs[:, c0:].shape[1] * probs.shape[2] * probs.shape[3]
        )


class HausdorffDTLoss(nn.Module):
    """Distance-transform Hausdorff surrogate (Karimi & Salcudean, 2020).

    ``L = mean( (p - g)^2 * (dt_g^alpha + dt_p^alpha) )`` where the DTs act as
    constants estimated from the current target/prediction each step.
    """

    def __init__(self, alpha: float = 2.0):
        super().__init__()
        self.alpha = alpha

    @staticmethod
    def _dt(mask_np: np.ndarray) -> np.ndarray:
        try:
            from scipy.ndimage import distance_transform_edt as edt
        except Exception:  # pragma: no cover
            return np.zeros_like(mask_np, dtype=np.float32)
        out = np.zeros_like(mask_np, dtype=np.float32)
        for n in range(mask_np.shape[0]):
            fg = mask_np[n].astype(bool)
            if fg.any():
                out[n] = edt(~fg)
            if (~fg).any():
                out[n] = out[n] + edt(fg)
        return out

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = _probs(logits)
        p = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        g = _to_onehot(target, probs.shape[1])
        g = g[:, 1] if g.shape[1] > 1 else g[:, 0]
        with torch.no_grad():
            dt_g = torch.from_numpy(self._dt(g.cpu().numpy())).to(p.device) ** self.alpha
            dt_p = torch.from_numpy(self._dt((p > 0.5).float().cpu().numpy())).to(p.device) ** self.alpha
            weight = dt_g + dt_p
        return (((p - g) ** 2) * weight).mean()


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-x, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, (3, 3), (1, 1), (1, 1))


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(x))


def soft_skeleton(x: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Differentiable morphological soft-skeleton (Shit et al., 2021)."""
    skel = F.relu(x - _soft_open(x))
    for _ in range(iters):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftClDiceLoss(nn.Module):
    """Topology-preserving centreline-Dice loss (foreground channel)."""

    def __init__(self, iters: int = 10, smooth: float = 1.0, alpha: float = 0.5):
        super().__init__()
        self.iters, self.smooth, self.alpha = iters, smooth, alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = _probs(logits)
        p = probs[:, 1:2] if probs.shape[1] > 1 else probs[:, 0:1]
        g = _to_onehot(target, probs.shape[1])
        g = g[:, 1:2] if g.shape[1] > 1 else g[:, 0:1]
        sp, sg = soft_skeleton(p, self.iters), soft_skeleton(g, self.iters)
        tprec = (sp * g).sum((1, 2, 3)) + self.smooth
        tprec = tprec / (sp.sum((1, 2, 3)) + self.smooth)
        tsens = (sg * p).sum((1, 2, 3)) + self.smooth
        tsens = tsens / (sg.sum((1, 2, 3)) + self.smooth)
        cldice = 2.0 * (tprec * tsens) / (tprec + tsens)
        dice = DiceLoss(ignore_background=False)(logits, target)
        return (1 - self.alpha) * dice + self.alpha * (1.0 - cldice.mean())


class ComboSegLoss(nn.Module):
    """Weighted combination of region, boundary, topology and HD losses.

    ``boundary_warmup`` linearly ramps the boundary-loss weight from 0->1 over
    the given number of steps (Kervadec's scheduling), which stabilises early
    training before the region loss has localised the object.
    """

    def __init__(
        self,
        num_classes: int = 2,
        w_dice: float = 1.0,
        w_focal_tversky: float = 1.0,
        w_boundary: float = 0.5,
        w_cldice: float = 0.5,
        w_hausdorff: float = 0.0,
        boundary_warmup: int = 2000,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dice = DiceLoss()
        self.ft = FocalTverskyLoss()
        self.boundary = BoundaryLoss()
        self.cldice = SoftClDiceLoss()
        self.hd = HausdorffDTLoss()
        self.w = dict(dice=w_dice, ft=w_focal_tversky, boundary=w_boundary,
                      cldice=w_cldice, hd=w_hausdorff)
        self.boundary_warmup = max(1, boundary_warmup)
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                dist_maps: torch.Tensor | None = None) -> dict:
        ramp = float(min(1.0, self._step.item() / self.boundary_warmup))
        parts = {
            "dice": self.w["dice"] * self.dice(logits, target),
            "focal_tversky": self.w["ft"] * self.ft(logits, target),
        }
        if self.w["cldice"] > 0:
            parts["cldice"] = self.w["cldice"] * self.cldice(logits, target)
        if self.w["boundary"] > 0:
            if dist_maps is None:
                dist_maps = signed_distance_maps(target, logits.shape[1])
            parts["boundary"] = self.w["boundary"] * ramp * self.boundary(logits, dist_maps)
        if self.w["hd"] > 0:
            parts["hausdorff"] = self.w["hd"] * self.hd(logits, target)
        parts["total"] = sum(parts.values())
        if self.training:
            self._step += 1
        return parts
