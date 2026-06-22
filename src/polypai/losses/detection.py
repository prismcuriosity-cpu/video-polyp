"""Detection losses for the small-polyp detector (Module 2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VarifocalLoss(nn.Module):
    """IoU-aware Varifocal loss (Zhang et al., 2021).

    Positives are weighted by their target IoU (a soft quality score), negatives
    by ``alpha * p^gamma`` — focusing learning on accurately localised, hard
    examples, which matters for tiny low-contrast lesions.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, logits: torch.Tensor, target_score: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        pos = (target_score > 0).float()
        weight = target_score * pos + self.alpha * p.pow(self.gamma) * (1 - pos)
        loss = F.binary_cross_entropy_with_logits(logits, target_score, reduction="none") * weight
        if self.reduction == "mean":
            return loss.sum() / pos.sum().clamp_min(1.0)
        if self.reduction == "sum":
            return loss.sum()
        return loss


def bbox_ciou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Complete-IoU between xyxy boxes (per-row)."""
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)
    inter = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp_min(0) * (
        torch.min(py2, ty2) - torch.max(py1, ty1)
    ).clamp_min(0)
    pa = (px2 - px1).clamp_min(0) * (py2 - py1).clamp_min(0)
    ta = (tx2 - tx1).clamp_min(0) * (ty2 - ty1).clamp_min(0)
    union = pa + ta - inter + eps
    iou = inter / union
    cw = torch.max(px2, tx2) - torch.min(px1, tx1)
    ch = torch.max(py2, ty2) - torch.min(py1, ty1)
    c2 = cw**2 + ch**2 + eps
    rho2 = ((px1 + px2 - tx1 - tx2) ** 2 + (py1 + py2 - ty1 - ty2) ** 2) / 4
    import math

    pw, ph = (px2 - px1).clamp_min(eps), (py2 - py1).clamp_min(eps)
    tw, th = (tx2 - tx1).clamp_min(eps), (ty2 - ty1).clamp_min(eps)
    v = (4 / math.pi**2) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (rho2 / c2 + alpha * v)


class CIoULoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 1.0 - bbox_ciou(pred, target)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
