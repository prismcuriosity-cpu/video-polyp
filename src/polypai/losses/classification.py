"""Classification losses for Paris / NICE / pit-pattern heads (Modules 4-6)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCE(nn.Module):
    """Cross-entropy with label smoothing and optional class weights."""

    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor | None = None):
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits, target.long(), weight=self.weight, label_smoothing=self.smoothing
        )


class FocalCELoss(nn.Module):
    """Multi-class focal loss (Lin et al.) for imbalanced grades.

    NICE-3 / malignant lesions are rare; the focal term keeps gradient on these
    hard minority classes.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma, self.reduction = gamma, reduction
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, target.long(), weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
