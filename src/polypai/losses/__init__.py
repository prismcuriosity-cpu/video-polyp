"""Loss functions for the unified polyp model.

Segmentation (Module 3): Dice, Focal-Tversky, Boundary, Hausdorff-DT, soft-clDice
topology loss, plus a configurable combo loss.
Detection (Module 2): Quality/Varifocal focal loss + CIoU.
Classification (Modules 5/6): label-smoothed / focal CE.
Multi-task: homoscedastic uncertainty weighting and GradNorm-style balancing.

All losses import torch lazily-friendly (module import requires torch, but the
package ``polypai`` does not import this submodule eagerly).
"""

from __future__ import annotations

from polypai.losses.classification import FocalCELoss, LabelSmoothingCE
from polypai.losses.detection import CIoULoss, VarifocalLoss
from polypai.losses.multitask import GradNormWeighter, UncertaintyWeighting
from polypai.losses.segmentation import (
    BoundaryLoss,
    ComboSegLoss,
    DiceLoss,
    FocalTverskyLoss,
    HausdorffDTLoss,
    SoftClDiceLoss,
    signed_distance_maps,
)

__all__ = [
    "DiceLoss",
    "FocalTverskyLoss",
    "BoundaryLoss",
    "HausdorffDTLoss",
    "SoftClDiceLoss",
    "ComboSegLoss",
    "signed_distance_maps",
    "CIoULoss",
    "VarifocalLoss",
    "LabelSmoothingCE",
    "FocalCELoss",
    "UncertaintyWeighting",
    "GradNormWeighter",
]
