"""Unified multi-task criterion.

Combines: FCOS detection loss (Varifocal cls + CIoU box + centerness BCE),
boundary-aware segmentation combo loss, heteroscedastic uncertainty NLL, and the
characterisation classification losses (with per-sample label masking so mixed
datasets work).  Task groups are balanced with homoscedastic uncertainty
weighting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from polypai.engine.fcos_target import DEFAULT_STRIDES, assign_image, decode_boxes
from polypai.losses.detection import CIoULoss, VarifocalLoss
from polypai.losses.multitask import UncertaintyWeighting
from polypai.losses.segmentation import ComboSegLoss

IGNORE_INDEX = -100


class UnifiedCriterion(nn.Module):
    def __init__(self, num_seg_classes: int = 2, cls_tasks=("paris", "nice", "kudo", "malignancy"),
                 strides=DEFAULT_STRIDES, balance: bool = True, w_boundary_aux: float = 0.5):
        super().__init__()
        self.strides = strides
        self.seg_loss = ComboSegLoss(num_classes=num_seg_classes)
        self.vfl = VarifocalLoss()
        self.ciou = CIoULoss(reduction="none")
        self.cls_tasks = list(cls_tasks)
        self.w_boundary_aux = w_boundary_aux
        self.balancer = UncertaintyWeighting(["detection", "segmentation", "classification"]) if balance else None

    # -- detection ----------------------------------------------------------
    def detection_loss(self, det_out: dict, boxes_per_image: list[torch.Tensor], device) -> torch.Tensor:
        levels = det_out["levels"]
        B = det_out["cls"][0].shape[0]
        level_shapes = [tuple(det_out["cls"][i].shape[-2:]) for i in range(len(levels))]

        # flatten predictions over levels -> (B, M, .)
        def flat(maps, ch):
            return torch.cat([m.permute(0, 2, 3, 1).reshape(B, -1, ch) for m in maps], dim=1)

        cls_pred = flat(det_out["cls"], det_out["cls"][0].shape[1])  # (B,M,1)
        ctr_pred = flat(det_out["centerness"], 1)                    # (B,M,1)
        reg_pred = flat(det_out["reg"], 4)                           # (B,M,4) feature units
        stride_vec = torch.cat([
            cls_pred.new_full((h * w,), s) for (h, w), s in zip(level_shapes, self.strides)
        ])  # (M,)

        total_cls = total_reg = total_ctr = 0.0
        num_pos = 0
        for b in range(B):
            tgt = assign_image(boxes_per_image[b].to(device), level_shapes, self.strides)
            pos = tgt["pos_mask"]
            cls_target = (tgt["centerness"] * pos.float()).unsqueeze(-1)  # IoU-aware soft label
            total_cls = total_cls + self.vfl(cls_pred[b], cls_target)
            n = int(pos.sum())
            num_pos += n
            if n > 0:
                locs = tgt["locations"]
                reg_pix = reg_pred[b] * stride_vec.unsqueeze(-1)
                pred_box = decode_boxes(locs, reg_pix)[pos]
                gt_box = decode_boxes(locs, tgt["reg"])[pos]
                ctr_t = tgt["centerness"][pos]
                reg_l = (self.ciou(pred_box, gt_box) * ctr_t).sum()
                total_reg = total_reg + reg_l
                total_ctr = total_ctr + F.binary_cross_entropy_with_logits(
                    ctr_pred[b].squeeze(-1)[pos], ctr_t, reduction="sum"
                )
        denom = max(num_pos, 1)
        return total_cls / B + total_reg / denom + total_ctr / denom

    # -- segmentation -------------------------------------------------------
    def segmentation_loss(self, seg_out: dict, mask: torch.Tensor, unc=None) -> tuple[torch.Tensor, dict]:
        parts = self.seg_loss(seg_out["seg_logits"], mask)
        loss = parts["total"]
        # auxiliary boundary supervision
        if "boundary_logits" in seg_out:
            edge = _mask_edges(mask).unsqueeze(1).float()
            loss = loss + self.w_boundary_aux * F.binary_cross_entropy_with_logits(
                seg_out["boundary_logits"], edge
            )
        # heteroscedastic uncertainty NLL (attenuates loss where logvar is high)
        if unc is not None:
            ce = F.cross_entropy(seg_out["seg_logits"], mask, reduction="none").unsqueeze(1)
            logvar = F.interpolate(unc, size=ce.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss + (0.5 * torch.exp(-logvar) * ce + 0.5 * logvar).mean()
        return loss, parts

    # -- classification -----------------------------------------------------
    def classification_loss(self, cls_out: dict, labels: dict, device) -> torch.Tensor:
        total = torch.zeros((), device=device)
        count = 0
        for task in self.cls_tasks:
            if task not in cls_out or task not in labels:
                continue
            y = labels[task].to(device)
            if (y != IGNORE_INDEX).any():
                total = total + F.cross_entropy(cls_out[task], y, ignore_index=IGNORE_INDEX)
                count += 1
        return total / max(count, 1)

    # -- combine ------------------------------------------------------------
    def forward(self, outputs: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        device = outputs["segmentation"]["seg_logits"].device if "segmentation" in outputs else \
            outputs["detection"]["cls"][0].device
        losses: dict[str, torch.Tensor] = {}

        if "detection" in outputs:
            losses["detection"] = self.detection_loss(outputs["detection"], batch["boxes"], device)
        if "segmentation" in outputs:
            seg_l, parts = self.segmentation_loss(
                outputs["segmentation"], batch["mask"].to(device), outputs.get("uncertainty")
            )
            losses["segmentation"] = seg_l
        if "classification" in outputs:
            losses["classification"] = self.classification_loss(
                outputs["classification"], batch["labels"], device
            )

        if self.balancer is not None and len(losses) > 1:
            total, weights = self.balancer(losses)
        else:
            total, weights = sum(losses.values()), {k: 1.0 for k in losses}

        log = {k: float(v.detach()) for k, v in losses.items()}
        log["total"] = float(total.detach())
        log["weights"] = weights
        return total, log


def _mask_edges(mask: torch.Tensor) -> torch.Tensor:
    """Binary boundary map of an integer mask (B,H,W) via 1-px morphological gradient."""
    m = (mask > 0).float().unsqueeze(1)
    maxp = F.max_pool2d(m, 3, 1, 1)
    minp = -F.max_pool2d(-m, 3, 1, 1)
    return (maxp - minp).squeeze(1)
