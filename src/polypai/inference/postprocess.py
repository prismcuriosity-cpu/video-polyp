"""Decode the anchor-free detection head into boxes and run NMS."""

from __future__ import annotations

import torch

from polypai.engine.fcos_target import DEFAULT_STRIDES, decode_boxes, locations_for_level


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.5) -> torch.Tensor:
    """Pure-torch greedy NMS (no torchvision dependency). Returns kept indices."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.max(x1[i], x1[rest])
        yy1 = torch.max(y1[i], y1[rest])
        xx2 = torch.min(x2[i], x2[rest])
        yy2 = torch.min(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thresh]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


@torch.no_grad()
def decode_detections(det_out: dict, strides=DEFAULT_STRIDES, score_thresh: float = 0.3,
                      nms_iou: float = 0.5, max_det: int = 100):
    """Decode per-image detections -> list of dicts {boxes (N,4), scores (N,)}.

    Detection confidence = sqrt(sigmoid(cls) * sigmoid(centerness)), the standard
    FCOS test-time score that down-weights off-centre, low-quality locations.
    """
    levels = det_out["levels"]
    B = det_out["cls"][0].shape[0]
    device = det_out["cls"][0].device
    results = []
    for b in range(B):
        all_boxes, all_scores = [], []
        for i, _lv in enumerate(levels):
            s = strides[i]
            cls = det_out["cls"][i][b]            # (C,H,W)
            ctr = det_out["centerness"][i][b]     # (1,H,W)
            reg = det_out["reg"][i][b]            # (4,H,W)
            h, w = cls.shape[-2:]
            locs = locations_for_level(h, w, s, device)
            score = (cls.sigmoid().amax(0) * ctr.sigmoid().squeeze(0)).reshape(-1).sqrt()
            reg_pix = (reg.permute(1, 2, 0).reshape(-1, 4)) * s
            boxes = decode_boxes(locs, reg_pix)
            keep = score >= score_thresh
            all_boxes.append(boxes[keep])
            all_scores.append(score[keep])
        boxes = torch.cat(all_boxes, 0) if all_boxes else torch.zeros((0, 4), device=device)
        scores = torch.cat(all_scores, 0) if all_scores else torch.zeros((0,), device=device)
        if boxes.numel():
            k = nms(boxes, scores, nms_iou)[:max_det]
            boxes, scores = boxes[k], scores[k]
        results.append({"boxes": boxes.cpu(), "scores": scores.cpu()})
    return results
