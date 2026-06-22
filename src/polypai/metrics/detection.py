"""Detection metrics: AP / mAP (VOC all-points) and FROC.

Inputs use a simple per-image schema::

    preds: list over images of (boxes_xyxy [M,4], scores [M])
    gts:   list over images of boxes_xyxy [K,4]

which keeps the metrics framework-agnostic (the torch model's outputs are
converted to NumPy before evaluation).
"""

from __future__ import annotations

import numpy as np


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def _match(preds, gts, iou_thr: float):
    """Return (sorted scores, tp flags, fp flags, n_gt)."""
    all_scores, all_tp, all_fp, n_gt = [], [], [], 0
    for (boxes, scores), gt in zip(preds, gts):
        boxes, scores, gt = np.asarray(boxes), np.asarray(scores), np.asarray(gt)
        n_gt += len(gt)
        if len(boxes) == 0:
            continue
        order = np.argsort(-scores)
        boxes, scores = boxes[order], scores[order]
        iou = _iou_matrix(boxes, gt)
        matched = set()
        for i in range(len(boxes)):
            all_scores.append(scores[i])
            if iou.shape[1] == 0:
                all_tp.append(0); all_fp.append(1); continue
            j = int(np.argmax(iou[i]))
            if iou[i, j] >= iou_thr and j not in matched:
                matched.add(j); all_tp.append(1); all_fp.append(0)
            else:
                all_tp.append(0); all_fp.append(1)
    return np.asarray(all_scores), np.asarray(all_tp), np.asarray(all_fp), n_gt


def average_precision(preds, gts, iou_thr: float = 0.5) -> float:
    """VOC all-points AP at a single IoU threshold."""
    scores, tp, fp, n_gt = _match(preds, gts, iou_thr)
    if n_gt == 0:
        return 0.0
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp, fp = tp[order].cumsum(), fp[order].cumsum()
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    # all-points interpolation
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def mean_ap(preds, gts, iou_thrs=np.arange(0.5, 1.0, 0.05)) -> dict:
    """COCO-style mAP@[.5:.95] plus AP50/AP75."""
    aps = {float(t): average_precision(preds, gts, float(t)) for t in iou_thrs}
    return {
        "mAP": float(np.mean(list(aps.values()))),
        "AP50": aps.get(0.5, average_precision(preds, gts, 0.5)),
        "AP75": aps.get(0.75, average_precision(preds, gts, 0.75)),
        "per_iou": aps,
    }


def froc(preds, gts, iou_thr: float = 0.3, fppi_points=(0.125, 0.25, 0.5, 1, 2, 4, 8)) -> dict:
    """Free-response ROC: sensitivity at fixed false-positives-per-image.

    Lower IoU (0.3) is conventional for lesion *detection* (vs localisation).
    """
    scores, tp, fp, n_gt = _match(preds, gts, iou_thr)
    n_img = max(len(gts), 1)
    if len(scores) == 0 or n_gt == 0:
        return {"fppi": list(fppi_points), "sensitivity": [0.0] * len(fppi_points), "mean": 0.0}
    order = np.argsort(-scores)
    tp_c, fp_c = tp[order].cumsum(), fp[order].cumsum()
    fppi = fp_c / n_img
    sens = tp_c / n_gt
    out = []
    for f in fppi_points:
        idx = np.where(fppi <= f)[0]
        out.append(float(sens[idx[-1]]) if len(idx) else 0.0)
    return {"fppi": list(fppi_points), "sensitivity": out, "mean": float(np.mean(out))}
