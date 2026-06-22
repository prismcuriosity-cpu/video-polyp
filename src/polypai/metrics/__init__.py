"""Evaluation metrics covering the full clinical protocol.

Segmentation: Dice, IoU, HD95, ASSD.
Detection:    precision/recall, AP/mAP, FROC.
Classification: accuracy, AUC, F1, sensitivity, specificity.
Clinical:     Adenoma Detection Rate, per-polyp miss rate, malignancy accuracy.

Pure NumPy (+ optional SciPy for surface distances); no torch dependency, so the
evaluator can run on a laptop or in CI.
"""

from __future__ import annotations

from polypai.metrics.classification import (
    accuracy,
    auc_score,
    classification_report,
    f1_score,
    sensitivity_specificity,
)
from polypai.metrics.clinical import (
    adenoma_detection_rate,
    malignancy_accuracy,
    polyp_miss_rate,
)
from polypai.metrics.detection import average_precision, froc, mean_ap
from polypai.metrics.segmentation import assd, dice_score, hd95, iou_score

__all__ = [
    "dice_score",
    "iou_score",
    "hd95",
    "assd",
    "average_precision",
    "mean_ap",
    "froc",
    "accuracy",
    "auc_score",
    "f1_score",
    "sensitivity_specificity",
    "classification_report",
    "adenoma_detection_rate",
    "polyp_miss_rate",
    "malignancy_accuracy",
]
