"""Clinically meaningful endpoints beyond pixel/box metrics.

These are the numbers a gastroenterologist cares about: does the system raise
the Adenoma Detection Rate, how many polyps are missed, and how accurately is
malignancy predicted.
"""

from __future__ import annotations

import numpy as np


def adenoma_detection_rate(procedure_has_adenoma: list[bool]) -> float:
    """Fraction of procedures in which >= 1 adenoma was detected (per-patient).

    ADR is a per-colonoscopy quality benchmark (target >= 25%).  ``procedure_has_
    adenoma[i]`` is True iff the system flagged at least one histologically-
    confirmed adenoma in procedure *i*.
    """
    if not procedure_has_adenoma:
        return 0.0
    return float(np.mean([1.0 if x else 0.0 for x in procedure_has_adenoma]))


def polyp_miss_rate(n_ground_truth_polyps: int, n_detected_unique_polyps: int) -> float:
    """Per-polyp miss rate = 1 - (unique detected / ground-truth polyps).

    Counts *unique lesions* tracked across the video, not per-frame detections,
    so transient detections of the same polyp are not double-counted.
    """
    if n_ground_truth_polyps <= 0:
        return 0.0
    missed = max(0, n_ground_truth_polyps - n_detected_unique_polyps)
    return float(missed / n_ground_truth_polyps)


def malignancy_accuracy(y_true_malignant, y_pred_malignant) -> dict:
    """Accuracy / sensitivity / specificity for the malignant-vs-benign decision.

    Sensitivity is the safety-critical number here (a missed malignancy is the
    costly error), so it is reported alongside accuracy.
    """
    from polypai.metrics.classification import accuracy, sensitivity_specificity

    y_true = np.asarray(y_true_malignant).astype(int)
    y_pred = np.asarray(y_pred_malignant).astype(int)
    sens, spec = sensitivity_specificity(y_true, y_pred, positive_class=1)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "sensitivity": sens,
        "specificity": spec,
    }
