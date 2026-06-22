"""Tests for the evaluation-protocol metrics (pure NumPy/SciPy)."""

import numpy as np

from polypai.metrics import (
    accuracy,
    adenoma_detection_rate,
    assd,
    auc_score,
    average_precision,
    dice_score,
    f1_score,
    froc,
    hd95,
    iou_score,
    malignancy_accuracy,
    mean_ap,
    polyp_miss_rate,
    sensitivity_specificity,
)


def _square_mask(h, w, x0, y0, s):
    m = np.zeros((h, w), np.uint8)
    m[y0 : y0 + s, x0 : x0 + s] = 1
    return m


def test_dice_iou_perfect_and_disjoint():
    a = _square_mask(40, 40, 5, 5, 10)
    b = _square_mask(40, 40, 25, 25, 10)
    assert dice_score(a, a) > 0.999
    assert iou_score(a, a) > 0.999
    assert dice_score(a, b) < 0.01


def test_hd95_assd_zero_for_identical():
    a = _square_mask(40, 40, 10, 10, 12)
    assert hd95(a, a) == 0.0
    assert assd(a, a) == 0.0


def test_hd95_positive_for_shift():
    a = _square_mask(60, 60, 10, 10, 20)
    b = _square_mask(60, 60, 18, 10, 20)
    assert hd95(a, b) > 0.0
    assert assd(a, b) > 0.0


def test_average_precision_perfect():
    gts = [np.array([[10, 10, 30, 30]]), np.array([[5, 5, 15, 15]])]
    preds = [
        (np.array([[10, 10, 30, 30]]), np.array([0.9])),
        (np.array([[5, 5, 15, 15]]), np.array([0.95])),
    ]
    assert average_precision(preds, gts, 0.5) > 0.99


def test_average_precision_drops_with_false_positive():
    gts = [np.array([[10, 10, 30, 30]])]
    good = [(np.array([[10, 10, 30, 30]]), np.array([0.9]))]
    noisy = [(np.array([[10, 10, 30, 30], [50, 50, 60, 60]]), np.array([0.9, 0.95]))]
    assert average_precision(noisy, gts, 0.5) < average_precision(good, gts, 0.5)


def test_mean_ap_and_froc_keys():
    gts = [np.array([[10, 10, 30, 30]])]
    preds = [(np.array([[11, 11, 31, 31]]), np.array([0.8]))]
    m = mean_ap(preds, gts)
    assert {"mAP", "AP50", "AP75"} <= set(m)
    fr = froc(preds, gts)
    assert "sensitivity" in fr and len(fr["sensitivity"]) == len(fr["fppi"])


def test_classification_basic():
    y_true = [0, 1, 2, 2, 1, 0]
    y_pred = [0, 1, 2, 1, 1, 0]
    assert 0.0 <= accuracy(y_true, y_pred) <= 1.0
    assert 0.0 <= f1_score(y_true, y_pred, 3) <= 1.0


def test_auc_perfect_ranking():
    y_true = [0, 0, 1, 1]
    score = [0.1, 0.2, 0.8, 0.9]
    assert abs(auc_score(y_true, score) - 1.0) < 1e-6


def test_auc_multiclass():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    scores = np.eye(3)[y_true] * 0.6 + 0.2  # confident-correct-ish
    a = auc_score(y_true, scores, num_classes=3)
    assert 0.5 <= a <= 1.0


def test_sensitivity_specificity():
    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 0, 1]
    sens, spec = sensitivity_specificity(y_true, y_pred, positive_class=1)
    assert abs(sens - 2 / 3) < 1e-6
    assert abs(spec - 1.0) < 1e-6


def test_clinical_metrics():
    assert adenoma_detection_rate([True, False, True, False]) == 0.5
    assert polyp_miss_rate(10, 7) == 0.3
    res = malignancy_accuracy([1, 0, 1, 0], [1, 0, 0, 0])
    assert set(res) == {"accuracy", "sensitivity", "specificity"}
    assert abs(res["sensitivity"] - 0.5) < 1e-6
