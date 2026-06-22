"""Classification metrics: accuracy, AUC, F1, sensitivity, specificity.

AUC is computed via the Mann-Whitney U statistic (rank method) so there is no
scikit-learn dependency; multi-class AUC uses one-vs-rest macro averaging.
"""

from __future__ import annotations

import numpy as np


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        cm[int(t), int(p)] += 1
    return cm


def _binary_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auc_score(y_true, y_score, num_classes: int | None = None) -> float:
    """Binary or macro one-vs-rest multi-class ROC AUC."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 1):
        return _binary_auc(y_true, y_score.ravel())
    if y_score.ndim == 2 and y_score.shape[1] == 2 and (num_classes in (None, 2)):
        return _binary_auc(y_true, y_score[:, 1])
    k = num_classes or y_score.shape[1]
    aucs = []
    for c in range(k):
        a = _binary_auc((y_true == c).astype(int), y_score[:, c])
        if not np.isnan(a):
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else float("nan")


def f1_score(y_true, y_pred, num_classes: int, average: str = "macro") -> float:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    f1s = np.asarray(f1s)
    if average == "macro":
        return float(f1s.mean())
    weights = cm.sum(1)
    return float((f1s * weights).sum() / max(weights.sum(), 1))


def sensitivity_specificity(y_true, y_pred, positive_class: int = 1) -> tuple[float, float]:
    """Binary (or one-vs-rest) sensitivity (recall) and specificity."""
    y_true = (np.asarray(y_true) == positive_class).astype(int)
    y_pred = (np.asarray(y_pred) == positive_class).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return float(sens), float(spec)


def classification_report(y_true, y_pred, y_score=None, num_classes: int | None = None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    k = num_classes or int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    rep = {
        "accuracy": accuracy(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, k, "macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, k).tolist(),
    }
    sens, spec = sensitivity_specificity(y_true, y_pred, positive_class=k - 1)
    rep["sensitivity_top_class"] = sens
    rep["specificity_top_class"] = spec
    if y_score is not None:
        rep["auc"] = auc_score(y_true, y_score, k)
    return rep
