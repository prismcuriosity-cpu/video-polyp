"""Confidence calibration: temperature scaling and Expected Calibration Error.

A clinically deployed classifier must report *trustworthy* probabilities; raw
softmax outputs are typically over-confident.  Temperature scaling (Guo et al.,
2017) is a single-parameter post-hoc fix that preserves accuracy while improving
calibration, measured here by ECE.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE over equal-width confidence bins."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


class TemperatureScaler(nn.Module):
    """Learns a single temperature T to divide logits by, minimising NLL."""

    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> float:
        return float(self.log_t.detach().exp())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_t.exp()

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> TemperatureScaler:
        opt = torch.optim.LBFGS([self.log_t], lr=0.05, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)
        return self
