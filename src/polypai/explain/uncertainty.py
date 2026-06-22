"""Uncertainty quantification helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def predictive_entropy(logits: torch.Tensor) -> np.ndarray:
    """Per-pixel (or per-sample) Shannon entropy of the softmax distribution.

    High entropy marks ambiguous boundaries / lesions the model is unsure about —
    exactly the regions to surface to the endoscopist.
    """
    p = F.softmax(logits, dim=1)
    ent = -(p * (p.clamp_min(1e-8)).log()).sum(dim=1)
    return ent.detach().cpu().numpy()


def segmentation_uncertainty(model_output: dict) -> np.ndarray:
    """Aleatoric uncertainty map from the model's predicted log-variance head,
    falling back to predictive entropy when the uncertainty head is absent."""
    if "uncertainty" in model_output:
        logvar = model_output["uncertainty"]
        seg = model_output["segmentation"]["seg_logits"]
        logvar = F.interpolate(logvar, size=seg.shape[-2:], mode="bilinear", align_corners=False)
        return torch.exp(logvar).squeeze(1).detach().cpu().numpy()
    return predictive_entropy(model_output["segmentation"]["seg_logits"])


def mc_dropout_passes(model, x: torch.Tensor, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Epistemic uncertainty via MC-Dropout: returns (mean_prob, std_prob).

    Enables dropout at inference and averages ``n`` stochastic forward passes.
    """
    was_training = model.training
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
            m.train()
    probs = []
    with torch.no_grad():
        for _ in range(n):
            out = model(x)
            probs.append(F.softmax(out["segmentation"]["seg_logits"], dim=1).cpu().numpy())
    model.train(was_training)
    probs = np.stack(probs)
    return probs.mean(0), probs.std(0)
