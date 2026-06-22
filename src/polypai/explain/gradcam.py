"""Grad-CAM and Grad-CAM++ for the unified model.

The target is any spatial module in the network (e.g. the last encoder stage or
an FPN level).  ``score_fn`` reduces the model output dict to the scalar being
explained — a classification logit, a malignancy probability, or the mean
segmentation logit inside an ROI — making the same machinery serve every head.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._act = None
        self._grad = None
        self._h1 = target_layer.register_forward_hook(self._save_act)
        self._h2 = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self._act = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self._grad = grad_out[0].detach()

    def _weights(self, grad: torch.Tensor) -> torch.Tensor:
        return grad.mean(dim=(2, 3), keepdim=True)

    def __call__(self, x: torch.Tensor, score_fn: Callable[[dict], torch.Tensor],
                 out_size: tuple[int, int] | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(x)
        score = score_fn(outputs)
        score.backward(retain_graph=False)
        weights = self._weights(self._grad)
        cam = F.relu((weights * self._act).sum(dim=1, keepdim=True))
        size = out_size or x.shape[-2:]
        cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.cpu().numpy()

    def remove(self):
        self._h1.remove()
        self._h2.remove()

    def __del__(self):  # best-effort hook cleanup
        try:
            self.remove()
        except Exception:
            pass


class GradCAMpp(GradCAM):
    """Grad-CAM++ weights (Chattopadhay et al., 2018) for sharper localisation."""

    def _weights(self, grad: torch.Tensor) -> torch.Tensor:
        g2 = grad.pow(2)
        g3 = grad.pow(3)
        act_sum = self._act.sum(dim=(2, 3), keepdim=True)
        alpha = g2 / (2 * g2 + act_sum * g3 + 1e-8)
        return (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)


def overlay_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] CAM over an RGB image, returning a uint8 RGB visualisation."""
    img = image.astype(np.float32)
    if img.max() > 1:
        img = img / 255.0
    cam = np.clip(cam, 0, 1)
    try:
        import cv2

        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    except Exception:  # simple red colormap fallback
        heat = np.zeros((*cam.shape, 3), np.float32)
        heat[..., 0] = cam
    out = (1 - alpha) * img + alpha * heat
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)
