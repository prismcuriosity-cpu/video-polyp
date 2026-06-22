"""Explainable-AI utilities for clinical deployment (Module 9).

Grad-CAM / Grad-CAM++ decision heatmaps, segmentation uncertainty and predictive
entropy, and confidence calibration (temperature scaling + ECE).  These power the
"why was this lesion called malignant / which pit pattern drove the prediction"
narrative in the clinical report.
"""

from __future__ import annotations

from polypai.explain.calibration import expected_calibration_error, TemperatureScaler
from polypai.explain.gradcam import GradCAM, GradCAMpp, overlay_heatmap
from polypai.explain.uncertainty import predictive_entropy, segmentation_uncertainty

__all__ = [
    "GradCAM",
    "GradCAMpp",
    "overlay_heatmap",
    "predictive_entropy",
    "segmentation_uncertainty",
    "expected_calibration_error",
    "TemperatureScaler",
]
