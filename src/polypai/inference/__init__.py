"""Inference: detection post-processing and the end-to-end clinical pipeline."""

from __future__ import annotations

from polypai.inference.pipeline import ClinicalPipeline, PipelineConfig
from polypai.inference.postprocess import decode_detections, nms
from polypai.inference.report import LesionFinding, ProcedureReport

__all__ = [
    "ClinicalPipeline",
    "PipelineConfig",
    "decode_detections",
    "nms",
    "LesionFinding",
    "ProcedureReport",
]
