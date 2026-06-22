"""Unified multi-task model for the polyp pipeline.

A single shared encoder feeds a feature-pyramid neck and four task heads
(detection, segmentation, classification group, uncertainty/explainability),
trained jointly with the multi-task balancers in :mod:`polypai.losses.multitask`.

Importing this package requires PyTorch.  The configuration object
(:class:`ModelConfig`) and the model factory keep the architecture fully
declarative so it can be driven from YAML.
"""

from __future__ import annotations

from polypai.models.config import ModelConfig
from polypai.models.unified import UnifiedPolypModel, build_model

__all__ = ["ModelConfig", "UnifiedPolypModel", "build_model"]
