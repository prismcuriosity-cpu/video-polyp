"""Training / evaluation engine: FCOS target assignment, the unified multi-task
criterion, the AMP trainer, and the evaluator."""

from __future__ import annotations

from polypai.engine.criterion import UnifiedCriterion
from polypai.engine.evaluator import Evaluator
from polypai.engine.trainer import Trainer, TrainConfig

__all__ = ["UnifiedCriterion", "Trainer", "TrainConfig", "Evaluator"]
