"""UnifiedPolypModel: shared encoder -> FPN -> {detection, segmentation,
classification, uncertainty} with cross-task attention fusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from polypai.models.attention import CrossTaskAttention
from polypai.models.backbone import PolypEncoder
from polypai.models.config import ModelConfig
from polypai.models.fpn import FPN
from polypai.models.heads import (
    BoundaryAwareSegHead,
    ClassificationHeads,
    FCOSDetHead,
    UncertaintyHead,
)


class UnifiedPolypModel(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.encoder = PolypEncoder(self.cfg)
        self.neck = FPN(self.cfg, self.encoder.out_channels)
        self.levels = self.neck.levels
        self.det_head = FCOSDetHead(self.cfg, self.levels)
        self.seg_head = BoundaryAwareSegHead(self.cfg, self.levels)
        self.cls_task_names = list(self.cfg.cls_heads.keys())
        self.cross_task = CrossTaskAttention(self.neck.out_channels, self.cls_task_names)
        self.cls_heads = ClassificationHeads(self.cfg, self.neck.out_channels)
        self.unc_head = UncertaintyHead(self.cfg) if self.cfg.enable_uncertainty else None

    def forward(self, x: torch.Tensor, tasks: list[str] | None = None) -> dict:
        tasks = tasks or ["detection", "segmentation", "classification"]
        feats = self.encoder(x)
        pyr = self.neck(feats)
        out: dict = {"feature_levels": self.levels}

        if "detection" in tasks:
            out["detection"] = self.det_head(pyr)

        if "segmentation" in tasks:
            seg = self.seg_head(pyr, out_size=x.shape[-2:])
            out["segmentation"] = seg
            if self.unc_head is not None:
                out["uncertainty"] = self.unc_head(seg["seg_feat"])

        if "classification" in tasks:
            gated = self.cross_task(pyr["p5"])
            task_feats = {name: {"p5": gated[name], "p2": pyr["p2"]} for name in self.cls_task_names}
            out["classification"] = self.cls_heads(pyr, task_feats)

        return out

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: ModelConfig | dict | None = None) -> UnifiedPolypModel:
    if isinstance(cfg, dict):
        cfg = ModelConfig(**cfg)
    return UnifiedPolypModel(cfg)
