"""Task heads: anchor-free detection, boundary-aware segmentation,
classification group, and uncertainty/explainability."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from polypai.models.blocks import ConvBNAct
from polypai.models.config import ModelConfig


class Scale(nn.Module):
    def __init__(self, init=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x):
        return x * self.scale


class FCOSDetHead(nn.Module):
    """Anchor-free detector head (FCOS-style): per-level cls / centerness / box.

    Box regression is in FCOS l,t,r,b form with a per-level learnable scale, and
    a P2 level is included so ~2 mm lesions still occupy several stride-4 cells.
    """

    def __init__(self, cfg: ModelConfig, levels: list[str]):
        super().__init__()
        c = cfg.fpn_channels
        self.levels = levels
        self.cls_tower = self._tower(c, cfg.det_head_convs)
        self.reg_tower = self._tower(c, cfg.det_head_convs)
        self.cls = nn.Conv2d(c, cfg.det_classes, 3, padding=1)
        self.reg = nn.Conv2d(c, 4, 3, padding=1)
        self.ctr = nn.Conv2d(c, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in levels])
        nn.init.constant_(self.cls.bias, -4.595)  # prior prob 0.01 for focal stability

    @staticmethod
    def _tower(c, n):
        return nn.Sequential(*[ConvBNAct(c, c, 3, 1) for _ in range(n)])

    def forward(self, feats: dict[str, torch.Tensor]) -> dict[str, list]:
        cls_out, reg_out, ctr_out = [], [], []
        for i, lv in enumerate(self.levels):
            x = feats[lv]
            ct = self.cls_tower(x)
            rt = self.reg_tower(x)
            cls_out.append(self.cls(ct))
            ctr_out.append(self.ctr(ct))
            reg_out.append(F.relu(self.scales[i](self.reg(rt))))  # positive distances
        return {"cls": cls_out, "reg": reg_out, "centerness": ctr_out, "levels": self.levels}


class BoundaryAwareSegHead(nn.Module):
    """Semantic-FPN decoder with an auxiliary boundary branch (Module 3).

    The boundary branch supervises edges explicitly (paired with the boundary/HD
    losses) and its features are concatenated back into the mask predictor, which
    sharpens the segmentation contour under low illumination / partial occlusion.
    """

    def __init__(self, cfg: ModelConfig, levels: list[str]):
        super().__init__()
        c = cfg.fpn_channels
        d = cfg.seg_decoder_channels
        self.levels = levels
        self.project = nn.ModuleDict({lv: ConvBNAct(c, d, 3, 1) for lv in levels})
        self.boundary = nn.Sequential(ConvBNAct(d, d, 3, 1), nn.Conv2d(d, 1, 1))
        self.fuse = ConvBNAct(d + 1, d, 3, 1)
        self.classifier = nn.Conv2d(d, cfg.seg_classes, 1)

    def forward(self, feats: dict[str, torch.Tensor], out_size=None) -> dict[str, torch.Tensor]:
        target = feats[self.levels[0]].shape[-2:]  # P2 resolution
        fused = None
        for lv in self.levels:
            p = self.project[lv](feats[lv])
            p = F.interpolate(p, size=target, mode="bilinear", align_corners=False)
            fused = p if fused is None else fused + p
        boundary = self.boundary(fused)
        seg_feat = self.fuse(torch.cat([fused, boundary], dim=1))
        logits = self.classifier(seg_feat)
        if out_size is not None:
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
            boundary = F.interpolate(boundary, size=out_size, mode="bilinear", align_corners=False)
        return {"seg_logits": logits, "boundary_logits": boundary, "seg_feat": seg_feat}


class ClassificationHeads(nn.Module):
    """Per-characterisation-task linear heads on pooled pyramid features.

    Uses both the deepest (semantic, P5) and shallowest (textural, P2) global
    descriptors so Paris (shape) and NICE/Kudo (fine texture/vessels) each get
    the cues they need.
    """

    def __init__(self, cfg: ModelConfig, channels: int):
        super().__init__()
        self.heads = nn.ModuleDict(
            {name: nn.Sequential(nn.Linear(channels * 2, channels), nn.SiLU(inplace=True),
                                 nn.Dropout(0.2), nn.Linear(channels, n))
             for name, n in cfg.cls_heads.items()}
        )

    def forward(self, feats: dict[str, torch.Tensor], task_feats: dict | None = None) -> dict:
        deep = feats["p5"].mean((2, 3))
        shallow = feats["p2"].mean((2, 3))
        out = {}
        for name, head in self.heads.items():
            if task_feats is not None and name in task_feats:
                tf = task_feats[name]
                desc = torch.cat([tf["p5"].mean((2, 3)), tf["p2"].mean((2, 3))], dim=1)
            else:
                desc = torch.cat([deep, shallow], dim=1)
            out[name] = head(desc)
        return out


class UncertaintyHead(nn.Module):
    """Predicts a per-pixel segmentation log-variance map (aleatoric uncertainty)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.seg_decoder_channels
        self.net = nn.Sequential(ConvBNAct(d, d // 2, 3, 1), nn.Conv2d(d // 2, 1, 1))

    def forward(self, seg_feat: torch.Tensor) -> torch.Tensor:
        return self.net(seg_feat)  # log-variance
