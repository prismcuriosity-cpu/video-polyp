"""Shared encoder producing a multi-scale feature pyramid {P2..P5}.

A clean residual CNN (no torchvision dependency, so the architecture is fully
inspectable and exportable).  Strides 4/8/16/32 are returned; the high-resolution
P2 (stride 4) is what gives the detector its 2 mm sensitivity.  Optional
gradient checkpointing trades compute for memory on the RTX 5090.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

from polypai.models.blocks import ConvBNAct, ResidualBlock
from polypai.models.config import ModelConfig


class PolypEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        chs = cfg.scaled_stage_channels
        stem = int(round(cfg.stem_channels * cfg.width_mult))
        # stem: stride 2 (-> /2), then stage1 stride 2 (-> /4 = P2)
        self.stem = nn.Sequential(
            ConvBNAct(cfg.in_channels, stem, 3, 2),
            ConvBNAct(stem, stem, 3, 1),
        )
        self.stages = nn.ModuleList()
        cin = stem
        for i, (cout, depth) in enumerate(zip(chs, cfg.stage_depths)):
            blocks = [ResidualBlock(cin, cout, stride=2, use_cbam=cfg.use_cbam)]
            for _ in range(depth - 1):
                blocks.append(ResidualBlock(cout, cout, stride=1, use_cbam=cfg.use_cbam))
            self.stages.append(nn.Sequential(*blocks))
            cin = cout
        self.out_channels = {f"p{i + 2}": chs[i] for i in range(len(chs))}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        feats = {}
        for i, stage in enumerate(self.stages):
            if self.cfg.gradient_checkpointing and self.training:
                x = checkpoint_sequential(stage, len(stage), x, use_reentrant=False)
            else:
                x = stage(x)
            feats[f"p{i + 2}"] = x  # p2:/4, p3:/8, p4:/16, p5:/32
        return feats
