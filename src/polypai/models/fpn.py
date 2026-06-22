"""Feature-pyramid neck with top-down fusion and learnable level weights.

A BiFPN-style weighted fusion (Tan et al., 2020) is used so the network can
adaptively emphasise the high-resolution P2 level for tiny lesions — the
"adaptive feature pyramid" / "multi-scale feature fusion" requirement of the
small-polyp detection module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from polypai.models.blocks import ConvBNAct
from polypai.models.config import ModelConfig


class FPN(nn.Module):
    def __init__(self, cfg: ModelConfig, in_channels: dict[str, int]):
        super().__init__()
        c = cfg.fpn_channels
        self.levels = ["p2", "p3", "p4", "p5"] if cfg.extra_p2 else ["p3", "p4", "p5"]
        self.lateral = nn.ModuleDict(
            {lv: nn.Conv2d(in_channels[lv], c, 1) for lv in self.levels}
        )
        self.output = nn.ModuleDict({lv: ConvBNAct(c, c, 3, 1) for lv in self.levels})
        # learnable normalised fusion weights for the top-down path
        self.fuse_w = nn.ParameterDict(
            {lv: nn.Parameter(torch.ones(2)) for lv in self.levels[:-1]}
        )
        self.out_channels = c

    def forward(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        lat = {lv: self.lateral[lv](feats[lv]) for lv in self.levels}
        out = {self.levels[-1]: lat[self.levels[-1]]}
        for i in range(len(self.levels) - 2, -1, -1):
            lv, higher = self.levels[i], self.levels[i + 1]
            up = F.interpolate(out[higher], size=lat[lv].shape[-2:], mode="nearest")
            w = F.relu(self.fuse_w[lv])
            w = w / (w.sum() + 1e-4)
            out[lv] = w[0] * lat[lv] + w[1] * up
        return {lv: self.output[lv](out[lv]) for lv in self.levels}
