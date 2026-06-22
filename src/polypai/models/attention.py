"""Cross-task attention fusion.

Each task reads the shared pyramid through its own learned query, producing a
task-conditioned channel modulation of the features.  This lets, e.g., the NICE
head emphasise vascular-frequency channels while the segmentation head keeps
boundary channels — the "cross-task attention fusion" research addition — without
fully decoupling the representations (the encoder stays shared).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossTaskAttention(nn.Module):
    def __init__(self, channels: int, task_names: list[str], embed_dim: int = 64):
        super().__init__()
        self.task_names = list(task_names)
        self.task_embed = nn.Parameter(torch.randn(len(task_names), embed_dim) * 0.02)
        self.gate = nn.Sequential(
            nn.Linear(channels + embed_dim, channels), nn.SiLU(inplace=True),
            nn.Linear(channels, channels),
        )
        self.channels = channels

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return per-task channel-gated copies of ``feat`` (B,C,H,W)."""
        g = feat.mean((2, 3))  # global context (B, C)
        out = {}
        for i, name in enumerate(self.task_names):
            emb = self.task_embed[i].unsqueeze(0).expand(g.shape[0], -1)
            gate = torch.sigmoid(self.gate(torch.cat([g, emb], dim=1)))
            out[name] = feat * gate.unsqueeze(-1).unsqueeze(-1)
        return out
