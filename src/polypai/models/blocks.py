"""Reusable network blocks: ConvBNAct, residual, SE/CBAM attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, p=None, groups=1, act=True):
        if p is None:
            p = k // 2
        layers = [
            nn.Conv2d(cin, cout, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class SEModule(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        s = F.silu(self.fc1(s))
        return x * torch.sigmoid(self.fc2(s))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial).

    The spatial branch is the "fine-grained lesion attention" called for in the
    small-polyp detection requirements: it lets the network emphasise tiny,
    low-contrast focal regions.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.SiLU(inplace=True), nn.Conv2d(hidden, channels, 1)
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(x.mean((2, 3), keepdim=True)) + self.mlp(x.amax((2, 3), keepdim=True)))
        x = x * ca
        sa = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        x = x * torch.sigmoid(self.spatial(sa))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, use_cbam=True):
        super().__init__()
        self.conv1 = ConvBNAct(cin, cout, 3, stride)
        self.conv2 = ConvBNAct(cout, cout, 3, 1, act=False)
        self.attn = CBAM(cout) if use_cbam else nn.Identity()
        self.act = nn.SiLU(inplace=True)
        self.down = (
            ConvBNAct(cin, cout, 1, stride, act=False)
            if (stride != 1 or cin != cout)
            else nn.Identity()
        )

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        out = self.attn(out)
        return self.act(out + self.down(x))
